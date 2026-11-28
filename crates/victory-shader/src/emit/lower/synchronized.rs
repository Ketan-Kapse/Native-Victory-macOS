//! Multi-wave loops without a guessed pair barrier. A wave keeps its console PC;
//! all threads visit components in the same order, and only communicating
//! components require a group vote for each traversal.
//! `bitcode::synchronized` carries live values between visits, including EXEC/SCC.
//! Resource accesses use the same guards as acyclic if-conversion. The original
//! loop counters and their limit remain part of each wave's private state.
use super::*;

/// Instruction positions, by block: a fence goes before each.
type Sites = Set<(usize, usize)>;

impl Lower<'_> {
    pub(super) fn lower_synchronized(&mut self, order: &[usize]) -> Result<(), Stop> {
        // Do not infer uniform barrier participation from wave-uniform SCC.
        // These barriers must be outside cycles and dominate every original exit.
        let once = once_blocks(self.x.p);
        let mut title_barriers = 0;
        for &b in order {
            for i in &self.x.p.blocks[b].insts {
                if i.name() == "s_barrier" {
                    title_barriers += 1;
                    if !once[b] {
                        return Err(unsupported(i.machine.offset, UNPROVED_TITLE_BARRIER));
                    }
                }
            }
        }
        if let Some((offset, why)) =
            super::super::lockstep::split_branch(self.x.p, self.x.st, &self.x.res.waterfall.reads)
        {
            return Err(unsupported(offset, why));
        }
        self.schedule_has_barriers = title_barriers != 0;
        if std::env::var_os("VICTORY_TRACE_CONSOLE_FLOW").is_some() {
            let mut lds = [0usize; 2];
            for &b in order {
                let mut scheduled = b != order[0];
                for i in &self.x.p.blocks[b].insts {
                    scheduled |= i.name() == "s_barrier";
                    if matches!(i.fields, Fields::Ds { .. })
                        && !matches!(i.name(), "ds_swizzle_b32" | "ds_bpermute_b32")
                    {
                        lds[usize::from(scheduled)] += 1;
                    }
                }
            }
            eprintln!(
                "console schedule: {title_barriers} title barriers, {} prologue LDS sites, {} scheduled LDS sites",
                lds[0], lds[1]
            );
        }
        (self.split_barriers, self.order_fences) = self.lds_phase_barriers();
        self.lower_blocks(order, self.s.entry)?;
        self.fill_phis()?;
        if self.communicating {
            let halves = self.s.threads.unwrap().iter().product::<u32>().div_ceil(64) * 2;
            self.s.m.batch_exchanges(halves, 8);
        }
        // Entry is a one-time prologue; it owns input/descriptor handles. Each other
        // region is one console block plus its acyclic, instruction-local guards.
        let mut regions = std::mem::take(&mut self.schedule_regions);
        let mut edges: Vec<_> = self.edges.values().copied().collect();
        edges.sort_unstable();
        edges.dedup();
        regions.extend(edges.into_iter().map(|b| (b, b, None)));
        if let Some(b) = self.abort {
            regions.push((b, b, None));
        }
        let schedules = self
            .s
            .m
            .synchronize_regions(
                &regions,
                (title_barriers != 0).then_some(self.abort).flatten(),
            )
            .map_err(|w| unsupported(0, w))?;
        for schedule in &schedules {
            if !schedule.collective {
                continue;
            }
            let (any, aborted) =
                self.group_vote(schedule.header, schedule.active, schedule.aborted);
            let next = if let Some(aborted) = aborted {
                // A failed producer must cancel the group before a later phase.
                let next = self.s.m.new_block();
                let stop = self.s.m.new_block();
                self.s.m.terminate(stop, Term::Ret);
                self.s
                    .m
                    .terminate(schedule.header, Term::CondBr(aborted, stop, next));
                next
            } else {
                schedule.header
            };
            self.s
                .m
                .terminate(next, Term::CondBr(any, schedule.body, schedule.end));
        }
        if self.communicating {
            let fences = self.s.m.elide_order_fences();
            let removed = self.s.m.elide_exchange_releases();
            if std::env::var_os("VICTORY_TRACE_CONSOLE_FLOW").is_some() {
                eprintln!("console schedule: {removed} scratch release barriers and {fences} atomic order fences replaced by existing rendezvous");
            }
        }
        // Cutting every generated backedge leaves an acyclic collective
        // schedule. Local backedges contain no barrier; collective ones vote.
        if !self.s.m.schedule_barriers_are_uniform(&schedules) || !self.s.m.values_dominate_uses() {
            return Err(unsupported(
                0,
                "a synchronized loop failed its barrier or SSA proof",
            ));
        }
        self.s.m.trace_schedule_costs(&schedules);
        Ok(())
    }

    /// The mandatory, noncyclic phase is a singleton in the component DAG.
    /// All preceding components finish before any thread visits it, including
    /// the group cancellation vote on loop-limit exits. No arrival poll is
    /// necessary; every wave's PC is here by the original once-block proof.
    pub(super) fn schedule_barrier(&mut self) -> usize {
        let wait = self.s.m.new_block();
        let next = self.s.m.new_block();
        self.s.m.terminate(self.cur, Term::Br(wait));
        self.cur = wait;
        let on = self.s.m.phi(wait, self.s.t.i1);
        self.group_barrier();
        self.s.m.terminate(wait, Term::Br(next));
        self.schedule_regions.push((wait, wait, Some(on)));
        self.cur = next;
        next
    }
}

impl Lower<'_> {
    /// Every two LDS accesses one console wave makes in program order keep a
    /// group barrier between them, as on the console where all 64 lanes finish
    /// one instruction before any starts the next. The exceptions commute, so
    /// no interleaving of the two host halves changes a value: a run of reads,
    /// and a run of stores of the same proved constant (even if their addresses
    /// alias). A run has one fence before and one after; any other write has
    /// both. An atomic needs only the fence before it, and none where the
    /// previous access in its block already left one; a title barrier is a
    /// fence itself. A block's first access is fenced in any case, so nothing
    /// is assumed about the block the wave came from.
    /// Also returns the atomics' own fences: the only ones another barrier
    /// can stand in for (`elide_order_fences`, on the emitted schedule).
    fn lds_phase_barriers(&self) -> (Sites, Sites) {
        let mut fences = Set::default();
        let mut atomics = Set::default();
        for (b, block) in self.x.p.blocks.iter().enumerate() {
            // None = reads, Some(value) = stores of one uniform constant.
            let mut run: Option<(usize, usize, Option<u32>)> = None;
            // A fence already follows the block's previous access.
            let mut fenced = false;
            for (i, inst) in block.insts.iter().enumerate() {
                let name = inst.name();
                let lds = matches!(inst.fields, Fields::Ds { .. })
                    && !matches!(name, "ds_swizzle_b32" | "ds_bpermute_b32");
                if !lds && name != "s_barrier" { continue; }
                let kind = if name.starts_with("ds_read") { Some(None) }
                    else { self.uniform_lds_store(b, i).map(Some) };
                if let Some((first, _, previous)) = run
                    && kind == Some(previous)
                {
                    run = Some((first, i, previous));
                    continue;
                }
                if let Some((first, last, _)) = run.take() {
                    fences.extend([(b, first), (b, last + 1)]);
                    fenced = true;
                }
                if let Some(kind) = kind { run = Some((i, i, kind)); }
                else if name == "s_barrier" { fenced = true; }
                else if name.starts_with("ds_write") {
                    fences.extend([(b, i), (b, i + 1)]);
                    fenced = true;
                } else {
                    if !fenced {
                        fences.insert((b, i));
                        atomics.insert((b, i));
                    }
                    fenced = false;
                }
            }
            if let Some((first, last, _)) = run {
                fences.extend([(b, first), (b, last + 1)]);
            }
            // The title fence supplies the same ordering at its boundary.
            fences.retain(|&(q, i)| q != b || !block.insts.get(i).is_some_and(|x| x.name() == "s_barrier")
                && (i == 0 || block.insts[i - 1].name() != "s_barrier"));
        }
        atomics.retain(|at| fences.contains(at));
        (fences, atomics)
    }

    /// All dwords of this write are one immediate, defined in this block under
    /// an EXEC containing the store's EXEC. Never infer this from an inactive
    /// lane's old VGPR, a phi or a value from a different scheduled region.
    fn uniform_lds_store(&self, b: usize, i: usize) -> Option<u32> {
        let block = &self.x.p.blocks[b];
        let inst = &block.insts[i];
        let Fields::Ds { data, .. } = &inst.fields else { return None };
        if !inst.name().starts_with("ds_write") { return None; }
        let exec = self.x.st.execs[b][i].unwrap_or(Bool::TRUE);
        let mut constant = None;
        for operand in data {
            let Operand::Vector { reg, dwords } = *operand else { return None };
            for reg in reg..reg + dwords {
                let value = inst.read(Reg::V(reg))?;
                let (j, def) = block.insts[..i].iter().enumerate()
                    .find(|(_, x)| x.writes.iter().any(|w| w.1 == value))?;
                // A title barrier splits this block into separate PC regions.
                if block.insts[j..i].iter().any(|x| x.name() == "s_barrier") { return None; }
                let Fields::Valu(v) = &def.fields else { return None };
                if def.name() != "v_mov_b32" || v.dpp.is_some() || v.sdwa.is_some() { return None; }
                let x = scalar_constant(*v.src.first()?)?;
                let defined = self.x.st.execs[b][j].unwrap_or(Bool::TRUE);
                if self.x.st.bools.meet(&[(exec, true), (defined, false)]) { return None; }
                if constant.is_some_and(|old| old != x) { return None; }
                constant = Some(x);
            }
        }
        constant
    }
}
