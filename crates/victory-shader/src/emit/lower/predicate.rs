//! If-conversion of acyclic multi-wave compute. Every thread visits every IR block in
//! topological order. Edges carry a *console-wave* predicate; joins select on those edges.
//! Vector effects additionally keep EXEC. Exchange barriers are outside both predicates.
//!
//! Cyclic programs use the synchronized schedule in `synchronized.rs`; the instruction
//! guards are shared. Memory communication uses the synchronized path, including
//! acyclic programs. Both are fallbacks after a divergent-exchange refusal.
use super::*;

#[derive(Default)]
pub(super) struct Predication {
    edges: Map<(usize, usize), V>,
    on: Option<V>,
}

pub(super) fn cyclic(p: &Program) -> bool {
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut rank = vec![usize::MAX; p.blocks.len()];
    for (i, &b) in order.iter().enumerate() {
        rank[b] = i;
    }
    order.iter().any(|&b| {
        p.blocks[b]
            .exit
            .successors()
            .iter()
            .any(|&q| rank[q] <= rank[b])
    })
}

pub(super) fn scheduled(p: &Program) -> bool {
    cyclic(p)
        || p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| lds_memory(i) || i.name() == "s_barrier" || i.name().contains("atomic"))
}

fn lds_memory(i: &Inst) -> bool {
    matches!(i.fields, Fields::Ds { .. })
        && !matches!(i.name(), "ds_swizzle_b32" | "ds_bpermute_b32")
}

/// Classify by the earliest round which could admit this candidate. In particular,
/// round 2 already admitted cyclic buffer/global add, and round 3 must not silently
/// re-enable a disabled round 1 or 2. The normal successful path runs before this.
pub(super) fn enabled(x: &Parts, o: super::super::Options) -> bool {
    let loops = cyclic(x.p);
    let communicating = communicating(x);
    if communicating { !o.no_comm_waves }
    else if loops { !o.no_sync_loops }
    else { !o.no_if_convert }
}

pub(super) fn communicating(x: &Parts) -> bool {
    let StageIo::Cs(cs) = x.native.io else { return false };
    cs.lds_bytes() != 0 || shares(x)
}

/// The programs that take the cheaper communicating schedule. An LDS
/// allocation no instruction uses is still gated by the round 3 switch
/// (`communicating`), but it must not change the emitted program.
pub(super) fn shares(x: &Parts) -> bool {
    let StageIo::Cs(cs) = x.native.io else { return false };
    let loops = cyclic(x.p);
    cs.threads.iter().product::<u32>() > 256
        || x.p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
            lds_memory(i) || i.name() == "s_barrier"
                || i.name().contains("atomic")
                    && !(loops && matches!(i.name(), "buffer_atomic_add" | "global_atomic_add"))
        })
}

pub(super) fn eligible(x: &Parts) -> bool {
    let StageIo::Cs(cs) = x.native.io else {
        return false;
    };
    let lanes = cs.threads.iter().product::<u32>();
    if !x.wave32_host
        || !x.native.wave64
        || !(65..=1024).contains(&lanes)
        || lanes > 256 && !scheduled(x.p)
        || !x.res.waterfall.reads.is_empty()
    {
        return false;
    }
    // `emit_fast` already retries by keeping lanes at a bare end with EXEC off. Let that
    // smaller transformation run first; two corpus programs already translate that way.
    if crate::ir::keep_lanes_to_end(x.p).is_some() {
        return false;
    }
    let order = crate::ir::reverse_postorder(&x.p.blocks);
    let once = once_blocks(x.p);
    let loops = cyclic(x.p);
    order.iter().all(|&b| {
        let block = &x.p.blocks[b];
        !matches!(block.exit, Exit::Indirect)
            && block.insts.iter().all(|i| match &i.fields {
                // Preserve mode writes already reached by every wave. The existing round-mode
                // analysis still owns them; do not speculate one from a skipped arm.
                Fields::Sop { .. } if i.name().starts_with("s_setreg") => once[b],
                // The decoder already proves PC-relative far jumps. Their register
                // arithmetic is not a new control decision; the IR owns the target.
                Fields::Sop { .. } if loops && i.name() == "s_setpc_b64" => {
                    matches!(block.exit, Exit::Jump(_))
                        && block
                            .insts
                            .last()
                            .is_some_and(|last| last.machine.offset == i.machine.offset)
                }
                Fields::Sop { .. } => {
                    !i.name().starts_with("s_set") && !i.name().starts_with("s_swap")
                }
                Fields::Sopp { .. } => matches!(
                    i.name(),
                    "s_branch"
                        | "s_cbranch_scc0"
                        | "s_cbranch_scc1"
                        | "s_cbranch_execz"
                        | "s_cbranch_execnz"
                        | "s_cbranch_vccz"
                        | "s_cbranch_vccnz"
                        | "s_endpgm"
                        | "s_barrier"
                ),
                Fields::Smem { .. } => {
                    i.name().starts_with("s_load_") || i.name().starts_with("s_buffer_load_")
                }
                Fields::Ds { gds: false, .. } => matches!(
                    i.name(),
                    "ds_swizzle_b32"
                        | "ds_bpermute_b32"
                        | "ds_read_b32"
                        | "ds_read_b64"
                        | "ds_read_b96"
                        | "ds_read_b128"
                        | "ds_read2_b32"
                        | "ds_read2_b64"
                        | "ds_read2st64_b32"
                        | "ds_read2st64_b64"
                        | "ds_write_b32"
                        | "ds_write_b64"
                        | "ds_write_b96"
                        | "ds_write_b128"
                        | "ds_write2_b32"
                        | "ds_write2_b64"
                        | "ds_write2st64_b32"
                        | "ds_write2st64_b64"
                        | "ds_add_u32"
                        | "ds_add_rtn_u32"
                        | "ds_sub_u32"
                        | "ds_sub_rtn_u32"
                        | "ds_and_b32"
                        | "ds_and_rtn_b32"
                        | "ds_or_b32"
                        | "ds_or_rtn_b32"
                        | "ds_xor_b32"
                        | "ds_xor_rtn_b32"
                        | "ds_wrxchg_rtn_b32"
                        | "ds_min_i32"
                        | "ds_min_rtn_i32"
                        | "ds_max_i32"
                        | "ds_max_rtn_i32"
                        | "ds_min_u32"
                        | "ds_min_rtn_u32"
                        | "ds_max_u32"
                        | "ds_max_rtn_u32"
                        | "ds_cmpst_b32"
                        | "ds_cmpst_rtn_b32"
                ),
                Fields::Ds { .. } => false,
                Fields::Exp { .. } | Fields::Vintrp { .. } => false,
                // The scheduler retains each wave's original order. These effects
                // stay under both wave reach and EXEC, including the returned value.
                Fields::Buffer { .. } | Fields::Flat { .. }
                    if matches!(
                        i.name(),
                        "buffer_atomic_add"
                            | "global_atomic_add"
                            | "buffer_atomic_umin"
                            | "global_atomic_umin"
                            | "buffer_atomic_cmpswap"
                            | "global_atomic_cmpswap"
                    ) =>
                {
                    true
                }
                _ => !i.name().contains("atomic"),
            })
    })
}

impl Lower<'_> {
    pub(super) fn lower_predicated(&mut self, order: &[usize]) -> Result<(), Stop> {
        // A per-lane scalar (notably a waterfall descriptor) cannot choose one console PC.
        if let Some((offset, what)) =
            super::super::lockstep::split_branch(self.x.p, self.x.st, &self.x.res.waterfall.reads)
        {
            return Err(unsupported(offset, what));
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        self.cur = self.s.entry;
        self.inputs()?;
        if self.x.p.thread_counts {
            self.launched();
        }
        for (n, &b) in order.iter().enumerate() {
            self.cur_ir = b;
            self.cur = self.block[b].expect("reachable");
            let mut on = self.s.m.bool(b == 0);
            for &q in &self.x.p.blocks[b].preds {
                if let Some(&edge) = self.predication.as_ref().unwrap().edges.get(&(q, b)) {
                    on = self.s.m.bin(self.cur, Bin::Or, on, edge);
                }
            }
            self.predication.as_mut().unwrap().on = Some(on);
            self.mode16 = self.round16[b];
            for i in 0..self.x.p.blocks[b].insts.len() {
                self.predicate_inst(b, i)?;
                self.mode16 = round16_after(&self.x.p.blocks[b].insts[i], self.mode16);
            }
            self.mask_ok = true;
            self.tail[b] = Some(self.cur);
            match self.x.p.blocks[b].exit {
                Exit::Jump(q) => {
                    self.predication.as_mut().unwrap().edges.insert((b, q), on);
                }
                Exit::Branch { taken, not_taken } => {
                    let offset = self.x.p.blocks[b].insts.last().unwrap().machine.offset;
                    let bit = self.console_branch(b).map_err(|w| unsupported(offset, w))?;
                    let yes = self.s.m.bin(self.cur, Bin::And, on, bit);
                    let t = self.s.m.bool(true);
                    let no = self.s.m.bin(self.cur, Bin::Xor, bit, t);
                    let no = self.s.m.bin(self.cur, Bin::And, on, no);
                    let edges = &mut self.predication.as_mut().unwrap().edges;
                    edges.insert((b, taken), yes);
                    edges.insert((b, not_taken), no);
                }
                Exit::End => {}
                Exit::Indirect => unreachable!("eligibility excludes indirect flow"),
            }
            let term = order
                .get(n + 1)
                .map_or(Term::Ret, |&q| Term::Br(self.block[q].unwrap()));
            self.s.m.terminate(self.cur, term);
        }
        if !self.s.m.calls_on_all_paths("dx.op.barrier") {
            return Err(unsupported(0, DIVERGENT_EXCHANGE));
        }
        if !self.s.m.values_dominate_uses() {
            return Err(unsupported(0, "a value outside its predicated definition"));
        }
        Ok(())
    }

    /// Do not speculate a resource access from a wave's skipped arm. Resource operands
    /// can be valid only on that arm. Keep exchanges outside this guard: the final CFG
    /// check refuses any access whose operand lowering needs an exchange inside it.
    pub(super) fn predicate_inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst = &self.x.p.blocks[b].insts[i];
        if self.synchronized && lds_memory(inst) {
            // Both halves finish the previous LDS access before either starts
            // this one. All operand exchanges precede the lane/resource guard.
            let guard = self
                .guard(b, i)
                .map_err(|w| unsupported(inst.machine.offset, w))?;
            let mut old = Vec::new();
            for &(reg, v) in &inst.writes {
                let value = inst
                    .read(reg)
                    .and_then(|v| self.i32_of(v).ok())
                    .unwrap_or_else(|| self.s.m.i32(0));
                old.push((v, value));
            }
            let at = self.begin_guarded(guard);
            self.inst(b, i)?;
            let mut outputs = Vec::new();
            let mut values = Vec::new();
            for (v, outside) in old {
                if let Some(Val::I32(x)) = self.vals.get(&v).copied() {
                    outputs.push(v);
                    values.push((x, outside));
                }
            }
            for (v, x) in outputs.into_iter().zip(self.end_guarded(at, &values)) {
                self.vals.insert(v, Val::I32(x));
            }
            self.floats.clear();
            return Ok(());
        }
        if !matches!(
            inst.fields,
            Fields::Smem { .. }
                | Fields::Image { .. }
                | Fields::Buffer { .. }
                | Fields::Flat { .. }
        ) {
            return self.inst(b, i);
        }
        let on = self.wave_on().expect("a scheduled block");
        if self.s.m.int_value(on) == Some(1) {
            return self.inst(b, i);
        }
        let at = self.begin_guarded(Some(on));
        self.inst(b, i)?;
        let mut outputs = Vec::new();
        let mut values = Vec::new();
        for &(_, v) in &inst.writes {
            match self.vals.get(&v).copied() {
                Some(Val::I32(x)) => {
                    outputs.push((v, false));
                    values.push((x, self.s.m.i32(0)));
                }
                Some(Val::Bit(x)) => {
                    outputs.push((v, true));
                    values.push((x, self.s.m.bool(false)));
                }
                _ => {}
            }
        }
        for ((v, bit), x) in outputs.into_iter().zip(self.end_guarded(at, &values)) {
            self.vals
                .insert(v, if bit { Val::Bit(x) } else { Val::I32(x) });
        }
        // A cast of an earlier register made while forming an address/coordinate belongs
        // to this guard; a later instruction must form its own cast outside it.
        self.floats.clear();
        Ok(())
    }

    /// EXEC for vector work; wave reach alone for instructions which ignore EXEC.
    pub(super) fn predicate_guard(&mut self, guard: Option<V>) -> Option<V> {
        let Some(on) = self.wave_on() else {
            return guard;
        };
        Some(match guard {
            Some(g) if self.synchronized => {
                let no = self.s.m.bool(false);
                self.s.m.select(self.cur, on, g, no)
            }
            Some(g) => self.s.m.bin(self.cur, Bin::And, on, g),
            None => on,
        })
    }

    fn wave_on(&self) -> Option<V> {
        self.predication
            .as_ref()
            .and_then(|p| p.on)
            .or_else(|| self.schedule_on.get(self.cur_ir).copied().flatten())
    }

    /// Select a join's incoming values on the console edges, in its defining block.
    pub(super) fn predicate_select(&mut self, b: usize, incoming: &[(V, usize)]) -> Result<V, Why> {
        let mut out = incoming.first().ok_or("an empty predicated join")?.0;
        let at = self.block[b].ok_or("an unreachable predicated join")?;
        for &(x, q) in incoming {
            let &on = self
                .predication
                .as_ref()
                .unwrap()
                .edges
                .get(&(q, b))
                .ok_or("a predicated join from a later block")?;
            out = self.s.m.select(at, on, x, out);
        }
        Ok(out)
    }

    pub(super) fn predicate_data_phi(&mut self, v: Value) -> Result<Val, Why> {
        let &(b, i) = self
            .phi_at
            .get(&v)
            .ok_or("a predicated value without a join")?;
        let phi = &self.x.p.blocks[b].phis[i];
        let here = self.cur;
        let ir = self.cur_ir;
        self.cur = self.block[b].ok_or("an unreachable predicated join")?;
        self.cur_ir = b;
        let result = (|| {
            let mut incoming = Vec::new();
            for (&q, &a) in self.x.p.blocks[b].preds.iter().zip(&phi.args) {
                if self.block[q].is_none() {
                    continue;
                }
                let x = if phi.reg == Reg::Scc {
                    match self.get(a)? {
                        Val::Bit(x) => x,
                        _ => return Err("an SCC join without a bit"),
                    }
                } else {
                    self.i32_of(a)?
                };
                incoming.push((x, q));
            }
            let x = self.predicate_select(b, &incoming)?;
            Ok(if phi.reg == Reg::Scc {
                Val::Bit(x)
            } else {
                Val::I32(x)
            })
        })();
        (self.cur, self.cur_ir) = (here, ir);
        if let Ok(x) = result {
            self.vals.insert(v, x);
        }
        result
    }

    pub(super) fn predicate_mask_phi(&mut self, v: Value) -> Result<V, Why> {
        let &(b, i) = self
            .phi_at
            .get(&v)
            .ok_or("a lane-mask join without a phi")?;
        let phi = &self.x.p.blocks[b].phis[i];
        let high = self
            .pair_high(v)
            .and_then(|h| self.phi_at.get(&h))
            .map(|&(b, i)| &self.x.p.blocks[b].phis[i]);
        let at = self.block[b].ok_or("an unreachable predicated join")?;
        let (here, ir) = (self.cur, self.cur_ir);
        (self.cur, self.cur_ir) = (at, b);
        let result = (|| {
            let mut incoming = Vec::new();
            for (n, (&q, &a)) in self.x.p.blocks[b].preds.iter().zip(&phi.args).enumerate() {
                if self.block[q].is_none() {
                    continue;
                }
                let bit = if self.unneeded_edges().contains(&(v, n)) {
                    self.s.m.bool(false)
                } else if let Some(&mask) = self.x.st.masks.get(&a) {
                    self.boolean(mask, at)?
                } else if let Some(&mask) = self.x.st.other.get(&a) {
                    self.boolean(mask, at)?
                } else {
                    let lo = self.i32_of(a)?;
                    let hi = high.ok_or("a predicated mask join without its high dword")?;
                    let hi = self.i32_of(hi.args[n])?;
                    self.lane_bit(at, lo, Some(hi))
                };
                incoming.push((bit, q));
            }
            self.predicate_select(b, &incoming)
        })();
        (self.cur, self.cur_ir) = (here, ir);
        let x = result?;
        self.mask_phis.insert(v, x);
        Ok(x)
    }
}
