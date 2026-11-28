//! A console-PC schedule. Each region is an original console block with its
//! instruction guards; only its terminal edge is replaced. Internal guards must
//! be acyclic and reconverge before every wave operation and before the tail.
//!
//! Induction at region boundaries: a wave's PC and previous block are uniform;
//! each live SSA value is the value from that wave's last execution of its defining
//! block. A matching PC selects the original phi edge, executes the block, commits
//! its values and follows its original edge. Other waves retain their state and
//! only participate in exchanges. No resource effect is speculated (the lowerer
//! guards them with the supplied `on` placeholder). Cyclic components run in
//! topological order. Their host backedges are local when exchange-free, and
//! otherwise controlled by a group-wide vote supplied by the caller.
use super::*;

const MAX_ADDED: usize = 65_536;
const MAX_CARRIED: usize = 16_384;

/// One cyclic component of the console CFG. Its test is local when the
/// component has no group communication; otherwise the lowerer supplies a
/// group-uniform continuation vote. All components are visited in DAG order.
#[derive(Debug)]
pub(crate) struct ScheduleLoop {
    pub header: usize,
    pub latch: usize,
    pub end: usize,
    pub body: usize,
    pub active: V,
    pub aborted: Option<V>,
    pub collective: bool,
}

impl Module {
    /// The lowerer replaces only these tests with group votes/cancellation.
    /// For the mandatory-barrier proof take their bodies and cut their backedges.
    /// Local tests enclose no barrier; collective tests are group uniform.
    pub(crate) fn schedule_barriers_are_uniform(&mut self, loops: &[ScheduleLoop]) -> bool {
        let mut saved = Vec::new();
        let valid = (|| {
            for l in loops {
                let mut test = l.header;
                if l.aborted.is_some() {
                    let Some(Term::CondBr(_, stop, next)) = self.blocks[test].term else {
                        return false;
                    };
                    if !matches!(self.blocks[stop].term, Some(Term::Ret)) {
                        return false;
                    }
                    saved.push((test, self.blocks[test].term));
                    self.blocks[test].term = Some(Term::Br(next));
                    test = next;
                }
                let Some(Term::CondBr(_, body, end)) = self.blocks[test].term else {
                    return false;
                };
                if body != l.body || end != l.end {
                    return false;
                }
                saved.push((test, self.blocks[test].term));
                self.blocks[test].term = Some(Term::Br(body));
                if !matches!(self.blocks[l.latch].term, Some(Term::Br(to)) if to == l.header) {
                    return false;
                }
                saved.push((l.latch, self.blocks[l.latch].term));
                self.blocks[l.latch].term = Some(Term::Br(end));
            }
            self.calls_on_all_paths("dx.op.barrier")
        })();
        for (b, term) in saved {
            self.blocks[b].term = term;
        }
        valid
    }

    pub(crate) fn trace_schedule_costs(&self, loops: &[ScheduleLoop]) {
        if std::env::var_os("VICTORY_TRACE_CONSOLE_FLOW").is_none() { return; }
        let live = self.live();
        let mut repeating = std::collections::HashSet::new();
        let mut costs = Vec::new();
        for l in loops {
            let mut seen = std::collections::HashSet::new();
            let mut work = vec![l.header];
            while let Some(b) = work.pop() {
                if b == l.end || !seen.insert(b) { continue; }
                if b != l.latch { work.extend(successors(self.blocks[b].term).iter().copied()); }
            }
            costs.push(self.schedule_cost(seen.iter().flat_map(|&b| self.blocks[b].body.iter().copied()), &live));
            repeating.extend(seen);
        }
        let once = self.schedule_cost(self.blocks.iter().enumerate().filter(|(b, _)| !repeating.contains(b))
            .flat_map(|(_, block)| block.body.iter().copied()), &live);
        eprintln!("console schedule: final once {once:?}; final loops {costs:?} [barriers,tgsm,uav,slot_stores,slot_loads,atomics], including header votes");
    }

    pub(crate) fn synchronize_regions(
        &mut self,
        regions: &[(usize, usize, Option<V>)],
        abort: Option<usize>,
    ) -> Result<Vec<ScheduleLoop>, &'static str> {
        const SHAPE: &str = "a synchronized loop with non-reconverging instruction guards";
        const VALUE: &str = "a synchronized loop value unavailable at its block boundary";
        if regions.len() < 2 || regions.len().saturating_sub(1) > MAX_ADDED / 4 {
            return Err("a synchronized loop beyond the code-growth bound");
        }
        let trace = std::env::var_os("VICTORY_TRACE_CONSOLE_FLOW").is_some();
        if trace {
            let mut entry = std::collections::HashSet::new();
            let mut work = vec![regions[0].0];
            while let Some(b) = work.pop() {
                if !entry.insert(b) || b == regions[0].1 {
                    continue;
                }
                work.extend(successors(self.blocks[b].term).iter().copied());
            }
            let prologue = self.schedule_cost(
                entry
                    .iter()
                    .flat_map(|&b| self.blocks[b].body.iter().copied()),
                &self.live(),
            );
            let cost = self.schedule_cost(
                self.blocks.iter().flat_map(|b| b.body.iter().copied()),
                &self.live(),
            );
            eprintln!(
                "console schedule: original static [barriers,tgsm,uav,slot_stores,slot_loads,atomics] {cost:?} (before latch votes)"
            );
            eprintln!("console schedule: original prologue cost {prologue:?}");
        }
        self.simplify_exchange_inputs();
        let unused = self.discard_unused_exchanges();
        let private = self.private_exchange_barriers();
        let hoisted = self.hoist_exchanges(regions[0].1);
        let live = self.live();
        if std::env::var_os("VICTORY_TRACE_CONSOLE_FLOW").is_some() {
            eprintln!(
                "console schedule: {} regions, {} live instructions, {hoisted} hoisted exchanges, {unused} unused exchanges removed",
                regions.len(),
                live.iter().filter(|&&x| x).count()
            );
        }
        // Every scheduled region adds at least four instructions. Bound the
        // generated work, not pre-existing code that this pass never duplicates.
        if regions.len() < 2 || regions.len().saturating_sub(1) > MAX_ADDED / 4 {
            return Err("a synchronized loop beyond the code-growth bound");
        }
        let old = self.blocks.clone();
        let insts = self.insts.clone();
        let mut owner = vec![usize::MAX; old.len()];
        let mut members = Vec::new();
        let heads: std::collections::HashSet<_> = regions.iter().map(|r| r.0).collect();
        for (r, &(head, tail, _)) in regions.iter().enumerate() {
            let mut stack = vec![head];
            let mut blocks = Vec::new();
            while let Some(b) = stack.pop() {
                if owner[b] == r {
                    continue;
                }
                if owner[b] != usize::MAX || (b != head && heads.contains(&b)) {
                    return Err(SHAPE);
                }
                owner[b] = r;
                blocks.push(b);
                if b != tail {
                    let next = successors(old[b].term);
                    if next.is_empty() {
                        return Err(SHAPE);
                    }
                    stack.extend(next.iter().copied());
                }
            }
            if !blocks.contains(&tail) {
                return Err(SHAPE);
            }
            // Topological order inside the region, and domination of its tail.
            let mut ordered = Vec::new();
            while ordered.len() < blocks.len() {
                let before = ordered.len();
                for &b in &blocks {
                    if ordered.contains(&b) {
                        continue;
                    }
                    if blocks
                        .iter()
                        .filter(|&&p| p != tail && successors(old[p].term).contains(&b))
                        .all(|p| ordered.contains(p))
                    {
                        ordered.push(b);
                    }
                }
                if before == ordered.len() {
                    return Err(SHAPE);
                }
            }
            members.push(ordered);
        }
        // Every terminal edge is another region's entry, never the one-time prologue.
        for &(_, tail, _) in regions {
            for &q in successors(old[tail].term).iter() {
                if !heads.contains(&q) || q == regions[0].0 {
                    return Err(SHAPE);
                }
            }
        }
        let mut def = vec![usize::MAX; insts.len()];
        for (b, block) in old.iter().enumerate() {
            for &i in block.phis.iter().chain(&block.body) {
                def[i as usize] = b;
            }
        }
        if def
            .iter()
            .enumerate()
            .any(|(i, &b)| live[i] && (b == usize::MAX || owner[b] == usize::MAX))
        {
            return Err(SHAPE);
        }
        let placeholders: std::collections::HashSet<_> =
            regions.iter().filter_map(|r| r.2).collect();
        let mut carry = std::collections::BTreeSet::new();
        let mut boundary = std::collections::BTreeSet::new();
        for (r, blocks) in members.iter().enumerate() {
            for &b in blocks {
                let mut need = |v: V, phi: bool| {
                    if let V::Inst(i) = v {
                        if placeholders.contains(&v) {
                            return;
                        }
                        let from = owner[def[i as usize]];
                        if from != r || phi {
                            boundary.insert(i);
                            if from != 0 {
                                carry.insert(i);
                            }
                        }
                    }
                };
                for &i in old[b]
                    .phis
                    .iter()
                    .chain(&old[b].body)
                    .filter(|&&i| live[i as usize])
                {
                    for_each_operand(&insts[i as usize].op, |v| {
                        need(
                            v,
                            b == regions[r].0 && matches!(insts[i as usize].op, Op::Phi(_)),
                        )
                    });
                }
                if let Some(Term::CondBr(c, _, _)) = old[b].term {
                    need(c, false);
                }
            }
        }
        if carry.len() > MAX_CARRIED {
            return Err("a synchronized loop beyond the live-state bound");
        }
        // Region-local dominance also proves that a wave op never sits under a
        // lane/resource guard. This is stronger than checking barriers alone.
        for (r, blocks) in members.iter().enumerate() {
            let (head, tail, _) = regions[r];
            let mut dom: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
            for &b in blocks {
                let preds: Vec<_> = blocks
                    .iter()
                    .copied()
                    .filter(|&p| p != tail && successors(old[p].term).contains(&b))
                    .collect();
                let mut d = preds.first().map(|p| dom[p].clone()).unwrap_or_default();
                for p in preds.iter().skip(1) {
                    d.retain(|x| dom[p].contains(x));
                }
                d.insert(b);
                dom.insert(b, d);
            }
            if !dom[&tail].contains(&head) {
                return Err(SHAPE);
            }
            for &b in blocks {
                let required = old[b].body.iter().any(|&i| live[i as usize] && matches!(
                    &insts[i as usize].op, Op::Call(f, _) if self.funcs[*f as usize].name == "dx.op.barrier"
                    || self.funcs[*f as usize].name.starts_with("dx.op.wave")
                    || self.funcs[*f as usize].name.starts_with("dx.op.quad")
                ));
                if required && !dom[&tail].contains(&b) {
                    return Err("a synchronized loop exchange inside an instruction guard");
                }
                if old[b]
                    .phis
                    .iter()
                    .chain(&old[b].body)
                    .any(|i| boundary.contains(i))
                    && !dom[&tail].contains(&b)
                {
                    return Err(VALUE);
                }
            }
        }
        // The condensation of the console CFG is a DAG. Finish one cyclic
        // component before visiting the next: no unrelated arm, later loop or
        // title phase is replayed by its backedge. An exchange-free component
        // needs no rendezvous, even when different console waves take different
        // trip counts. The branch-uniformity proof in the lowerer still applies.
        let graph: Vec<Vec<usize>> = regions
            .iter()
            .map(|&(_, tail, _)| {
                successors(old[tail].term)
                    .iter()
                    .map(|&q| owner[q])
                    .collect()
            })
            .collect();
        let components = components(&graph);
        let collective: Vec<bool> = members.iter().map(|blocks| blocks.iter().any(|&b| {
            old[b].body.iter().any(|&i| live[i as usize] && matches!(
                &insts[i as usize].op, Op::Call(f, _) if self.funcs[*f as usize].name == "dx.op.barrier"
            ))
        })).collect();
        let mut state = HashMap::new();
        let saved: Vec<_> = carry.into_iter().collect();
        for &i in &saved {
            let ty = insts[i as usize].ty;
            let zero = match self.types[ty.0 as usize] {
                TypeDef::Int(_) => self.int_const(ty, 0),
                TypeDef::Float => self.f32(0),
                _ => return Err("a synchronized loop carries a non-numeric value"),
            };
            state.insert(i, zero);
        }
        let target = |m: &mut Module, at: usize, term: Option<Term>| -> V {
            match term.unwrap_or(Term::Ret) {
                Term::Ret => m.i32(u32::MAX),
                Term::Br(q) => m.i32(q as u32),
                Term::CondBr(c, t, f) => {
                    let (t, f) = (m.i32(t as u32), m.i32(f as u32));
                    m.select(at, c, t, f)
                }
            }
        };
        let mut at = regions[0].1;
        let mut pc = target(self, at, old[at].term);
        let mut prev = self.i32(at as u32);
        let i32t = self.int(32);
        let mut loops = Vec::new();
        let mut loop_costs = Vec::new();
        let mut loop_ops = Vec::new();
        let mut costs = Vec::new();
        let mut once_cost = self.schedule_cost(
            members[0].iter().flat_map(|&b| old[b].body.iter().copied()),
            &live,
        );
        for component in components {
            if component.contains(&0) {
                if component.len() != 1 {
                    return Err(SHAPE);
                }
                continue;
            }
            let cyclic = component.len() > 1 || graph[component[0]].contains(&component[0]);
            let together = component.iter().any(|&r| collective[r]) || abort.is_some();
            let before_component = at;
            let initial_pc = pc;
            let initial_prev = prev;
            let mut carried = Vec::new();
            let mut header = at;
            if cyclic {
                header = self.new_block();
                self.terminate(at, Term::Br(header));
                pc = self.phi(header, i32t);
                prev = self.phi(header, i32t);
                for &i in &saved {
                    if component.contains(&owner[def[i as usize]]) {
                        let initial = state[&i];
                        let phi = self.phi(header, insts[i as usize].ty);
                        state.insert(i, phi);
                        carried.push((i, phi, initial));
                    }
                }
                at = header;
            }
            let (pc_phi, prev_phi) = (pc, prev);
            for &r in &component {
                let (head, tail, placeholder) = regions[r];
                // An inactive PC can skip this entire body iff it contains no
                // group rendezvous. This predicate is uniform within each host
                // wave, so its wave/quad intrinsics still have all their lanes.
                let skip = !collective[r];
                let gate = if skip { self.new_block() } else { head };
                self.terminate(at, Term::Br(gate));
                let id = self.i32(head as u32);
                let body = std::mem::take(&mut self.blocks[head].body);
                let on = self.cmp(gate, pred::EQ, pc, id);
                let mut joins = HashMap::new();
                if let Some(V::Inst(i)) = placeholder {
                    joins.insert(i, on);
                }
                let before = state.clone();
                for &i in &old[head].phis {
                    if !live[i as usize] || placeholder == Some(V::Inst(i)) {
                        continue;
                    }
                    let Op::Phi(ref inc) = insts[i as usize].op else {
                        unreachable!()
                    };
                    let mut value = self.undef(insts[i as usize].ty);
                    for &(v, p) in inc {
                        let v = match v {
                            V::Inst(j) if owner[def[j as usize]] != 0 => {
                                *before.get(&j).ok_or(VALUE)?
                            }
                            _ => v,
                        };
                        let p = self.i32(p as u32);
                        let edge = self.cmp(head, pred::EQ, prev, p);
                        value = self.select(head, edge, v, value);
                    }
                    joins.insert(i, value);
                }
                self.blocks[head].phis.clear();
                self.blocks[head].body.extend(body);
                let remap = |v: V| -> V {
                    match v {
                        V::Inst(i) if joins.contains_key(&i) => joins[&i],
                        V::Inst(i)
                            if owner[def[i as usize]] != r && owner[def[i as usize]] != 0 =>
                        {
                            before[&i]
                        }
                        _ => v,
                    }
                };
                for &b in &members[r] {
                    for &i in old[b]
                        .phis
                        .iter()
                        .chain(&old[b].body)
                        .filter(|&&i| live[i as usize])
                    {
                        if b == head && matches!(insts[i as usize].op, Op::Phi(_)) {
                            continue;
                        }
                        self.insts[i as usize].op = remap_op(insts[i as usize].op.clone(), &remap);
                    }
                    self.blocks[b].term = old[b].term.map(|t| match t {
                        Term::CondBr(c, t, f) => Term::CondBr(remap(c), t, f),
                        _ => t,
                    });
                }
                let mut next = target(self, tail, self.blocks[tail].term);
                let last = self.i32(tail as u32);
                let defined: Vec<_> = saved
                    .iter()
                    .copied()
                    .filter(|&i| owner[def[i as usize]] == r)
                    .collect();
                let mut values: Vec<_> = defined
                    .iter()
                    .map(|&i| joins.get(&i).copied().unwrap_or(V::Inst(i)))
                    .collect();
                let mut tail = tail;
                if !skip && head == tail {
                    values.push(next);
                    if let Some((end, guarded)) =
                        self.guard_exchange_phases(head, on, &values, &private)
                    {
                        tail = end;
                        values = guarded;
                    }
                    next = values.pop().unwrap();
                }
                at = if skip { self.new_block() } else { tail };
                if skip {
                    self.terminate(gate, Term::CondBr(on, head, at));
                    self.terminate(tail, Term::Br(at));
                }
                let commit = |m: &mut Module, value: V, previous: V| {
                    if skip {
                        let phi = m.phi(at, m.type_of(value));
                        m.set_phi(phi, vec![(value, tail), (previous, gate)]);
                        phi
                    } else {
                        m.select(at, on, value, previous)
                    }
                };
                for (i, value) in defined.into_iter().zip(values) {
                    state.insert(i, commit(self, value, before[&i]));
                }
                pc = commit(self, next, pc);
                prev = commit(self, last, prev);
            }
            if cyclic {
                let latch = at;
                let end = self.new_block();
                let mut active = self.bool(false);
                for &r in &component {
                    let id = self.i32(regions[r].0 as u32);
                    let here = self.cmp(header, pred::EQ, pc_phi, id);
                    active = self.bin(header, Bin::Or, active, here);
                }
                let aborted = abort.map(|b| {
                    let b = self.i32(b as u32);
                    self.cmp(header, pred::EQ, pc_phi, b)
                });
                for (i, phi, initial) in carried {
                    self.set_phi(phi, vec![(initial, before_component), (state[&i], latch)]);
                    state.insert(i, phi);
                }
                self.set_phi(pc_phi, vec![(initial_pc, before_component), (pc, latch)]);
                self.set_phi(
                    prev_phi,
                    vec![(initial_prev, before_component), (prev, latch)],
                );
                let Some(Term::Br(body)) = self.blocks[header].term else {
                    return Err(SHAPE);
                };
                self.terminate(header, Term::CondBr(active, body, end));
                self.terminate(latch, Term::Br(header));
                pc = pc_phi;
                prev = prev_phi;
                loops.push(ScheduleLoop {
                    header,
                    latch,
                    body,
                    end,
                    active,
                    aborted,
                    collective: together,
                });
                let barriers = component
                    .iter()
                    .flat_map(|&r| &members[r])
                    .flat_map(|&b| &old[b].body)
                    .filter(|&&i| {
                        live[i as usize]
                            && matches!(&insts[i as usize].op,
                        Op::Call(f, _) if self.funcs[*f as usize].name == "dx.op.barrier")
                    })
                    .count();
                loop_costs.push(barriers + if together { 2 } else { 0 });
                if std::env::var_os("VICTORY_TRACE_CONSOLE_FLOW").is_some() {
                    costs.push(
                        self.schedule_cost(
                            component
                                .iter()
                                .flat_map(|&r| &members[r])
                                .flat_map(|&b| old[b].body.iter().copied()),
                            &live,
                        ),
                    );
                    let mut ops = std::collections::BTreeMap::new();
                    for &i in component
                        .iter()
                        .flat_map(|&r| &members[r])
                        .flat_map(|&b| &old[b].body)
                    {
                        if live[i as usize]
                            && let Op::Call(f, _) = &insts[i as usize].op
                        {
                            let name = &self.funcs[*f as usize].name;
                            if name.starts_with("dx.op.wave") || name.contains("atomic") {
                                *ops.entry(name.clone()).or_insert(0usize) += 1;
                            }
                        }
                    }
                    loop_ops.push(ops);
                }
                at = end;
            } else if trace {
                let cost = self.schedule_cost(
                    component
                        .iter()
                        .flat_map(|&r| &members[r])
                        .flat_map(|&b| old[b].body.iter().copied()),
                    &live,
                );
                for (sum, x) in once_cost.iter_mut().zip(cost) {
                    *sum += x;
                }
            }
        }
        self.terminate(at, Term::Ret);
        if self.insts.len() - insts.len() > MAX_ADDED {
            return Err("a synchronized loop beyond the code-growth bound");
        }
        if std::env::var_os("VICTORY_TRACE_CONSOLE_FLOW").is_some() {
            eprintln!("console schedule: component loop barriers per traversal {loop_costs:?}");
            eprintln!("console schedule: component loop collectives {loop_ops:?}");
            eprintln!(
                "console schedule: once cost {once_cost:?}; loop body costs {costs:?} (before header votes)"
            );
        }
        Ok(loops)
    }
}

/// Iterative Kosaraju, in deterministic topological order. Recursion here can
/// exhaust the translator stack on the large (thousands of blocks) programs.
fn components(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = graph.len();
    let mut seen = vec![false; n];
    let mut post = Vec::new();
    for root in 0..n {
        let mut stack = vec![(root, false)];
        while let Some((r, finish)) = stack.pop() {
            if finish {
                post.push(r);
                continue;
            }
            if std::mem::replace(&mut seen[r], true) {
                continue;
            }
            stack.push((r, true));
            stack.extend(graph[r].iter().rev().map(|&q| (q, false)));
        }
    }
    let mut reverse = vec![Vec::new(); n];
    for (r, next) in graph.iter().enumerate() {
        for &q in next {
            reverse[q].push(r);
        }
    }
    seen.fill(false);
    let mut out = Vec::new();
    for &root in post.iter().rev() {
        if seen[root] {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![root];
        while let Some(r) = stack.pop() {
            if std::mem::replace(&mut seen[r], true) {
                continue;
            }
            component.push(r);
            stack.extend(&reverse[r]);
        }
        component.sort_unstable();
        out.push(component);
    }
    out
}

pub(super) fn remap_op(op: Op, f: &impl Fn(V) -> V) -> Op {
    match op {
        Op::Bin(op, a, b) => Op::Bin(op, f(a), f(b)),
        Op::Cast(op, a) => Op::Cast(op, f(a)),
        Op::Cmp(op, a, b) => Op::Cmp(op, f(a), f(b)),
        Op::Select(a, b, c) => Op::Select(f(a), f(b), f(c)),
        Op::Extract(a, n) => Op::Extract(f(a), n),
        Op::Call(fun, args) => Op::Call(fun, args.into_iter().map(f).collect()),
        Op::Phi(inc) => Op::Phi(inc.into_iter().map(|(v, b)| (f(v), b)).collect()),
        Op::Gep(t, p, idx) => Op::Gep(t, f(p), idx.into_iter().map(f).collect()),
        Op::Load(p) => Op::Load(f(p)),
        Op::Store(p, v) => Op::Store(f(p), f(v)),
        Op::AtomicRmw(op, p, v) => Op::AtomicRmw(op, f(p), f(v)),
        Op::CmpXchg(p, a, b) => Op::CmpXchg(f(p), f(a), f(b)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::shader::Shader;

    #[test]
    fn excessive_schedule_growth_is_refused_before_mutation() {
        let mut m = Module::default();
        let regions = vec![(0, 0, None); MAX_ADDED / 4 + 2];
        assert_eq!(
            m.synchronize_regions(&regions, None).unwrap_err(),
            "a synchronized loop beyond the code-growth bound"
        );
        assert!(m.insts.is_empty() && m.blocks.is_empty());
    }

    #[test]
    fn an_exchange_or_an_escaping_value_inside_a_guard_fails_closed() {
        for barrier in [false, true] {
            let mut s = Shader::new(false);
            let (head, tail, yes, end) = (
                s.m.new_block(),
                s.m.new_block(),
                s.m.new_block(),
                s.m.new_block(),
            );
            let on = s.m.phi(head, s.t.i1);
            let c = s.op(head, "waveIsFirstLane", Attr::NoUnwind, s.t.i1, 110, &[]);
            let one = s.m.i32(1);
            // Keep the escaping value nonconstant: folding 1 + 2 is safe even
            // when its old definition was guarded, so it cannot test dominance.
            let unknown = s.m.cast(yes, Cast::ZExt, c, s.t.i32);
            let value = s.m.bin(yes, Bin::Add, one, unknown);
            if barrier {
                let mode = s.m.i32(9);
                s.op(yes, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
            } else {
                s.op(
                    end,
                    "storeOutput.i32",
                    Attr::NoUnwind,
                    s.t.void,
                    5,
                    &[value],
                );
            }
            s.m.terminate(0, Term::Br(head));
            s.m.terminate(head, Term::CondBr(c, yes, tail));
            s.m.terminate(yes, Term::Br(tail));
            s.m.terminate(tail, Term::CondBr(c, head, end));
            s.m.terminate(end, Term::Ret);
            let why =
                s.m.synchronize_regions(
                    &[(0, 0, None), (head, tail, Some(on)), (end, end, None)],
                    None,
                )
                .unwrap_err();
            assert_eq!(
                why,
                if barrier {
                    "a synchronized loop exchange inside an instruction guard"
                } else {
                    "a synchronized loop value unavailable at its block boundary"
                }
            );
        }
    }

    #[test]
    fn the_ssa_check_accepts_loop_phis_but_not_values_escaping_a_guard() {
        for joined in [false, true] {
            let mut s = Shader::new(false);
            let (head, yes, join, end) = (
                s.m.new_block(),
                s.m.new_block(),
                s.m.new_block(),
                s.m.new_block(),
            );
            let count = s.m.phi(head, s.t.i32);
            let (zero, one, two) = (s.m.i32(0), s.m.i32(1), s.m.i32(2));
            let cond = s.op(head, "waveIsFirstLane", Attr::NoUnwind, s.t.i1, 110, &[]);
            let value = s.m.bin(yes, Bin::Add, count, one);
            let value = if joined {
                let phi = s.m.phi(join, s.t.i32);
                s.m.set_phi(phi, vec![(value, yes), (count, head)]);
                phi
            } else {
                value
            };
            let more = s.m.cmp(join, pred::IULT, value, two);
            s.m.set_phi(count, vec![(zero, 0), (value, join)]);
            s.m.terminate(0, Term::Br(head));
            s.m.terminate(head, Term::CondBr(cond, yes, join));
            s.m.terminate(yes, Term::Br(join));
            s.m.terminate(join, Term::CondBr(more, head, end));
            s.m.terminate(end, Term::Ret);
            assert_eq!(s.m.values_dominate_uses(), joined);
        }
    }
}
