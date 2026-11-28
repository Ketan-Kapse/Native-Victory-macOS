//! Move an invariant private cross-half exchange to the one-time prologue.
//! No title access, phi, division, atomic or unknown call may be speculated.
//! The exchange's complete write/fence/read/fence sequence moves as a unit;
//! its result stays in SSA, so subsequent scratch reuse cannot invalidate it.
use super::*;

impl Module {
    /// Batch independent scalar exchanges in one straight-line block. Scratch
    /// banks are disjoint and bounded by the caller's allocation. A later store
    /// depending on an earlier read ends the batch; resource accesses and other
    /// collectives also end it. No instruction moves across a CFG edge.
    pub(crate) fn batch_exchanges(&mut self, stride: u32, capacity: usize) {
        let mut by_block: std::collections::BTreeMap<usize, Vec<Vec<u32>>> = Default::default();
        for (b, ids) in std::mem::take(&mut self.exchanges) {
            by_block.entry(b).or_default().push(ids);
        }
        let mut next_bank = 0;
        for (b, mut exchanges) in by_block {
            let original = self.blocks[b].body.clone();
            let positions: HashMap<_, _> = original.iter().enumerate().map(|(n, &i)| (i, n)).collect();
            exchanges.sort_by_key(|ids| positions[&ids[0]]);
            let mut out = Vec::new();
            let mut cursor = 0;
            let mut e = 0;
            while e < exchanges.len() {
                let start = positions[&exchanges[e][0]];
                let mut end = positions[exchanges[e].last().unwrap()] + 1;
                let mut count = 1;
                let mut barriers: std::collections::HashSet<_> = exchanges[e].iter().copied().filter(|&i| self.is_barrier(i)).collect();
                while count < capacity && e + count < exchanges.len() {
                    let next = &exchanges[e + count];
                    let next_end = positions[next.last().unwrap()] + 1;
                    let mut candidate = barriers.clone();
                    candidate.extend(next.iter().copied().filter(|&i| self.is_barrier(i)));
                    if self.exchange_batch_phases(&original[start..next_end], &candidate).is_none() { break; }
                    barriers = candidate;
                    end = next_end;
                    count += 1;
                }
                out.extend_from_slice(&original[cursor..start]);
                if next_bank + count > capacity { next_bank = 0; }
                let mut added = HashMap::new();
                let mut banks = 0;
                for n in 0..count {
                    let bank = next_bank + n;
                    banks |= 1 << bank;
                    for &i in &exchanges[e + n] {
                        if let Op::Gep(ty, V::Global(g), ref idx) = self.insts[i as usize].op.clone()
                            && self.globals[g as usize].name == "xchg"
                        {
                            self.exchange_banks.insert(i, 1 << bank);
                            if bank != 0 {
                                let mut idx = idx.clone();
                                let offset = self.i32(bank as u32 * stride);
                                let value = self.bin(b, Bin::Add, *idx.last().unwrap(), offset);
                                let V::Inst(id) = value else { unreachable!() };
                                *idx.last_mut().unwrap() = value;
                                self.insts[i as usize].op = Op::Gep(ty, V::Global(g), idx);
                                added.insert(i, id);
                            }
                        }
                    }
                }
                next_bank += count;
                let ids: Vec<_> = original[start..end].iter().flat_map(|&i| added.get(&i).copied().into_iter().chain([i])).collect();
                let last = *original[start..end].iter().rev().find(|i| barriers.contains(i)).unwrap();
                // Only a fence after the reads is a release: the one between
                // the stores and the reads is the exchange itself.
                let read = original[start..end].iter().take_while(|&&i| i != last)
                    .any(|&i| matches!(self.insts[i as usize].op, Op::Load(_)));
                if read && barriers.len() >= 2 {
                    self.exchange_releases.push((last, banks));
                }
                let batch = if count == 1 { ids } else {
                    let (mut writes, reads) = self.exchange_batch_phases(&ids, &barriers).expect("bank offsets are pure");
                    let first = *original[start..end].iter().find(|i| barriers.contains(i)).unwrap();
                    writes.push(first);
                    writes.extend(reads);
                    writes.push(last);
                    writes
                };
                out.extend(&batch);
                self.exchanges.push((b, batch));
                cursor = end;
                e += count;
            }
            out.extend_from_slice(&original[cursor..]);
            self.blocks[b].body = out;
        }
    }

    fn is_barrier(&self, i: u32) -> bool {
        matches!(&self.insts[i as usize].op, Op::Call(f, args) if self.funcs[*f as usize].name == "dx.op.barrier"
            && self.int_value(args[1]).is_some_and(|mode| mode & 1 != 0))
    }

    fn exchange_batch_phases(&self, ids: &[u32], barriers: &std::collections::HashSet<u32>) -> Option<(Vec<u32>, Vec<u32>)> {
        let mut after = std::collections::HashSet::new();
        let mut phases = [Vec::new(), Vec::new()];
        for &i in ids {
            if barriers.contains(&i) { continue; }
            let op = &self.insts[i as usize].op;
            let mut read = matches!(op, Op::Load(_));
            for_each_operand(op, |v| { if let V::Inst(j) = v { read |= after.contains(&j); } });
            let allowed = match op {
                // As in `hoist_exchanges`: a batch moves as one exchange.
                Op::Bin(Bin::UDiv | Bin::SDiv | Bin::URem, ..) => false,
                Op::Bin(..) | Op::Cmp(..) | Op::Cast(..) | Op::Extract(..) | Op::Select(..) | Op::Gep(..) => true,
                Op::Load(p) | Op::Store(p, _) => {
                    let mut p = *p;
                    while let V::Inst(j) = p {
                        if let Op::Gep(_, base, _) = self.insts[j as usize].op { p = base; }
                        else { break; }
                    }
                    matches!(p, V::Global(g) if self.globals[g as usize].name == "xchg")
                        && (!matches!(op, Op::Store(..)) || !read)
                }
                Op::Call(f, _) => self.funcs[*f as usize].attr == Some(Attr::ReadNone)
                    || matches!(self.funcs[*f as usize].name.as_str(), "dx.op.waveActiveBallot" | "dx.op.waveReadLaneAt.i32"),
                _ => false,
            };
            if !allowed { return None; }
            if read { after.insert(i); }
            phases[usize::from(read)].push(i);
        }
        let [writes, reads] = phases;
        Some((writes, reads))
    }

    /// A release protects scratch reuse, not title memory. Omit it only when
    /// every path to a possibly aliasing store crosses another group barrier.
    /// Check the final CFG (after hoisting, component scheduling and votes),
    /// treating all candidate removals as absent. Retaining a fence only makes
    /// the other proofs stronger. The walk includes all reachable cycle edges.
    pub(crate) fn elide_exchange_releases(&mut self) -> usize {
        let positions: HashMap<_, _> = self.blocks.iter().enumerate().flat_map(|(b, block)|
            block.body.iter().enumerate().map(move |(n, &i)| (i, (b, n + 1)))).collect();
        let mut removed: std::collections::HashSet<_> = self.exchange_releases.iter()
            .map(|&(i, _)| i).filter(|i| positions.contains_key(i)).collect();
        for &(release, banks) in &self.exchange_releases {
            let Some(&start) = positions.get(&release) else { continue };
            let mut work = vec![start];
            let mut seen = std::collections::HashSet::new();
            let mut safe = true;
            while let Some((b, n)) = work.pop() {
                if !seen.insert((b, n)) { continue; }
                let mut fenced = false;
                for &i in &self.blocks[b].body[n..] {
                    if self.is_barrier(i) && !removed.contains(&i) {
                        fenced = true;
                        break;
                    }
                    if let Op::Store(V::Inst(p), _) = self.insts[i as usize].op {
                        if let Some(&written) = self.exchange_banks.get(&p) {
                            safe &= written & banks == 0;
                        } else {
                            // Unannotated scratch (votes/reductions) may alias.
                            let mut ptr = V::Inst(p);
                            while let V::Inst(j) = ptr {
                                if let Op::Gep(_, base, _) = self.insts[j as usize].op { ptr = base; }
                                else { break; }
                            }
                            safe &= !matches!(ptr, V::Global(g) if self.globals[g as usize].name == "xchg");
                        }
                    }
                    if !safe { break; }
                }
                if !safe { break; }
                if !fenced { work.extend(successors(self.blocks[b].term).iter().map(|&q| (q, 0))); }
            }
            if !safe { removed.remove(&release); }
        }
        for block in &mut self.blocks { block.body.retain(|i| !removed.contains(i)); }
        removed.len()
    }

    /// The barrier just emitted in `block` only orders the LDS atomic after it.
    pub(crate) fn mark_order_fence(&mut self, block: usize) {
        if let Some(&i) = self.blocks[block].body.last() {
            self.order_fences.push(i);
        }
    }

    /// An order fence keeps a group barrier between an LDS atomic and the LDS
    /// access before it. It is redundant when every path that arrives at it
    /// crosses another group barrier after the last access to title memory (or
    /// has none). Walk the final CFG backwards over every edge, cycles
    /// included, counting neither the other candidates nor any scratch release
    /// (`elide_exchange_releases` may still drop those) as a barrier. Memory
    /// that is not provably the exchange scratch counts as title memory.
    /// Run this before `elide_exchange_releases`, which then sees what is left.
    pub(crate) fn elide_order_fences(&mut self) -> usize {
        let mut positions = HashMap::new();
        let mut preds = vec![Vec::new(); self.blocks.len()];
        for (b, block) in self.blocks.iter().enumerate() {
            positions.extend(block.body.iter().enumerate().map(|(n, &i)| (i, (b, n))));
            for &q in successors(block.term).iter() { preds[q].push(b); }
        }
        let releases: std::collections::HashSet<_> = self.exchange_releases.iter().map(|r| r.0).collect();
        let mut removed: std::collections::HashSet<_> = self.order_fences.iter().copied()
            .filter(|i| positions.contains_key(i) && self.is_barrier(*i)).collect();
        let scratch = |mut p: V| {
            while let V::Inst(j) = p {
                if let Op::Gep(_, base, _) = self.insts[j as usize].op { p = base; }
                else { return false; }
            }
            matches!(p, V::Global(g) if self.globals[g as usize].name == "xchg")
        };
        for &fence in &self.order_fences {
            if !removed.contains(&fence) { continue; }
            let mut work = vec![positions[&fence]];
            let mut seen = std::collections::HashSet::new();
            let mut needed = false;
            while let Some((b, n)) = work.pop() {
                let mut fenced = false;
                for &i in self.blocks[b].body[..n].iter().rev() {
                    if self.is_barrier(i) && !removed.contains(&i) && !releases.contains(&i) {
                        fenced = true;
                        break;
                    }
                    needed = match self.insts[i as usize].op {
                        Op::Load(p) | Op::Store(p, _) | Op::AtomicRmw(_, p, _) | Op::CmpXchg(p, ..) => !scratch(p),
                        _ => false,
                    };
                    if needed { break; }
                }
                if needed { break; }
                if !fenced {
                    work.extend(preds[b].iter().filter(|&&q| seen.insert(q)).map(|&q| (q, self.blocks[q].body.len())));
                }
            }
            if needed { removed.remove(&fence); }
        }
        for block in &mut self.blocks { block.body.retain(|i| !removed.contains(i)); }
        removed.len()
    }

    pub(crate) fn private_exchange_barriers(&self) -> std::collections::HashSet<u32> {
        self.exchanges.iter().flat_map(|(_, ids)| ids.iter().copied()).filter(|&i| {
            matches!(&self.insts[i as usize].op, Op::Call(f, _) if self.funcs[*f as usize].name == "dx.op.barrier")
        }).collect()
    }

    /// A straight-line region can skip its arithmetic and private slot traffic
    /// while still attending the *same* barrier instructions. Both halves have
    /// the same PC predicate; no other console wave reads their private slots.
    /// Do not apply this to title LDS ordering or group-wide vote traffic.
    pub(crate) fn guard_exchange_phases(
        &mut self,
        head: usize,
        on: V,
        exports: &[V],
        private: &std::collections::HashSet<u32>,
    ) -> Option<(usize, Vec<V>)> {
        let mut phases = Vec::new();
        let mut chunk = Vec::new();
        for &i in &self.blocks[head].body {
            if V::Inst(i) == on {
                continue;
            }
            if matches!(&self.insts[i as usize].op, Op::Call(f, _) if self.funcs[*f as usize].name == "dx.op.barrier")
            {
                if !private.contains(&i) {
                    return None;
                }
                phases.push((std::mem::take(&mut chunk), Some(i)));
            } else {
                chunk.push(i);
            }
        }
        phases.push((chunk, None));
        let mut outputs = Vec::new();
        for (p, (chunk, _)) in phases.iter().enumerate() {
            let defined: std::collections::HashSet<_> = chunk.iter().copied().collect();
            let mut used = std::collections::BTreeSet::new();
            let mut need = |v| {
                if let V::Inst(i) = v
                    && defined.contains(&i)
                {
                    used.insert(i);
                }
            };
            for (rest, _) in &phases[p + 1..] {
                for &i in rest {
                    for_each_operand(&self.insts[i as usize].op, &mut need);
                }
            }
            for &v in exports {
                need(v);
            }
            // Numeric results have a harmless inactive value; pointer/aggregate
            // lifetime splitting is deliberately outside this optimization.
            if used.iter().any(|&i| {
                !matches!(
                    self.types[self.insts[i as usize].ty.0 as usize],
                    TypeDef::Int(_) | TypeDef::Float
                )
            }) {
                return None;
            }
            outputs.push(used);
        }
        let mut at = head;
        let mut replace = HashMap::new();
        self.blocks[head].body.retain(|&i| V::Inst(i) == on);
        for ((chunk, barrier), outputs) in phases.into_iter().zip(outputs) {
            if !chunk.is_empty() {
                let body = self.new_block();
                let join = self.new_block();
                self.terminate(at, Term::CondBr(on, body, join));
                self.terminate(body, Term::Br(join));
                for &i in &chunk {
                    self.insts[i as usize].op =
                        synchronized::remap_op(self.insts[i as usize].op.clone(), &|v| {
                            if let V::Inst(i) = v {
                                replace.get(&i).copied().unwrap_or(v)
                            } else {
                                v
                            }
                        });
                }
                self.blocks[body].body = chunk;
                for i in outputs {
                    let ty = self.insts[i as usize].ty;
                    let zero = if self.types[ty.0 as usize] == TypeDef::Float {
                        self.f32(0)
                    } else {
                        self.int_const(ty, 0)
                    };
                    let phi = self.phi(join, ty);
                    self.set_phi(phi, vec![(V::Inst(i), body), (zero, at)]);
                    replace.insert(i, phi);
                }
                at = join;
            }
            if let Some(barrier) = barrier {
                self.blocks[at].body.push(barrier);
            }
        }
        let exports = exports
            .iter()
            .map(|&v| {
                if let V::Inst(i) = v {
                    replace.get(&i).copied().unwrap_or(v)
                } else {
                    v
                }
            })
            .collect();
        Some((at, exports))
    }

    /// Structural counts only, never shader bytes: group barriers, TGSM fences,
    /// UAV fences, private slot stores, private slot loads, title atomic sites.
    pub(crate) fn schedule_cost(
        &self,
        ids: impl Iterator<Item = u32>,
        live: &[bool],
    ) -> [usize; 6] {
        let scratch = |mut v: V| {
            while let V::Inst(i) = v {
                match &self.insts[i as usize].op {
                    Op::Gep(_, p, _) => v = *p,
                    _ => return false,
                }
            }
            matches!(v, V::Global(g) if self.globals[g as usize].name == "xchg")
        };
        let mut count = [0; 6];
        for i in ids.filter(|&i| live[i as usize]) {
            match &self.insts[i as usize].op {
                Op::Call(f, args) if self.funcs[*f as usize].name == "dx.op.barrier" => {
                    let mode = self.int_value(args[1]).unwrap_or(0);
                    count[0] += usize::from(mode & 1 != 0);
                    count[1] += usize::from(mode & 8 != 0);
                    count[2] += usize::from(mode & 2 != 0);
                }
                Op::Store(p, _) if scratch(*p) => count[3] += 1,
                Op::Load(p) if scratch(*p) => count[4] += 1,
                Op::AtomicRmw(..) | Op::CmpXchg(..) => count[5] += 1,
                Op::Call(f, _) if self.funcs[*f as usize].name.contains("atomic") => count[5] += 1,
                _ => {}
            }
        }
        count
    }

    /// `on` is group-uniform dispatch coverage, emitted before `start`.
    /// Keep input setup in entry and guard everything that followed it.
    pub(crate) fn guard_dispatch(&mut self, on: V, start: usize) {
        let body = self.new_block();
        let end = self.new_block();
        self.blocks[body].body = self.blocks[0].body.split_off(start);
        self.blocks[body].term = self.blocks[0].term;
        for inst in &mut self.insts {
            if let Op::Phi(inc) = &mut inst.op {
                for (_, from) in inc {
                    if *from == 0 {
                        *from = body;
                    }
                }
            }
        }
        self.terminate(0, Term::CondBr(on, body, end));
        self.terminate(end, Term::Ret);
    }

    /// Expose invariant exchange operands hidden behind identity arithmetic and
    /// loop phis such as phi(seed, self). This is integer/boolean substitution,
    /// never a memory equivalence or an assumption about a scalar load's value.
    pub(crate) fn simplify_exchange_inputs(&mut self) {
        let mut alias = vec![None; self.insts.len()];
        let resolve = |mut v: V, alias: &[Option<V>]| {
            while let V::Inst(i) = v {
                let Some(next) = alias[i as usize] else { break };
                v = next;
            }
            v
        };
        let mut users = vec![Vec::new(); self.insts.len()];
        for (i, inst) in self.insts.iter().enumerate() {
            for_each_operand(&inst.op, |v| {
                if let V::Inst(j) = v {
                    users[j as usize].push(i);
                }
            });
        }
        let mut work: std::collections::VecDeque<_> = (0..self.insts.len()).collect();
        while let Some(i) = work.pop_front() {
            let op = synchronized::remap_op(self.insts[i].op.clone(), &|v| resolve(v, &alias));
            let ty = self.insts[i].ty;
            let value = match op {
                Op::Phi(ref inc) => {
                    let mut values = inc.iter().map(|p| p.0).filter(|&v| v != V::Inst(i as u32));
                    values.next().filter(|&v| values.all(|w| w == v))
                }
                Op::Select(c, t, f) => {
                    if t == f {
                        Some(t)
                    } else {
                        self.int_value(c).map(|v| if v != 0 { t } else { f })
                    }
                }
                Op::Bin(op, a, b) if matches!(self.types[ty.0 as usize], TypeDef::Int(1..=64)) => {
                    let TypeDef::Int(bits) = self.types[ty.0 as usize] else {
                        unreachable!()
                    };
                    let mask = u64::MAX >> (64 - bits);
                    let (av, bv) = (
                        self.int_value(a).map(|x| x as u64 & mask),
                        self.int_value(b).map(|x| x as u64 & mask),
                    );
                    let constant = match (av, bv, op) {
                        (Some(x), Some(y), Bin::And) => Some(x & y),
                        (Some(x), Some(y), Bin::Or) => Some(x | y),
                        (Some(x), Some(y), Bin::Xor) => Some(x ^ y),
                        (Some(x), Some(y), Bin::Add) => Some(x.wrapping_add(y)),
                        (Some(x), Some(y), Bin::Sub) => Some(x.wrapping_sub(y)),
                        (Some(x), Some(y), Bin::Mul) => Some(x.wrapping_mul(y)),
                        (Some(x), Some(y), Bin::Shl) if y < bits as u64 => Some(x << y),
                        (Some(x), Some(y), Bin::LShr) if y < bits as u64 => Some(x >> y),
                        _ => None,
                    };
                    if let Some(x) = constant {
                        Some(self.int_const(ty, (x & mask) as i64))
                    } else {
                        match (op, av, bv) {
                            (Bin::And, Some(0), _) | (Bin::And, _, Some(0)) => {
                                Some(self.int_const(ty, 0))
                            }
                            (Bin::And, Some(x), _) if x == mask => Some(b),
                            (Bin::And, _, Some(x)) if x == mask => Some(a),
                            (Bin::Or | Bin::Xor | Bin::Add, Some(0), _) => Some(b),
                            (Bin::Or | Bin::Xor | Bin::Add | Bin::Sub, _, Some(0)) => Some(a),
                            (Bin::And | Bin::Or, _, _) if a == b => Some(a),
                            _ => None,
                        }
                    }
                }
                _ => None,
            };
            if let Some(v) = value
                && v != V::Inst(i as u32) && alias[i] != Some(v)
            {
                alias[i] = Some(v);
                work.extend(&users[i]);
            }
        }
        for inst in &mut self.insts {
            inst.op = synchronized::remap_op(inst.op.clone(), &|v| resolve(v, &alias));
        }
        for block in &mut self.blocks {
            if let Some(Term::CondBr(c, t, f)) = block.term {
                block.term = Some(Term::CondBr(resolve(c, &alias), t, f));
            }
        }
    }

    pub(crate) fn exchange_start(&self, at: usize) -> usize {
        self.blocks[at].body.len()
    }

    pub(crate) fn exchange_end(&mut self, at: usize, start: usize) {
        self.exchanges
            .push((at, self.blocks[at].body[start..].to_vec()));
    }

    /// A private exchange has no observable effect unless its result is used.
    /// Treat the complete sequence as one operation for liveness, preserving
    /// its payload dependency (a load alone has no SSA edge to the slot store).
    /// Title LDS/barriers and atomics remain roots; only explicitly recorded
    /// private exchanges may be removed.
    pub(crate) fn discard_unused_exchanges(&mut self) -> usize {
        let mut owner = vec![None; self.insts.len()];
        for (e, (_, instructions)) in self.exchanges.iter().enumerate() {
            for &i in instructions {
                owner[i as usize] = Some(e);
            }
        }
        let mut live = vec![false; self.insts.len()];
        let mut used = vec![false; self.exchanges.len()];
        let mut work = Vec::new();
        let mark = |v: V, work: &mut Vec<u32>| {
            if let V::Inst(i) = v {
                work.push(i);
            }
        };
        for block in &self.blocks {
            work.extend(
                block
                    .body
                    .iter()
                    .copied()
                    .filter(|&i| owner[i as usize].is_none() && self.rooted(i)),
            );
            if let Some(Term::CondBr(c, _, _)) = block.term {
                mark(c, &mut work);
            }
        }
        while let Some(i) = work.pop() {
            if std::mem::replace(&mut live[i as usize], true) {
                continue;
            }
            if let Some(e) = owner[i as usize]
                && !std::mem::replace(&mut used[e], true)
            {
                work.extend(&self.exchanges[e].1);
            }
            for_each_operand(&self.insts[i as usize].op, |v| mark(v, &mut work));
        }
        for block in &mut self.blocks {
            block
                .body
                .retain(|&i| owner[i as usize].is_none_or(|e| used[e]));
        }
        let before = self.exchanges.len();
        let mut e = 0;
        self.exchanges.retain(|_| {
            let keep = used[e];
            e += 1;
            keep
        });
        before - self.exchanges.len()
    }

    pub(crate) fn hoist_exchanges(&mut self, prologue: usize) -> usize {
        // The entry region can include guarded descriptor loads. Exchanges
        // already in that region must not move *past* their existing consumers.
        let mut entry = std::collections::HashSet::new();
        let mut work = vec![0];
        while let Some(b) = work.pop() {
            if !entry.insert(b) || b == prologue {
                continue;
            }
            work.extend(successors(self.blocks[b].term).iter().copied());
        }
        let mut owner = vec![usize::MAX; self.insts.len()];
        for (b, block) in self.blocks.iter().enumerate() {
            for &i in block.phis.iter().chain(&block.body) {
                owner[i as usize] = b;
            }
        }
        let mut count = 0;
        for (at, exchange) in std::mem::take(&mut self.exchanges) {
            if entry.contains(&at) {
                continue;
            }
            let inside: std::collections::HashSet<_> = exchange.iter().copied().collect();
            let mut move_ids = Vec::new();
            let mut seen = std::collections::HashSet::new();
            let mut stack: Vec<_> = exchange.iter().rev().map(|&i| (i, false)).collect();
            let mut valid = true;
            while let Some((i, finish)) = stack.pop() {
                if finish {
                    move_ids.push(i);
                    continue;
                }
                if entry.contains(&owner[i as usize]) {
                    continue;
                }
                if !seen.insert(i) {
                    continue;
                }
                let op = &self.insts[i as usize].op;
                let pure = match op {
                    Op::Bin(Bin::UDiv | Bin::SDiv | Bin::URem, ..) => false,
                    Op::Bin(..)
                    | Op::Cast(..)
                    | Op::Cmp(..)
                    | Op::Select(..)
                    | Op::Extract(..)
                    | Op::Gep(..) => true,
                    Op::Call(f, _) => self.funcs[*f as usize].attr == Some(Attr::ReadNone),
                    _ => false,
                };
                if !inside.contains(&i) && !pure {
                    valid = false;
                    break;
                }
                stack.push((i, true));
                for_each_operand(op, |v| {
                    if let V::Inst(j) = v {
                        stack.push((j, false));
                    }
                });
            }
            if !valid {
                continue;
            }
            // Pure dependencies can be shared by several exchanges. Move each
            // once and update its owner before considering the next exchange.
            for (b, block) in self.blocks.iter_mut().enumerate() {
                if b != prologue {
                    block.body.retain(|i| !seen.contains(i));
                }
            }
            for i in move_ids {
                owner[i as usize] = prologue;
                self.blocks[prologue].body.push(i);
            }
            count += 1;
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::shader::Shader;

    #[test]
    fn scalar_batches_preserve_dependencies_and_scratch_bank_wraparound() {
        for dependent in [false, true] {
            for count in [2, 12] {
                let mut s = Shader::new(false);
                s.xchg_dwords = 32;
                let f = s.op(0, "flattenedThreadIdInGroup.i32", Attr::ReadNone, s.t.i32, 96, &[]);
                let (five, one, two, mode) = (s.m.i32(5), s.m.i32(1), s.m.i32(2), s.m.i32(9));
                let mine = s.m.bin(0, Bin::LShr, f, five);
                let peer = s.m.bin(0, Bin::Xor, mine, one);
                let mut value = mine;
                let mut sum = s.m.i32(0);
                for n in 0..count {
                    let base = if dependent { value } else { mine };
                    let n = s.m.i32(n);
                    let data = s.m.bin(0, Bin::Add, base, n);
                    let start = s.m.exchange_start(0);
                    let shared = s.xchg();
                    let to = s.m.element(0, shared, mine);
                    s.m.store(0, to, data);
                    s.op(0, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
                    let from = s.m.element(0, shared, peer);
                    value = s.m.load(0, from);
                    s.op(0, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
                    s.m.exchange_end(0, start);
                    sum = s.m.bin(0, Bin::Add, sum, value);
                }
                let off = s.m.bin(0, Bin::Shl, f, two);
                let h = s.root_uav_handle(0);
                let u = s.m.undef(s.t.i32);
                let mask = s.m.i8(1);
                let align = s.m.i32(4);
                s.op(0, "rawBufferStore.i32", Attr::NoUnwind, s.t.void, 140, &[h, off, u, sum, u, u, u, mask, align]);
                s.m.terminate(0, Term::Ret);
                let before = s.m.test_program();
                s.m.batch_exchanges(4, 8);
                s.m.elide_exchange_releases();
                assert!(s.m.values_dominate_uses());
                let after = s.m.test_program();
                for reverse in [false, true] {
                    let old = before.run(128, &[], reverse);
                    let new = after.run(128, &[], reverse);
                    assert_eq!(new.output, old.output);
                    assert_eq!(new.writes, vec![1; 128]);
                    assert!(new.barriers < old.barriers);
                    if count == 2 { assert_eq!(new.barriers, if dependent { 2 } else { 1 }); }
                }
            }
        }
    }

    #[test]
    fn an_order_fence_goes_only_where_another_barrier_separates_the_accesses() {
        // (a barrier between the two atomics, the loop repeats them)
        for (between, cyclic) in [(false, false), (true, false), (true, true), (false, true)] {
            let mut s = Shader::new(false);
            s.lds_dwords = 1;
            let (zero, one, mode) = (s.m.i32(0), s.m.i32(1), s.m.i32(9));
            let lds = s.lds();
            let counter = s.m.element(0, lds, zero);
            let body = s.m.new_block();
            let end = s.m.new_block();
            s.m.terminate(0, Term::Br(body));
            let fence = |s: &mut Shader| {
                s.op(body, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
                s.m.mark_order_fence(body);
            };
            // The first access of the program: nothing to be ordered after,
            // unless the backedge brings the second atomic round.
            fence(&mut s);
            let old = s.m.atomic(body, Rmw::Add, counter, one);
            if between { s.op(body, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]); }
            fence(&mut s);
            s.m.atomic(body, Rmw::Add, counter, one);
            let again = s.m.cmp(body, pred::EQ, old, zero);
            s.m.terminate(body, if cyclic { Term::CondBr(again, body, end) } else { Term::Br(end) });
            s.m.terminate(end, Term::Ret);
            let kept = 2 - s.m.elide_order_fences();
            assert_eq!(kept, usize::from(!between) + usize::from(cyclic), "{between} {cyclic}");
            assert_eq!(s.m.written_calls("dx.op.barrier"), kept + usize::from(between));
        }
    }

    #[test]
    fn unused_private_exchange_does_not_keep_its_fences_or_remove_its_atomic_input() {
        for used in [false, true] {
            let mut s = Shader::new(false);
            s.lds_dwords = 1;
            s.xchg_dwords = 128;
            let (zero, one, mode) = (s.m.i32(0), s.m.i32(1), s.m.i32(9));
            let lds = s.lds();
            let counter = s.m.element(0, lds, zero);
            let old = s.m.atomic(0, Rmw::Add, counter, one);
            let xchg = s.xchg();
            let start = s.m.exchange_start(0);
            let slot = s.m.element(0, xchg, zero);
            s.m.store(0, slot, old);
            s.op(0, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
            let result = s.m.load(0, slot);
            s.op(0, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
            s.m.exchange_end(0, start);
            if used {
                s.m.store(0, counter, result);
            }
            s.m.terminate(0, Term::Ret);
            assert_eq!(s.m.discard_unused_exchanges(), usize::from(!used));
            assert_eq!(s.m.written_calls("dx.op.barrier"), if used { 2 } else { 0 });
            let cost =
                s.m.schedule_cost(s.m.blocks[0].body.iter().copied(), &s.m.live());
            assert_eq!(cost[5], 1, "the atomic effect survives an unused old value");
        }
    }
}
