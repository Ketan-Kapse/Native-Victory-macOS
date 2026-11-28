//! Test-only integer DXIL executor. Runs the emitted graph, not a second copy of
//! the scheduling algorithm. Host waves rendezvous at wave calls; the complete
//! group must rendezvous at the same barrier instruction. Unknown ops fail.
//! Resource handles are opaque; fixtures use one raw input/output buffer.
use super::*;

type Data = Vec<u64>;

pub(crate) struct TestProgram {
    types: Vec<TypeDef>,
    consts: Vec<(Ty, Const)>,
    funcs: Vec<String>,
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    lds: Option<u64>,
}

pub(crate) struct Execution {
    pub output: Vec<u32>,
    pub writes: Vec<usize>,
    pub barriers: usize,
    pub steps: Vec<usize>,
    pub exchange_reads: usize,
    pub exchange_writes: usize,
    pub reads: Vec<usize>,
    pub atomics: Vec<(usize, u32)>,
    pub lds_reads: Vec<(usize, usize)>,
    pub lds_writes: Vec<(usize, usize)>,
    pub lds_atomics: Vec<(usize, usize, u32, u32)>,
    pub memory: HashMap<usize, u32>,
}

struct Lane {
    block: usize,
    at: usize,
    done: bool,
    values: Vec<Option<Data>>,
}

impl Module {
    pub(crate) fn test_program(&self) -> TestProgram {
        let live = self.live();
        let mut blocks = self.blocks.clone();
        for b in &mut blocks {
            b.body.retain(|i| live[*i as usize]);
            b.phis.retain(|i| live[*i as usize]);
        }
        TestProgram {
            types: self.types.clone(),
            consts: self.consts.clone(),
            funcs: self.funcs.iter().map(|f| f.name.clone()).collect(),
            insts: self.insts.clone(),
            blocks,
            lds: self
                .globals
                .iter()
                .position(|g| g.name == "lds")
                .map(|g| g as u64),
        }
    }
}

impl TestProgram {
    /// Shrink a compiled safety limit for an escape-path value test. This only
    /// changes the constant; all emitted control flow and rendezvous still run.
    pub(crate) fn replace_i32_constant(&mut self, from: u32, to: u32) {
        let mut found = false;
        for (ty, value) in &mut self.consts {
            if self.types[ty.0 as usize] == TypeDef::Int(32)
                && matches!(value, Const::Int(x) if *x == i64::from(from))
            {
                *value = Const::Int(i64::from(to));
                found = true;
            }
        }
        assert!(found, "the compiled safety limit must exist");
    }

    fn signed(&self, v: V, x: u64) -> i64 {
        let ty = match v {
            V::Inst(i) => self.insts[i as usize].ty,
            V::Const(i) => self.consts[i as usize].0,
            _ => panic!("non-integer operand {v:?}"),
        };
        let TypeDef::Int(bits) = self.types[ty.0 as usize] else {
            panic!("non-integer operand {v:?}");
        };
        ((x << (64 - bits)) as i64) >> (64 - bits)
    }

    fn get(&self, lane: &Lane, v: V) -> Data {
        match v {
            V::Inst(i) => lane.values[i as usize]
                .clone()
                .unwrap_or_else(|| panic!("unavailable {v:?} in block {}", lane.block)),
            V::Global(g) => vec![g as u64, 0],
            V::Const(i) => match &self.consts[i as usize].1 {
                Const::Int(x) => {
                    let TypeDef::Int(bits) = self.types[self.consts[i as usize].0.0 as usize]
                    else {
                        unreachable!()
                    };
                    vec![if bits < 64 {
                        *x as u64 & ((1u64 << bits) - 1)
                    } else {
                        *x as u64
                    }]
                }
                Const::Float(x) => vec![*x as u64],
                Const::Undef => vec![0; 4], // an arbitrary permitted value, never an unmade SSA value
                Const::Aggregate(xs) => {
                    xs.iter().map(|i| self.get(lane, V::Const(*i))[0]).collect()
                }
            },
            V::Func(_) => panic!("function as data"),
        }
    }

    fn enter(&self, lane: &mut Lane, to: usize) {
        let values: Vec<_> = self.blocks[to]
            .phis
            .iter()
            .map(|&i| {
                let Op::Phi(ref incoming) = self.insts[i as usize].op else {
                    unreachable!()
                };
                let v = incoming
                    .iter()
                    .find(|x| x.1 == lane.block)
                    .expect("phi predecessor")
                    .0;
                (i, self.get(lane, v))
            })
            .collect();
        for (i, value) in values {
            lane.values[i as usize] = Some(value);
        }
        lane.block = to;
        lane.at = 0;
    }

    /// `reverse` changes host-wave arrival order at every barrier. There is no
    /// hardware/validator emulation here: GPU fixtures must still validate DXIL.
    pub(crate) fn run(&self, threads: usize, input: &[u32], reverse: bool) -> Execution {
        self.run_with_user_data(threads, input, reverse, &[])
    }

    pub(crate) fn run_with_user_data(&self, threads: usize, input: &[u32], reverse: bool, user: &[u32]) -> Execution {
        let mut lanes: Vec<_> = (0..threads)
            .map(|_| Lane {
                block: 0,
                at: 0,
                done: false,
                values: vec![None; self.insts.len()],
            })
            .collect();
        let mut shared: HashMap<(u64, u64), Data> = HashMap::new();
        let mut memory: HashMap<usize, u32> = input.iter().copied().enumerate().collect();
        let mut out = Execution {
            output: vec![0xcccccccc; threads],
            writes: vec![0; threads],
            barriers: 0,
            steps: vec![0; threads],
            exchange_reads: 0,
            exchange_writes: 0,
            reads: Vec::new(),
            atomics: Vec::new(),
            lds_reads: Vec::new(),
            lds_writes: Vec::new(),
            lds_atomics: Vec::new(),
            memory: HashMap::new(),
        };
        for _ in 0..200_000 {
            let mut waiting = vec![None; threads];
            let order: Vec<_> = if reverse {
                (0..threads).rev().collect()
            } else {
                (0..threads).collect()
            };
            for id in order {
                let lane = &mut lanes[id];
                // Each thread runs until its next collective, even through guards.
                for _ in 0..100_000 {
                    if lane.done {
                        break;
                    }
                    let block = &self.blocks[lane.block];
                    let Some(&i) = block.body.get(lane.at) else {
                        match block.term.expect("terminated block") {
                            Term::Br(to) => self.enter(lane, to),
                            Term::CondBr(c, t, f) => {
                                let to = if self.get(lane, c)[0] != 0 { t } else { f };
                                self.enter(lane, to);
                            }
                            Term::Ret => lane.done = true,
                        }
                        continue;
                    };
                    let inst = &self.insts[i as usize];
                    if let Op::Call(f, _) = inst.op {
                        let name = &self.funcs[f as usize];
                        if name.starts_with("dx.op.wave") || name == "dx.op.barrier" {
                            waiting[id] = Some(i);
                            break;
                        }
                    }
                    let g = |v| self.get(lane, v);
                    out.steps[id] += 1;
                    let n = |v| g(v)[0];
                    let mut value = match &inst.op {
                        Op::Bin(op, a, b) => {
                            let (sa, sb) = (self.signed(*a, n(*a)), self.signed(*b, n(*b)));
                            let (a, b) = (n(*a), n(*b));
                            vec![match op {
                                Bin::Add => a.wrapping_add(b),
                                Bin::Sub => a.wrapping_sub(b),
                                Bin::Mul => a.wrapping_mul(b),
                                Bin::And => a & b,
                                Bin::Or => a | b,
                                Bin::Xor => a ^ b,
                                Bin::Shl => a.wrapping_shl(b as u32),
                                Bin::LShr => a.wrapping_shr(b as u32),
                                Bin::AShr => sa.wrapping_shr(b as u32) as u64,
                                Bin::UDiv => a.checked_div(b).unwrap_or(0),
                                Bin::URem => a.checked_rem(b).unwrap_or(0),
                                Bin::SDiv => sa.checked_div(sb).unwrap_or(0) as u64,
                            }]
                        }
                        Op::Cmp(p, a, b) => {
                            let (sa, sb) = (self.signed(*a, n(*a)), self.signed(*b, n(*b)));
                            let (a, b) = (n(*a), n(*b));
                            vec![u64::from(match *p {
                                pred::EQ => a == b,
                                pred::NE => a != b,
                                pred::IULT => a < b,
                                pred::IULE => a <= b,
                                pred::IUGT => a > b,
                                pred::IUGE => a >= b,
                                pred::SLT => sa < sb,
                                pred::SLE => sa <= sb,
                                pred::SGT => sa > sb,
                                pred::SGE => sa >= sb,
                                _ => panic!("comparison {p}"),
                            })]
                        }
                        Op::Cast(Cast::SExt, a) => vec![self.signed(*a, n(*a)) as u64],
                        Op::Cast(Cast::Bitcast | Cast::Trunc | Cast::ZExt, a) => g(*a),
                        Op::Select(c, t, f) => g(if n(*c) != 0 { *t } else { *f }),
                        Op::Extract(v, k) => vec![g(*v)[*k as usize]],
                        Op::Gep(_, p, indices) => {
                            let mut p = g(*p);
                            p[1] += n(*indices.last().unwrap());
                            p
                        }
                        Op::Store(p, v) => {
                            let p = g(*p);
                            if Some(p[0]) == self.lds {
                                out.lds_writes.push((id, p[1] as usize));
                            } else {
                                out.exchange_writes += 1;
                            }
                            shared.insert((p[0], p[1]), g(*v));
                            vec![0]
                        }
                        Op::Load(p) => {
                            let p = g(*p);
                            if Some(p[0]) == self.lds {
                                out.lds_reads.push((id, p[1] as usize));
                            } else {
                                out.exchange_reads += 1;
                            }
                            shared
                                .get(&(p[0], p[1]))
                                .cloned()
                                .expect("initialized shared memory slot")
                        }
                        Op::AtomicRmw(op, p, v) => {
                            let p = g(*p);
                            assert_eq!(Some(p[0]), self.lds, "LDS atomic fixture");
                            let key = (p[0], p[1]);
                            let old = shared.get(&key).expect("initialized LDS atomic")[0] as u32;
                            let v = n(*v) as u32;
                            let new = match op {
                                Rmw::Xchg => v,
                                Rmw::Add => old.wrapping_add(v),
                                Rmw::Sub => old.wrapping_sub(v),
                                Rmw::And => old & v,
                                Rmw::Or => old | v,
                                Rmw::Xor => old ^ v,
                                Rmw::Min => (old as i32).min(v as i32) as u32,
                                Rmw::Max => (old as i32).max(v as i32) as u32,
                                Rmw::UMin => old.min(v),
                                Rmw::UMax => old.max(v),
                            };
                            shared.insert(key, vec![new as u64]);
                            out.lds_atomics.push((id, p[1] as usize, old, new));
                            vec![old as u64]
                        }
                        Op::CmpXchg(p, expected, new) => {
                            let p = g(*p);
                            assert_eq!(Some(p[0]), self.lds, "LDS compare-exchange fixture");
                            let key = (p[0], p[1]);
                            let old =
                                shared.get(&key).expect("initialized LDS compare-exchange")[0];
                            let success = old == n(*expected);
                            let new = if success { n(*new) } else { old };
                            shared.insert(key, vec![new]);
                            out.lds_atomics
                                .push((id, p[1] as usize, old as u32, new as u32));
                            vec![old, u64::from(success)]
                        }
                        Op::Call(f, args) => {
                            let a: Vec<_> = args.iter().map(|v| n(*v)).collect();
                            let name = self.funcs[*f as usize].as_str();
                            match name {
                                "dx.op.flattenedThreadIdInGroup.i32" => vec![id as u64],
                                "dx.op.threadIdInGroup.i32" => vec![if a[1] == 0 { id as u64 } else { 0 }],
                                "dx.op.groupId.i32" => vec![0],
                                "dx.op.cbufferLoadLegacy.i32" if a[1] == 0 => (0..4)
                                    .map(|k| user.get(a[2] as usize * 4 + k).copied().unwrap_or(0) as u64).collect(),
                                "dx.op.cbufferLoadLegacy.i32" => {
                                    let at = a[2] as usize * 4;
                                    out.reads.push(at);
                                    (0..4)
                                        .map(|k| memory.get(&(at + k)).copied().unwrap_or(0) as u64)
                                        .collect()
                                }
                                "dx.op.createHandleFromBinding" => {
                                    vec![u64::from(g(args[1])[2] != 0)]
                                }
                                "dx.op.createHandleFromHeap" | "dx.op.createHandle" => vec![1],
                                "dx.op.annotateHandle" => g(args[1]),
                                "dx.op.rawBufferStore.i32" => {
                                    let at = a[2] as usize / 4;
                                    assert!(at < threads, "output address {at}");
                                    out.output[at] = a[4] as u32;
                                    out.writes[at] += 1;
                                    vec![0]
                                }
                                "dx.op.rawBufferLoad.i32" | "dx.op.rawBufferLoad.f32" => {
                                    let at = a[2] as usize / 4;
                                    out.reads.push(at);
                                    (0..4)
                                        .map(|k| memory.get(&(at + k)).copied().unwrap_or(0) as u64)
                                        .chain([1])
                                        .collect()
                                }
                                "dx.op.atomicBinOp.i32" => {
                                    let at = a[3] as usize / 4;
                                    let old = memory.get(&at).copied().unwrap_or(0);
                                    let value = a[6] as u32;
                                    let new = match a[2] {
                                        0 => old.wrapping_add(value),
                                        6 => old.min(value),
                                        _ => panic!("unimplemented buffer atomic {}", a[2]),
                                    };
                                    memory.insert(at, new);
                                    out.atomics.push((at, old));
                                    vec![old as u64]
                                }
                                "dx.op.atomicCompareExchange.i32" => {
                                    let at = a[2] as usize / 4;
                                    let old = memory.get(&at).copied().unwrap_or(0);
                                    if old == a[5] as u32 {
                                        memory.insert(at, a[6] as u32);
                                    }
                                    out.atomics.push((at, old));
                                    vec![old as u64]
                                }
                                "dx.op.unaryBits.i32" => {
                                    let x = a[1] as u32;
                                    vec![match a[0] {
                                        31 => x.count_ones(),
                                        32 => {
                                            if x == 0 {
                                                u32::MAX
                                            } else {
                                                x.trailing_zeros()
                                            }
                                        }
                                        33 => {
                                            if x == 0 {
                                                u32::MAX
                                            } else {
                                                31 - x.leading_zeros()
                                            }
                                        }
                                        _ => panic!("unary opcode {}", a[0]),
                                    } as u64]
                                }
                                _ => panic!("unimplemented value-test call {name}: {a:?}"),
                            }
                        }
                        op => panic!("unimplemented value-test op {op:?}"),
                    };
                    if let TypeDef::Int(bits) = self.types[inst.ty.0 as usize]
                        && bits < 64
                    {
                        value[0] &= (1u64 << bits) - 1;
                    }
                    lane.values[i as usize] = Some(value);
                    lane.at += 1;
                }
            }
            if lanes.iter().all(|l| l.done) {
                out.memory = memory;
                return out;
            }
            let mut progress = false;
            // Only a complete group releases a group barrier.
            if let Some(i) = waiting[0]
                && waiting.iter().all(|x| *x == Some(i))
                && matches!(self.insts[i as usize].op,
                    Op::Call(f, _) if self.funcs[f as usize] == "dx.op.barrier")
            {
                out.barriers += 1;
                for lane in &mut lanes {
                    lane.at += 1;
                }
                continue;
            }
            for start in (0..threads).step_by(32) {
                let end = (start + 32).min(threads);
                let Some(i) = waiting[start] else { continue };
                if !waiting[start..end].iter().all(|x| *x == Some(i)) {
                    continue;
                }
                let Op::Call(f, ref args) = self.insts[i as usize].op else {
                    unreachable!()
                };
                let name = self.funcs[f as usize].as_str();
                if name == "dx.op.barrier" {
                    continue;
                }
                let values: Vec<_> = (start..end)
                    .map(|id| {
                        let arg = |n, lane| self.get(&lanes[lane], args[n])[0];
                        match name {
                            "dx.op.waveGetLaneIndex" => vec![(id - start) as u64],
                            "dx.op.waveReadLaneFirst.i32" => vec![arg(1, start)],
                            "dx.op.waveReadLaneAt.i32" => vec![arg(1, start + arg(2, id) as usize)],
                            "dx.op.waveActiveBallot" => vec![
                                (start..end)
                                    .fold(0, |m, l| m | (u64::from(arg(1, l) != 0) << (l - start))),
                                0,
                                0,
                                0,
                            ],
                            _ => panic!("unimplemented value-test wave op {name}"),
                        }
                    })
                    .collect();
                for (lane, v) in lanes[start..end].iter_mut().zip(values) {
                    lane.values[i as usize] = Some(v);
                    lane.at += 1;
                }
                progress = true;
            }
            assert!(progress, "collective deadlock: {waiting:?}");
        }
        panic!("value-test instruction budget exceeded");
    }
}
