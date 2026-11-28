//! Allocation transitions shared by descriptor writes and CPU lifetime tests.
use std::collections::{HashMap, HashSet};

/// Change one entry's allocation. Equal allocations (including changed views of one buffer)
/// need no count change. The returned last reference must be retained until reconciliation.
pub(crate) fn replace<R: Clone>(
    counts: &mut HashMap<usize, (usize, R)>,
    old: Option<usize>,
    new: Option<(usize, &R)>,
) -> (bool, Option<(usize, R)>) {
    if old == new.map(|(id, _)| id) { return (false, None); }
    let mut added = false;
    if let Some((id, resource)) = new {
        let (n, _) = counts.entry(id).or_insert_with(|| (0, resource.clone()));
        added = *n == 0;
        *n += 1;
    }
    let removed = old.and_then(|id| {
        let (n, _) = counts.get_mut(&id).expect("entry allocation is counted");
        *n -= 1;
        (*n == 0).then(|| (id, counts.remove(&id).expect("counted").1))
    });
    (added, removed)
}

/// Only the final desired membership reaches Metal. No allocation can be added and removed
/// between two commits, even after A -> B -> A or many writes to the same ring slot.
pub(crate) fn reconcile<R>(
    counts: &HashMap<usize, (usize, R)>,
    in_set: &mut HashSet<usize>,
    retired: &HashMap<usize, R>,
    mut add: impl FnMut(&R),
    mut remove: impl FnMut(&R),
) -> (u64, u64) {
    let (mut adds, mut removes) = (0, 0);
    for (&id, (_, allocation)) in counts {
        if in_set.insert(id) { add(allocation); adds += 1; }
    }
    in_set.retain(|id| {
        if counts.contains_key(id) { return true; }
        remove(retired.get(id).expect("set allocation must be retained"));
        removes += 1;
        false
    });
    (adds, removes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn random_writes_match_recount_and_committed_membership() {
        let mut slots = [None; 97];
        let (mut counts, mut retired, mut in_set) = (HashMap::new(), HashMap::new(), HashSet::new());
        let mut seed = 0x243f6a8885a308d3u64;
        for n in 0..100_000 {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            let index = seed as usize % slots.len();
            let next = (!(seed >> 16).is_multiple_of(31)).then_some((seed >> 32) as usize % 23);
            let resource = next.map(Rc::new);
            let (_, gone) = replace(&mut counts, slots[index], next.zip(resource.as_ref()));
            if let Some((id, resource)) = gone { retired.insert(id, resource); }
            slots[index] = next;
            let mut old_bookkeeping = HashMap::new();
            for id in slots.iter().flatten() { *old_bookkeeping.entry(*id).or_insert(0) += 1; }
            assert_eq!(counts.iter().map(|(&id, (n, _))| (id, *n)).collect::<HashMap<_, _>>(), old_bookkeeping);
            for id in &in_set { assert!(counts.contains_key(id) || retired.contains_key(id)); }
            if n % 37 == 0 || n == 99_999 {
                let (mut adds, mut removes) = (HashSet::new(), HashSet::new());
                reconcile(&counts, &mut in_set, &retired,
                    |r| { assert!(adds.insert(**r)); }, |r| { assert!(removes.insert(**r)); });
                assert!(adds.is_disjoint(&removes));
                assert_eq!(in_set, old_bookkeeping.keys().copied().collect());
                retired.clear();
            }
        }
    }

    #[test]
    fn removed_allocation_is_owned_through_commit_and_readdition_cancels_removal() {
        let resource = Rc::new(7);
        let weak = Rc::downgrade(&resource);
        let (mut counts, mut retired, mut in_set) = (HashMap::new(), HashMap::new(), HashSet::new());
        replace(&mut counts, None, Some((7, &resource)));
        reconcile(&counts, &mut in_set, &retired, |_| {}, |_| unreachable!());
        drop(resource);
        let (_, gone) = replace(&mut counts, Some(7), None);
        let (id, resource) = gone.unwrap();
        retired.insert(id, resource);
        assert!(weak.upgrade().is_some());
        replace(&mut counts, None, Some((7, &weak.upgrade().unwrap())));
        assert_eq!(reconcile(&counts, &mut in_set, &retired, |_| unreachable!(), |_| unreachable!()), (0, 0));
        retired.clear();
        let (_, gone) = replace(&mut counts, Some(7), None);
        let (id, resource) = gone.unwrap();
        retired.insert(id, resource);
        assert_eq!(reconcile(&counts, &mut in_set, &retired, |_| unreachable!(), |_| assert!(weak.upgrade().is_some())), (0, 1));
        retired.clear();
        assert!(weak.upgrade().is_none());
    }
}
