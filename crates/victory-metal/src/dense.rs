//! Dense descriptor-heap ownership: what each entry names, as a small id into a per-heap slab of
//! allocations with reference counts, instead of a hash map by entry index and another by
//! allocation pointer. An entry that keeps naming an allocation the heap already holds (the pool
//! buffer of every buffer view into one pool) costs an index and an increment: no hash, no retain,
//! no allocation.
//!
//! The residency rules of `Heap::commit_set` are kept by construction:
//! - each allocation carries the membership Metal last saw (`committed`); [`Dense::reconcile`]
//!   compares it with the desired one (`count != 0`) and issues at most one add or one remove per
//!   allocation, never both, never an add of a member or a remove of a non-member;
//! - an allocation no entry names any more stays owned by its slab slot until
//!   [`Dense::finish_commit`], which the caller runs only after the set was published, so a
//!   pending removal always names a live object and its address cannot be reused meanwhile.
use std::collections::HashMap;

const PAGE: usize = 4096;
/// Allocations remembered without hashing: the pools a ring heap's fresh entries alternate over.
const RECENT: usize = 4;

struct Slot<V> {
    /// Index into `allocations`; 0 names nothing.
    allocation: u32,
    view: Option<V>,
}

struct Allocation<R> {
    key: usize,
    resource: R,
    /// Entries naming it: the desired membership is `count != 0`.
    count: usize,
    /// Whether the residency set holds it as of the last reconciliation.
    committed: bool,
}

/// What one entry write changed, for the heap's dirty flag and typed-view count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Change {
    /// An allocation went from no entries to one.
    pub added: bool,
    /// An allocation lost its last entry.
    pub retired: bool,
}

pub(crate) struct Dense<R, V> {
    pages: Vec<Option<Box<[Slot<V>]>>>,
    len: usize,
    /// Slot 0 is never used: id 0 means "names nothing".
    allocations: Vec<Option<Allocation<R>>>,
    free: Vec<u32>,
    ids: HashMap<usize, u32>,
    recent: [u32; RECENT],
    entries: usize,
    typed: usize,
    /// Allocations with entries (the old `alloc_counts.len()`).
    live: usize,
    /// Allocations with no entry, still owned for a pending reconciliation.
    retired: usize,
    /// Views of entries whose allocation retired, owned as long as it is.
    views: Vec<V>,
    /// Whether retired allocations wait for a commit (a residency set exists). Without one they
    /// are released at once, as the per-encoder naming needs nothing kept.
    keep_retired: bool,
}

impl<R, V> Dense<R, V> {
    pub fn new(len: usize, keep_retired: bool) -> Self {
        Self {
            pages: std::iter::repeat_with(|| None).take(len.div_ceil(PAGE)).collect(),
            len,
            allocations: vec![None],
            free: Vec::new(),
            ids: HashMap::new(),
            recent: [0; RECENT],
            entries: 0,
            typed: 0,
            live: 0,
            retired: 0,
            views: Vec::new(),
            keep_retired,
        }
    }

    #[inline]
    fn slot(&self, index: usize) -> Option<&Slot<V>> {
        self.pages.get(index / PAGE)?.as_ref()?.get(index % PAGE)
    }

    /// The allocation key, resource and view of entry `index`, if it names an allocation.
    #[inline]
    pub fn get(&self, index: usize) -> Option<(usize, &R, Option<&V>)> {
        let slot = self.slot(index)?;
        let a = self.allocations[slot.allocation as usize].as_ref()?;
        Some((a.key, &a.resource, slot.view.as_ref()))
    }

    /// Whether entry `index` names allocation `key` with no view: the raw-buffer rewrite.
    #[inline]
    pub fn is_raw(&self, index: usize, key: usize) -> bool {
        self.slot(index).is_some_and(|slot| {
            slot.view.is_none() && self.allocations[slot.allocation as usize].as_ref().is_some_and(|a| a.key == key)
        })
    }

    /// Every entry that names an allocation, in index order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &R, Option<&V>)> {
        let allocations = &self.allocations;
        self.pages.iter().enumerate().flat_map(move |(p, page)| {
            page.iter().flat_map(|slots| slots.iter().enumerate()).filter_map(move |(i, slot)| {
                let a = allocations[slot.allocation as usize].as_ref()?;
                Some((p * PAGE + i, a.key, &a.resource, slot.view.as_ref()))
            })
        })
    }

    /// The allocations entries name, with how many name each.
    pub fn counts(&self) -> impl Iterator<Item = (usize, usize, &R)> {
        self.allocations.iter().flatten().filter(|a| a.count != 0).map(|a| (a.key, a.count, &a.resource))
    }

    pub fn entries(&self) -> usize { self.entries }
    pub fn typed(&self) -> usize { self.typed }
    pub fn live(&self) -> usize { self.live }
    pub fn retired(&self) -> usize { self.retired }
    pub fn retired_views(&self) -> usize { self.views.len() }

    /// The keys the residency set holds as of the last reconciliation.
    #[cfg(test)]
    pub fn membership(&self) -> std::collections::HashSet<usize> {
        self.allocations.iter().flatten().filter(|a| a.committed).map(|a| a.key).collect()
    }

    /// The slab id of allocation `key`, made (with no entries yet) if the heap does not hold it.
    #[inline]
    fn id(&mut self, key: usize, make: impl FnOnce() -> R) -> u32 {
        for i in 0..RECENT {
            let id = self.recent[i];
            if id != 0 && self.allocations[id as usize].as_ref().is_some_and(|a| a.key == key) {
                if i != 0 { self.recent.swap(0, i); }
                return id;
            }
        }
        self.id_slow(key, make)
    }

    #[cold]
    fn id_slow(&mut self, key: usize, make: impl FnOnce() -> R) -> u32 {
        let id = match self.ids.get(&key) {
            Some(id) => *id,
            None => {
                let a = Some(Allocation { key, resource: make(), count: 0, committed: false });
                let id = match self.free.pop() {
                    Some(id) => {
                        self.allocations[id as usize] = a;
                        id
                    }
                    None => {
                        let id = u32::try_from(self.allocations.len()).expect("allocation slab fits u32");
                        self.allocations.push(a);
                        id
                    }
                };
                self.ids.insert(key, id);
                // No entry yet: the caller's increment takes it out of the retired count again.
                self.retired += 1;
                id
            }
        };
        self.recent.rotate_right(1);
        self.recent[0] = id;
        id
    }

    /// Entry `index` now names allocation `key` (none: nothing) through `view`. `make` produces
    /// the owned resource only when the heap does not hold `key` yet (a retired allocation not
    /// yet reconciled is still held, so naming it again cancels its removal).
    #[inline]
    pub fn set(&mut self, index: usize, key: Option<usize>, make: impl FnOnce() -> R, view: Option<V>) -> Change {
        assert!(index < self.len, "descriptor index {index} out of {}", self.len);
        let page = &mut self.pages[index / PAGE];
        if page.is_none() {
            if key.is_none() { return Change::default(); }
            let first = index - index % PAGE;
            *page = Some((0..PAGE.min(self.len - first)).map(|_| Slot { allocation: 0, view: None }).collect());
        }
        let old_id = page.as_ref().expect("made")[index % PAGE].allocation;
        let old_key = self.allocations[old_id as usize].as_ref().map(|a| a.key);
        let new_id = if old_key == key { old_id } else { key.map_or(0, |k| self.id(k, make)) };
        let mut change = Change::default();
        if old_id != new_id {
            if new_id != 0 {
                let a = self.allocations[new_id as usize].as_mut().expect("named allocation");
                if a.count == 0 {
                    self.retired -= 1;
                    self.live += 1;
                    change.added = true;
                }
                a.count += 1;
            }
            if old_id != 0 {
                let a = self.allocations[old_id as usize].as_mut().expect("counted allocation");
                a.count -= 1;
                if a.count == 0 {
                    self.retired += 1;
                    self.live -= 1;
                    change.retired = true;
                }
            }
            self.entries = self.entries + usize::from(new_id != 0) - usize::from(old_id != 0);
        }
        let slot = &mut self.pages[index / PAGE].as_mut().expect("made")[index % PAGE];
        slot.allocation = new_id;
        if slot.view.is_some() || view.is_some() {
            self.typed = self.typed + usize::from(view.is_some()) - usize::from(slot.view.is_some());
            let old_view = std::mem::replace(&mut slot.view, view);
            // The last entry of an allocation: its view waits for the commit with it.
            if change.retired && self.keep_retired && let Some(old_view) = old_view {
                self.views.push(old_view);
            }
        }
        if change.retired && !self.keep_retired { drop(self.release(old_id)); }
        change
    }

    /// Slab slot `id` has no entry and nothing pending in Metal: the heap lets the object go.
    fn release(&mut self, id: u32) -> R {
        let a = self.allocations[id as usize].take().expect("a retired allocation");
        assert!(a.count == 0 && !a.committed, "only an allocation Metal no longer holds is released");
        self.ids.remove(&a.key);
        self.free.push(id);
        self.retired -= 1;
        for recent in &mut self.recent {
            if *recent == id { *recent = 0; }
        }
        a.resource
    }

    /// Brings Metal's membership to the desired one: one `add` for each allocation with entries
    /// the set does not hold, one `remove` for each it holds that has none. The caller publishes
    /// the set (if either count is not 0) and only then calls [`Self::finish_commit`].
    pub fn reconcile(&mut self, mut add: impl FnMut(&R), mut remove: impl FnMut(&R)) -> (u64, u64) {
        let (mut adds, mut removes) = (0, 0);
        for a in self.allocations.iter_mut().flatten() {
            match (a.committed, a.count != 0) {
                (false, true) => {
                    add(&a.resource);
                    adds += 1;
                }
                (true, false) => {
                    remove(&a.resource);
                    removes += 1;
                }
                _ => {}
            }
            a.committed = a.count != 0;
        }
        (adds, removes)
    }

    /// The reconciled membership is published (or had no difference): the heap lets its retired
    /// objects and their views go, and returns them. Work submitted before may still read them
    /// (the caller keeps them until it has finished).
    pub fn finish_commit(&mut self) -> (Vec<R>, Vec<V>) {
        let mut gone = Vec::new();
        if self.retired != 0 {
            for id in 1..self.allocations.len() {
                if self.allocations[id].as_ref().is_some_and(|a| a.count == 0) {
                    gone.push(self.release(id as u32));
                }
            }
        }
        debug_assert_eq!(self.retired, 0);
        (gone, std::mem::take(&mut self.views))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::{reconcile, replace};
    use std::{collections::HashSet, rc::Rc};

    /// The old maps and the dense state under the same random writes: counts after every write,
    /// the adds and removes of every commit, and the committed membership.
    #[test]
    fn random_sequences_match_old_maps_and_retain_through_commit() {
        const SLOTS: usize = 8195;
        let mut dense = Dense::<Rc<usize>, Rc<usize>>::new(SLOTS, true);
        let mut slots: Vec<Option<(usize, bool)>> = vec![None; SLOTS];
        let (mut counts, mut retired, mut membership) = (HashMap::new(), HashMap::new(), HashSet::new());
        // One object per key while the old maps hold it, as one Metal object has one address.
        let mut objects: HashMap<usize, Rc<usize>> = HashMap::new();
        let mut seed = 0x243f6a8885a308d3u64;
        let mut commits = 0;
        for n in 0..200_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // Fewer slots than keys (both ends and across a page boundary), so allocations do
            // lose their last entry between commits.
            let i = match (seed >> 20) as usize % 14 { 12 => 0, 13 => SLOTS - 1, w => 4090 + w };
            let key = (!(seed >> 16).is_multiple_of(4)).then_some((seed >> 32) as usize % 23);
            let typed = key.is_some() && (seed >> 40).is_multiple_of(3);
            let resource = key.map(|k| objects.entry(k).or_insert_with(|| Rc::new(k)).clone());
            let (added, gone) = replace(&mut counts, slots[i].map(|(k, _)| k), key.zip(resource.as_ref()));
            let old_retired = gone.is_some();
            if let Some((id, r)) = gone { retired.insert(id, r); }
            let made = std::cell::Cell::new(false);
            let change = dense.set(i, key, || { made.set(true); resource.clone().expect("a key") }, typed.then(|| Rc::new(n)));
            assert_eq!(change, Change { added, retired: old_retired });
            if made.get() { assert!(added, "an object is taken only for an allocation the heap did not hold"); }
            slots[i] = key.map(|k| (k, typed));
            assert_eq!(dense.get(i).map(|(k, r, v)| (k, **r, v.is_some())), key.map(|k| (k, k, typed)));
            if n % 64 == 0 {
                assert_eq!(dense.counts().map(|(k, n, _)| (k, n)).collect::<HashMap<_, _>>(), counts.iter().map(|(&k, (n, _))| (k, *n)).collect());
                assert_eq!(dense.live(), counts.len());
                assert_eq!(dense.entries(), slots.iter().flatten().count());
                assert_eq!(dense.typed(), slots.iter().flatten().filter(|s| s.1).count());
                assert_eq!(dense.iter().map(|(i, k, _, v)| (i, (k, v.is_some()))).collect::<Vec<_>>(),
                    slots.iter().enumerate().filter_map(|(i, s)| s.map(|s| (i, s))).collect::<Vec<_>>());
            }
            // Every member, old and new, is owned until the commit that removes it.
            for id in &membership { assert!(counts.contains_key(id) || retired.contains_key(id)); }
            if n % 37 == 0 || n == 199_999 {
                let (mut adds, mut removes) = (HashSet::new(), HashSet::new());
                let before = dense.membership();
                dense.reconcile(
                    |r| { assert!(adds.insert(**r), "added twice"); assert!(!before.contains(&**r), "added a member"); },
                    |r| { assert!(removes.insert(**r), "removed twice"); assert!(before.contains(&**r), "removed a non-member"); },
                );
                assert!(adds.is_disjoint(&removes));
                let (mut old_adds, mut old_removes) = (HashSet::new(), HashSet::new());
                reconcile(&counts, &mut membership, &retired, |r| { old_adds.insert(**r); }, |r| { old_removes.insert(**r); });
                commits += usize::from(!old_removes.is_empty());
                assert_eq!((adds, removes), (old_adds, old_removes));
                assert_eq!(dense.membership(), membership);
                dense.finish_commit();
                retired.clear();
                assert_eq!((dense.retired(), dense.retired_views()), (0, 0));
                objects.retain(|k, _| counts.contains_key(k));
            }
        }
        assert!(commits > 100, "the sequence must exercise removals: {commits}");
    }

    #[test]
    fn retirement_readdition_recycling_and_same_pool_do_not_clone() {
        let mut dense = Dense::<Rc<usize>, Rc<usize>>::new(8195, true);
        let resource = Rc::new(7);
        let weak = Rc::downgrade(&resource);
        dense.set(4096, Some(7), || resource.clone(), None);
        dense.set(8194, Some(7), || panic!("the heap already holds it"), None);
        dense.set(4096, Some(7), || panic!("same allocation"), None);
        assert!(dense.is_raw(4096, 7) && !dense.is_raw(4096, 8) && !dense.is_raw(0, 7));
        assert_eq!(Rc::strong_count(&resource), 2);
        assert_eq!(dense.reconcile(|_| {}, |_| unreachable!()), (1, 0));
        dense.finish_commit();
        drop(resource);
        dense.set(4096, None, || unreachable!(), None);
        assert_eq!(dense.set(8194, None, || unreachable!(), None), Change { added: false, retired: true });
        assert!(weak.upgrade().is_some(), "owned until the removal is committed");
        // Named again before the commit: the removal is cancelled, with the object the heap kept.
        assert_eq!(dense.set(0, Some(7), || panic!("retired allocation reused"), None), Change { added: true, retired: false });
        assert_eq!(dense.reconcile(|_| unreachable!(), |_| unreachable!()), (0, 0));
        dense.finish_commit();
        dense.set(0, None, || unreachable!(), None);
        assert_eq!(dense.reconcile(|_| unreachable!(), |_| assert!(weak.upgrade().is_some())), (0, 1));
        assert!(weak.upgrade().is_some());
        dense.finish_commit();
        assert!(weak.upgrade().is_none());
        // The slab slot and the recent list do not resurrect the old key.
        dense.set(0, Some(8), || Rc::new(8), None);
        assert_eq!(dense.counts().map(|(k, n, _)| (k, n)).collect::<Vec<_>>(), [(8, 1)]);
        assert!(dense.get(8194).is_none());
    }

    #[test]
    fn views_wait_with_their_retired_allocation_and_no_set_releases_at_once() {
        let mut dense = Dense::<Rc<usize>, Rc<usize>>::new(4, true);
        let (buffer, view) = (Rc::new(1), Rc::new(2));
        dense.set(0, Some(1), || buffer.clone(), Some(view.clone()));
        dense.set(1, Some(1), || unreachable!(), None);
        assert_eq!(dense.typed(), 1);
        // Another entry still names the buffer: the replaced view is not kept.
        dense.set(0, Some(1), || unreachable!(), None);
        assert_eq!((Rc::strong_count(&view), dense.retired_views(), dense.typed()), (1, 0, 0));
        dense.set(1, Some(1), || unreachable!(), Some(view.clone()));
        dense.set(0, None, || unreachable!(), None);
        dense.set(1, None, || unreachable!(), None);
        assert_eq!((dense.retired(), dense.retired_views(), Rc::strong_count(&view), Rc::strong_count(&buffer)), (1, 1, 2, 2));
        dense.reconcile(|_| unreachable!(), |_| unreachable!());
        dense.finish_commit();
        assert_eq!((Rc::strong_count(&view), Rc::strong_count(&buffer)), (1, 1));

        let mut unset = Dense::<Rc<usize>, Rc<usize>>::new(4, false);
        unset.set(0, Some(1), || buffer.clone(), Some(view.clone()));
        unset.set(0, None, || unreachable!(), None);
        assert_eq!((unset.retired(), unset.retired_views(), Rc::strong_count(&view), Rc::strong_count(&buffer)), (0, 0, 1, 1));
        unset.set(3, Some(1), || buffer.clone(), None);
        assert_eq!(unset.counts().count(), 1);
    }
}
