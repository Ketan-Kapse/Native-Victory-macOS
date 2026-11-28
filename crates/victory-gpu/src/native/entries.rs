//! Per-entry host state of a descriptor heap, by entry index: lazily made pages of slots (a heap
//! has a known entry count, and a ring heap's fresh entries each touched several hash maps), or
//! with `VICTORY_DENSE_ENTRIES=0` an ordered map as the reference. Both iterate in index order,
//! so nothing a caller does depends on which one holds the state.
use crate::descriptor::paged::Paged;
use std::collections::BTreeMap;

pub struct Entries<T> {
    /// The slots of indices below `slots`, unless this is the reference.
    dense: Option<Paged<T>>,
    slots: usize,
    /// Everything in the reference; otherwise only indices past the heap's end, which no caller
    /// uses today and which must not be lost or panic if one ever does.
    sparse: BTreeMap<usize, T>,
    occupied: usize,
}

impl<T> Entries<T> {
    pub fn new(count: usize, dense: bool) -> Self {
        Self { dense: dense.then(|| Paged::new(count)), slots: count, sparse: BTreeMap::new(), occupied: 0 }
    }

    #[inline]
    pub fn get(&self, index: &usize) -> Option<&T> {
        if self.occupied == 0 { return None; }
        match &self.dense {
            Some(d) if *index < self.slots => d.get(*index),
            _ => self.sparse.get(index),
        }
    }

    #[inline]
    pub fn contains_key(&self, index: &usize) -> bool { self.get(index).is_some() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.occupied == 0 }
    pub fn len(&self) -> usize { self.occupied }

    #[inline]
    pub fn insert(&mut self, index: usize, value: T) -> Option<T> {
        let old = match self.dense.as_mut().and_then(|d| d.slot(index, true)) {
            Some(slot) => slot.replace(value),
            None => self.sparse.insert(index, value),
        };
        self.occupied += usize::from(old.is_none());
        old
    }

    #[inline]
    pub fn remove(&mut self, index: &usize) -> Option<T> {
        if self.occupied == 0 { return None; }
        let old = match &mut self.dense {
            Some(d) if *index < self.slots => d.slot(*index, false).and_then(Option::take),
            _ => self.sparse.remove(index),
        };
        self.occupied -= usize::from(old.is_some());
        old
    }

    /// Reads of neighbouring entries (a table's span) that find a dense page once.
    pub fn cursor(&self) -> Cursor<'_, T> {
        Cursor { entries: self, first: usize::MAX, page: None }
    }

    /// Every held entry in index order.
    pub fn iter(&self) -> Box<dyn Iterator<Item = (usize, &T)> + '_> {
        let sparse = self.sparse.iter().map(|(&i, v)| (i, v));
        match &self.dense {
            Some(d) if self.occupied != 0 => Box::new(d.iter().chain(sparse)),
            Some(_) => Box::new(std::iter::empty()),
            None => Box::new(sparse),
        }
    }

    pub fn values(&self) -> impl Iterator<Item = &T> { self.iter().map(|(_, v)| v) }

    /// The held entries of `range` in index order (a bound table's span: no allocation).
    pub fn for_each_in(&self, range: std::ops::Range<usize>, mut f: impl FnMut(usize, &T)) {
        if self.occupied == 0 { return; }
        if let Some(d) = &self.dense {
            (range.start..range.end.min(self.slots)).for_each(|i| {
                if let Some(v) = d.get(i) { f(i, v); }
            });
        }
        if !self.sparse.is_empty() {
            self.sparse.range(range).for_each(|(&i, v)| f(i, v));
        }
    }
}

/// See [`Entries::cursor`].
pub struct Cursor<'a, T> {
    entries: &'a Entries<T>,
    /// The first index of the page looked up last and its slots (none: no such page).
    first: usize,
    page: Option<&'a [Option<T>]>,
}

impl<'a, T> Cursor<'a, T> {
    #[inline]
    pub fn get(&mut self, index: usize) -> Option<&'a T> {
        let entries = self.entries;
        if entries.occupied == 0 { return None; }
        let Some(dense) = entries.dense.as_ref().filter(|_| index < entries.slots) else { return entries.sparse.get(&index) };
        if let Some(page) = self.page
            && let Some(slot) = index.checked_sub(self.first).and_then(|i| page.get(i))
        {
            return slot.as_ref();
        }
        match dense.page_of(index) {
            Some((first, page)) => {
                (self.first, self.page) = (first, Some(page));
                page.get(index - first)?.as_ref()
            }
            None => None,
        }
    }
}

impl<T> std::ops::Index<&usize> for Entries<T> {
    type Output = T;
    fn index(&self, index: &usize) -> &T { self.get(index).expect("occupied entry") }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dense pages against the reference map under random inserts and removes.
    #[test]
    fn dense_pages_match_the_reference_map() {
        const N: usize = 9000;
        let (mut dense, mut sparse) = (Entries::new(N, true), Entries::new(N, false));
        let mut seed = 0x243f6a8885a308d3u64;
        for n in 0..50_000u64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // Around a page boundary, anywhere, and now and then past the end.
            let i = match seed % 8 {
                0..4 => 4090 + (seed >> 8) as usize % 12,
                4 => N + (seed >> 8) as usize % 3,
                _ => (seed >> 8) as usize % N,
            };
            match (seed >> 40) % 3 {
                0 => assert_eq!(dense.remove(&i), sparse.remove(&i)),
                _ => assert_eq!(dense.insert(i, n), sparse.insert(i, n)),
            }
            assert_eq!((dense.get(&i), dense.len(), dense.is_empty()), (sparse.get(&i), sparse.len(), sparse.is_empty()));
            let (mut a, mut b) = (dense.cursor(), sparse.cursor());
            for j in [i, i + 1, i.saturating_sub(1), (i + 4096) % N, i, N + 5] {
                assert_eq!((a.get(j), b.get(j)), (dense.get(&j), sparse.get(&j)), "cursor at {j}");
            }
            if n % 997 == 0 {
                assert!(dense.iter().eq(sparse.iter()));
                assert!(dense.values().eq(sparse.values()));
                let (mut a, mut b) = (Vec::new(), Vec::new());
                for range in [4000..4200, N - 20..N + 2, N..N + 9] {
                    dense.for_each_in(range.clone(), |i, v| a.push((i, *v)));
                    sparse.for_each_in(range, |i, v| b.push((i, *v)));
                }
                assert_eq!(a, b);
            }
        }
    }
}
