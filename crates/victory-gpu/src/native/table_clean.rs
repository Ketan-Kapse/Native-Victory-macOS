//! Clean tables are valid only while preparing one submission. Plain title stores have no
//! notification; the next submission must read the records again, even with unchanged epochs.
use std::{cell::{Cell, RefCell}, collections::HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Dependencies {
    pub memory: u64,
    pub resources: u64,
    pub spans: u64,
}

type Table = (u64, usize, usize); // heap address, first entry, exclusive end

#[derive(Default)]
pub(super) struct CleanTables {
    active: Cell<bool>,
    entries: RefCell<HashMap<Table, Dependencies>>,
}

pub(super) struct Preparation<'a>(&'a CleanTables);

impl Drop for Preparation<'_> {
    fn drop(&mut self) {
        self.0.active.set(false);
        self.0.entries.borrow_mut().clear();
    }
}

impl CleanTables {
    pub fn begin(&self, enabled: bool) -> Preparation<'_> {
        self.entries.borrow_mut().clear();
        self.active.set(enabled);
        Preparation(self)
    }

    pub fn contains(&self, table: Table, dependencies: Dependencies) -> bool {
        self.active.get() && self.entries.borrow().get(&table) == Some(&dependencies)
    }

    pub fn mark(&self, table: Table, dependencies: Dependencies) {
        if self.active.get() { self.entries.borrow_mut().insert(table, dependencies); }
    }

    pub fn invalidate(&self, heap: u64, first: usize, end: usize) {
        if self.active.get() {
            self.entries.borrow_mut().retain(|&(h, a, b), _| h != heap || end <= a || b <= first);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boundaries_dependencies_and_overlapping_writes() {
        let clean = CleanTables::default();
        let deps = Dependencies { memory: 1, resources: 2, spans: 3 };
        let table = (100, 4, 16);
        clean.mark(table, deps);
        assert!(!clean.contains(table, deps));
        {
            let _scope = clean.begin(true);
            clean.mark(table, deps);
            assert!(clean.contains(table, deps));
            assert!(!clean.contains((100, 4, 17), deps));
            assert!(!clean.contains((101, 4, 16), deps));
            for changed in [Dependencies { memory: 2, ..deps }, Dependencies { resources: 3, ..deps }, Dependencies { spans: 4, ..deps }] {
                assert!(!clean.contains(table, changed));
            }
            for range in [(100, 0, 4), (100, 16, 20), (101, 4, 16)] {
                clean.invalidate(range.0, range.1, range.2);
                assert!(clean.contains(table, deps));
            }
            for range in [(3, 5), (15, 17), (5, 6), (0, 20)] {
                clean.mark(table, deps);
                clean.invalidate(100, range.0, range.1);
                assert!(!clean.contains(table, deps));
            }
            clean.mark(table, deps);
        }
        assert!(!clean.contains(table, deps));
        let _scope = clean.begin(true);
        assert!(!clean.contains(table, deps));
    }

    #[test]
    fn disabled_never_skips() {
        let clean = CleanTables::default();
        let _scope = clean.begin(false);
        let deps = Dependencies { memory: 1, resources: 1, spans: 1 };
        clean.mark((1, 0, 16), deps);
        assert!(!clean.contains((1, 0, 16), deps));
    }
}
