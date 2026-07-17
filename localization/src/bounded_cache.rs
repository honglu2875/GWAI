//! Small deterministic caches for target-specific reconstruction data.
//!
//! Reconstruction objects can be large, while their cache keys include target
//! parameters and truncation orders.  A process that explores many targets
//! must therefore not retain every object forever.  [`BoundedCache`] uses
//! insertion-order (FIFO) eviction: reads do not perturb the order, replacing
//! an existing key keeps its position, and inserting a new key at capacity
//! evicts the oldest entry.  The deliberately simple `VecDeque` storage keeps
//! both eviction and iteration deterministic; these caches are small enough
//! that linear lookup is preferable to duplicating keys in a hash map and an
//! order index.

use std::collections::VecDeque;

/// Shared bound for caches whose keys contain target/calibration parameters.
pub(crate) const TARGET_RECONSTRUCTION_CACHE_CAPACITY: usize = 64;

/// A fixed-capacity, insertion-ordered cache.
#[derive(Debug, Clone)]
pub(crate) struct BoundedCache<K, V> {
    capacity: usize,
    entries: VecDeque<(K, V)>,
}

impl<K, V> BoundedCache<K, V> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(key, value)| (key, value))
    }
}

impl<K: Eq, V> BoundedCache<K, V> {
    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.entries
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, value)| value)
    }

    /// Insert `value`, returning the entry evicted at the capacity boundary.
    ///
    /// Replacing an existing key is not an eviction and does not change its
    /// FIFO position.  A zero-capacity cache simply returns the supplied entry.
    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<(K, V)> {
        if self.capacity == 0 {
            return Some((key, value));
        }
        if let Some((_, current)) = self
            .entries
            .iter_mut()
            .find(|(candidate, _)| candidate == &key)
        {
            *current = value;
            return None;
        }

        let evicted = (self.entries.len() == self.capacity)
            .then(|| self.entries.pop_front())
            .flatten();
        self.entries.push_back((key, value));
        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_oldest_entry_deterministically() {
        let mut cache = BoundedCache::new(2);
        assert_eq!(cache.insert("first", 1), None);
        assert_eq!(cache.insert("second", 2), None);
        assert_eq!(cache.insert("third", 3), Some(("first", 1)));

        assert_eq!(cache.get(&"first"), None);
        assert_eq!(cache.get(&"second"), Some(&2));
        assert_eq!(cache.get(&"third"), Some(&3));
        assert_eq!(cache.len(), 2);
        assert_eq!(
            cache.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            vec!["second", "third"]
        );
    }

    #[test]
    fn replacement_preserves_fifo_position() {
        let mut cache = BoundedCache::new(2);
        cache.insert("first", 1);
        cache.insert("second", 2);
        assert_eq!(cache.insert("first", 10), None);
        assert_eq!(cache.insert("third", 3), Some(("first", 10)));

        assert_eq!(cache.get(&"first"), None);
        assert_eq!(cache.get(&"second"), Some(&2));
        assert_eq!(cache.get(&"third"), Some(&3));
    }

    #[test]
    fn zero_capacity_never_retains_entries() {
        let mut cache = BoundedCache::new(0);
        assert_eq!(cache.insert("unused", 7), Some(("unused", 7)));
        assert_eq!(cache.get(&"unused"), None);
        assert_eq!(cache.len(), 0);
    }
}
