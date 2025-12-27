//! Dynamic DAWG u64 zipper implementation.
//!
//! This module provides a zipper implementation for DynamicDawgU64 that uses
//! Arc-based node references for lock-free navigation. Unlike index-based
//! zippers, this directly references the nodes via Arc, enabling wait-free
//! concurrent reads without any locking.

use crate::dictionary::dynamic_dawg_u64::{DawgNodeU64, DynamicDawgU64};
use crate::dictionary::value::DictionaryValue;
use crate::dictionary::zipper::{DictZipper, ValuedDictZipper};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Zipper for lock-free Dynamic DAWG u64 dictionaries.
///
/// `DynamicDawgU64Zipper` provides efficient navigation through Dynamic DAWG structures
/// using Arc-based node references. This enables wait-free concurrent access - no locks,
/// no blocking, no retries.
///
/// # Design
///
/// The zipper stores:
/// - `node`: Arc reference to the current node
/// - `path`: Path from root to current position (Vec<u64>)
///
/// All operations are wait-free, using only atomic loads to read node state.
///
/// # Thread Safety
///
/// The zipper is fully thread-safe:
/// - Multiple zippers can navigate concurrently (no contention)
/// - Zippers can navigate while writers are modifying the DAWG
/// - Writers may add edges/nodes; readers see a consistent snapshot
///
/// # Performance
///
/// - Wait-free: No locks, no CAS retries for reads
/// - Arc-based: Direct node access without indirection through indices
/// - Lightweight Clone: Arc clone + path clone
///
/// # Example
///
/// ```ignore
/// use liblevenshtein::dictionary::DictZipper;
/// use liblevenshtein::dictionary::dynamic_dawg_u64::DynamicDawgU64;
/// use liblevenshtein::dictionary::dynamic_dawg_u64_zipper::DynamicDawgU64Zipper;
///
/// let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
/// dict.insert_sequence(&[100, 200, 300]);
///
/// let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);
///
/// // Navigate through [100, 200, 300]
/// if let Some(z1) = zipper.descend(100) {
///     if let Some(z2) = z1.descend(200) {
///         if let Some(z3) = z2.descend(300) {
///             assert!(z3.is_final());
///         }
///     }
/// }
/// ```
#[derive(Clone)]
pub struct DynamicDawgU64Zipper<V: DictionaryValue = ()> {
    /// Arc reference to the current node
    node: Arc<DawgNodeU64<V>>,

    /// Path from root to current position
    path: Vec<u64>,
}

impl<V: DictionaryValue> DynamicDawgU64Zipper<V> {
    /// Create a new zipper at the root of the Dynamic DAWG.
    ///
    /// # Arguments
    ///
    /// * `dict` - Reference to the DynamicDawgU64 dictionary
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use liblevenshtein::dictionary::dynamic_dawg_u64::DynamicDawgU64;
    /// use liblevenshtein::dictionary::dynamic_dawg_u64_zipper::DynamicDawgU64Zipper;
    ///
    /// let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
    /// let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);
    /// ```
    pub fn new_from_dict(dict: &DynamicDawgU64<V>) -> Self {
        DynamicDawgU64Zipper {
            node: dict.root_arc(),
            path: Vec::new(),
        }
    }

    /// Create a zipper from a node reference and path.
    ///
    /// This is primarily for internal use by the DAWG implementation.
    pub(crate) fn from_node(node: Arc<DawgNodeU64<V>>, path: Vec<u64>) -> Self {
        DynamicDawgU64Zipper { node, path }
    }

    /// Get a reference to the current node.
    ///
    /// Useful for debugging or advanced use cases.
    pub fn node_ref(&self) -> &Arc<DawgNodeU64<V>> {
        &self.node
    }
}

impl<V: DictionaryValue> DictZipper for DynamicDawgU64Zipper<V> {
    type Unit = u64;

    fn is_final(&self) -> bool {
        self.node.is_final.load(Ordering::Acquire)
    }

    fn descend(&self, label: Self::Unit) -> Option<Self> {
        // Load the current edge list (wait-free)
        let edges = self.node.edges.load();

        // Find the edge with the given label
        edges.find(label).map(|child| {
            let mut new_path = self.path.clone();
            new_path.push(label);
            DynamicDawgU64Zipper {
                node: child.clone(),
                path: new_path,
            }
        })
    }

    fn path(&self) -> Vec<Self::Unit> {
        self.path.clone()
    }

    fn children(&self) -> impl Iterator<Item = (Self::Unit, Self)> {
        // Load the edge list once (wait-free snapshot)
        let edges = self.node.edges.load();

        // Clone the edges to allow independent iteration
        let edge_vec: Vec<_> = edges.edges.iter().cloned().collect();
        let base_path = self.path.clone();

        edge_vec.into_iter().map(move |(label, child)| {
            let mut new_path = base_path.clone();
            new_path.push(label);
            (
                label,
                DynamicDawgU64Zipper {
                    node: child,
                    path: new_path,
                },
            )
        })
    }
}

impl<V: DictionaryValue> ValuedDictZipper for DynamicDawgU64Zipper<V> {
    type Value = V;

    fn value(&self) -> Option<Self::Value> {
        if self.node.is_final.load(Ordering::Acquire) {
            // Load the value (wait-free)
            // value_guard is Guard<Option<Arc<V>>>; *value_guard is Option<Arc<V>>
            let value_guard = self.node.value.load();
            // Map to clone the inner V
            value_guard.as_ref().map(|arc| (**arc).clone())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_zipper_not_final() {
        let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
        dict.insert_sequence(&[1, 2, 3, 4]);

        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);
        assert!(!zipper.is_final());
    }

    #[test]
    fn test_descend_nonexistent() {
        let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
        dict.insert_sequence(&[1, 2, 3, 4]);

        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);
        assert!(zipper.descend(999).is_none());
    }

    #[test]
    fn test_descend_and_finality() {
        let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
        dict.insert_sequence(&[10, 20, 30]); // [10, 20, 30]
        dict.insert_sequence(&[10, 20, 30, 40, 50]); // [10, 20, 30, 40, 50]

        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);

        // Navigate to [10, 20, 30]
        let n1 = zipper.descend(10).expect("Should descend to 10");
        assert!(!n1.is_final());

        let n2 = n1.descend(20).expect("Should descend to 20");
        assert!(!n2.is_final());

        let n3 = n2.descend(30).expect("Should descend to 30");
        assert!(n3.is_final(), "[10, 20, 30] should be a final state");

        // Continue to [10, 20, 30, 40, 50]
        let n4 = n3.descend(40).expect("Should descend to 40");
        let n5 = n4.descend(50).expect("Should descend to 50");
        assert!(n5.is_final(), "[10, 20, 30, 40, 50] should be a final state");
    }

    #[test]
    fn test_f64_navigation() {
        let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
        dict.insert_f64(&[1.0, 2.0, 3.0]);
        dict.insert_f64(&[1.0, 2.0, 4.0]);

        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);

        // Navigate using f64 bit patterns
        let v1 = 1.0f64.to_bits();
        let v2 = 2.0f64.to_bits();
        let v3 = 3.0f64.to_bits();

        let n1 = zipper.descend(v1).expect("Should descend to 1.0");
        let n2 = n1.descend(v2).expect("Should descend to 2.0");
        let n3 = n2.descend(v3).expect("Should descend to 3.0");
        assert!(n3.is_final());
    }

    #[test]
    fn test_children_iteration() {
        let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
        dict.insert_sequence(&[100, 200, 300]);
        dict.insert_sequence(&[100, 200, 400]);
        dict.insert_sequence(&[500, 600, 700]);

        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);

        // Root should have children 100 and 500
        let children: Vec<u64> = zipper.children().map(|(label, _)| label).collect();
        assert!(children.contains(&100));
        assert!(children.contains(&500));
    }

    #[test]
    fn test_valued_zipper() {
        let dict: DynamicDawgU64<u32> = DynamicDawgU64::new();
        dict.insert_sequence_with_value(&[10, 20, 30], 1);
        dict.insert_sequence_with_value(&[10, 20, 30, 40, 50], 2);

        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);

        // Navigate to [10, 20, 30]
        let n3 = zipper
            .descend(10)
            .and_then(|z| z.descend(20))
            .and_then(|z| z.descend(30))
            .expect("Should navigate to [10, 20, 30]");

        assert_eq!(n3.value(), Some(1));

        // Navigate to [10, 20, 30, 40, 50]
        let n5 = n3
            .descend(40)
            .and_then(|z| z.descend(50))
            .expect("Should navigate to [10, 20, 30, 40, 50]");

        assert_eq!(n5.value(), Some(2));
    }

    #[test]
    fn test_f64_valued_zipper() {
        let dict: DynamicDawgU64<String> = DynamicDawgU64::new();
        dict.insert_f64_with_value(&[1.0, 2.0, 3.0], "first".to_string());
        dict.insert_f64_with_value(&[1.0, 2.0, 4.0], "second".to_string());

        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);

        // Navigate to [1.0, 2.0, 3.0]
        let v1 = 1.0f64.to_bits();
        let v2 = 2.0f64.to_bits();
        let v3 = 3.0f64.to_bits();

        let n3 = zipper
            .descend(v1)
            .and_then(|z| z.descend(v2))
            .and_then(|z| z.descend(v3))
            .expect("Should navigate to [1.0, 2.0, 3.0]");

        assert_eq!(n3.value(), Some("first".to_string()));
    }

    #[test]
    fn test_clone_independence() {
        let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
        dict.insert_sequence(&[1, 2, 3]);

        let zipper1 = DynamicDawgU64Zipper::new_from_dict(&dict);
        let zipper2 = zipper1.clone();

        // Both zippers should navigate independently
        let z1_n = zipper1.descend(1);
        let z2_n = zipper2.descend(1);

        assert!(z1_n.is_some());
        assert!(z2_n.is_some());
    }

    #[test]
    fn test_empty_dictionary() {
        let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);

        assert!(!zipper.is_final());
        assert_eq!(zipper.children().count(), 0);
    }

    #[test]
    fn test_value_none_for_non_final() {
        let dict: DynamicDawgU64<u32> = DynamicDawgU64::new();
        dict.insert_sequence_with_value(&[10, 20, 30], 42);

        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);

        // Navigate to 10 (non-final)
        let n1 = zipper.descend(10).expect("Should descend to 10");
        assert_eq!(
            n1.value(),
            None,
            "Non-final node should have no value"
        );
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let dict = StdArc::new({
            let d: DynamicDawgU64<()> = DynamicDawgU64::new();
            d.insert_sequence(&[1, 2, 3]);
            d.insert_sequence(&[1, 2, 3, 4, 5]);
            d
        });

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let dict_clone = dict.clone();
                thread::spawn(move || {
                    let zipper = DynamicDawgU64Zipper::new_from_dict(&dict_clone);
                    zipper.descend(1).is_some()
                })
            })
            .collect();

        for handle in handles {
            assert!(handle.join().unwrap());
        }
    }

    #[test]
    fn test_path_tracking() {
        let dict: DynamicDawgU64<()> = DynamicDawgU64::new();
        dict.insert_sequence(&[100, 200, 300, 400]);

        let zipper = DynamicDawgU64Zipper::new_from_dict(&dict);

        // Navigate through [100, 200, 300, 400]
        let n1 = zipper.descend(100).unwrap();
        assert_eq!(n1.path(), vec![100]);

        let n2 = n1.descend(200).unwrap();
        assert_eq!(n2.path(), vec![100, 200]);

        let n3 = n2.descend(300).unwrap();
        assert_eq!(n3.path(), vec![100, 200, 300]);

        let n4 = n3.descend(400).unwrap();
        assert_eq!(n4.path(), vec![100, 200, 300, 400]);
    }

    #[test]
    fn test_f64_edge_cases() {
        let dict: DynamicDawgU64<()> = DynamicDawgU64::new();

        // Test special float values
        dict.insert_f64(&[0.0, f64::INFINITY, f64::NEG_INFINITY]);
        dict.insert_f64(&[-0.0]); // Different bit pattern from +0.0
        dict.insert_f64(&[f64::NAN]);

        assert!(dict.contains_f64(&[0.0, f64::INFINITY, f64::NEG_INFINITY]));
        assert!(dict.contains_f64(&[-0.0]));
        // NaN requires bit-pattern comparison
        let nan_bits = f64::NAN.to_bits();
        assert!(dict.contains_sequence(&[nan_bits]));
    }

    #[test]
    fn test_wait_free_reads_during_writes() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let dict = StdArc::new(DynamicDawgU64::<()>::new());
        let stop = StdArc::new(AtomicBool::new(false));

        // Pre-populate some data
        for i in 0..100 {
            dict.insert_sequence(&[i, i + 1, i + 2]);
        }

        let dict_clone = dict.clone();
        let stop_clone = stop.clone();

        // Reader thread - should never block
        let reader = thread::spawn(move || {
            let mut reads = 0u64;
            while !stop_clone.load(Ordering::Relaxed) {
                let zipper = DynamicDawgU64Zipper::new_from_dict(&dict_clone);
                for (_, child) in zipper.children() {
                    let _ = child.is_final();
                }
                reads += 1;
            }
            reads
        });

        // Writer thread - performs concurrent modifications
        let writer = thread::spawn(move || {
            for i in 100..200 {
                dict.insert_sequence(&[i, i + 1, i + 2]);
            }
        });

        writer.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        let read_count = reader.join().unwrap();

        // Verify reader thread was not starved (made progress)
        assert!(read_count > 0, "Reader should have completed at least one iteration");
    }
}
