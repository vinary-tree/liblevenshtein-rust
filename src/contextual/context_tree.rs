//! Hierarchical context tree for scope-based draft visibility.
//!
//! This module provides a tree structure for managing hierarchical lexical scopes
//! where child contexts can see drafts from parent contexts (like nested scopes
//! in programming languages).

use super::error::{ContextError, Result};
use std::collections::HashMap;

/// Unique identifier for a context in the tree.
pub type ContextId = u32;

/// Hierarchical tree of contexts with parent-child relationships.
///
/// Models lexical scoping where child contexts (inner scopes) can see
/// drafts from parent contexts (outer scopes), but not vice versa.
///
/// # Memory Efficiency
///
/// - Per node: ~16 bytes (id + parent Option)
/// - HashMap overhead: ~48 bytes + entries
/// - Typical tree (50 contexts): ~1KB
///
/// # Use Cases
///
/// - **Code editor**: Function scope sees module scope sees global scope
/// - **REPL**: Each expression creates a child context
/// - **Nested blocks**: Inner blocks see outer block definitions
///
/// # Examples
///
/// ```
/// use liblevenshtein::contextual::ContextTree;
///
/// let mut tree = ContextTree::new();
///
/// // Create global scope
/// let global = tree.create_root(1);
///
/// // Create module scope (child of global)
/// let module = tree.create_child(2, global).expect("doc/test fixture: create_child with existing parent");
///
/// // Create function scope (child of module)
/// let function = tree.create_child(3, module).expect("doc/test fixture: create_child with existing parent");
///
/// // Function can see: function, module, global
/// let visible = tree.visible_contexts(function);
/// assert_eq!(visible, vec![function, module, global]);
/// ```
#[derive(Debug, Clone)]
pub struct ContextTree {
    /// Map from context ID to parent ID (None for root)
    nodes: HashMap<ContextId, Option<ContextId>>,
}

impl ContextTree {
    /// Create a new empty context tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let tree = ContextTree::new();
    /// assert!(tree.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Create a new context tree with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Expected number of contexts
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let tree = ContextTree::with_capacity(100);
    /// assert!(tree.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: HashMap::with_capacity(capacity),
        }
    }

    /// Create a root context (no parent).
    ///
    /// # Arguments
    ///
    /// * `id` - Unique context ID
    ///
    /// # Returns
    ///
    /// The context ID (same as input)
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// let root = tree.create_root(1);
    /// assert!(tree.is_root(root));
    /// ```
    pub fn create_root(&mut self, id: ContextId) -> ContextId {
        self.nodes.insert(id, None);
        id
    }

    /// Create a child context.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique context ID for the child
    /// * `parent_id` - Parent context ID
    ///
    /// # Returns
    ///
    /// `Ok(id)` if successful, `Err` if parent doesn't exist
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// let root = tree.create_root(1);
    /// let child = tree.create_child(2, root).expect("doc/test fixture: create_child with existing parent");
    ///
    /// assert_eq!(tree.parent(child), Some(root));
    /// ```
    pub fn create_child(&mut self, id: ContextId, parent_id: ContextId) -> Result<ContextId> {
        if !self.nodes.contains_key(&parent_id) {
            return Err(ContextError::ContextNotFound(parent_id));
        }
        self.nodes.insert(id, Some(parent_id));
        Ok(id)
    }

    /// Get the parent of a context.
    ///
    /// # Arguments
    ///
    /// * `id` - Context ID
    ///
    /// # Returns
    ///
    /// `Some(parent_id)` if context has a parent, `None` if root or doesn't exist
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// let root = tree.create_root(1);
    /// let child = tree.create_child(2, root).expect("doc/test fixture: create_child with existing parent");
    ///
    /// assert_eq!(tree.parent(root), None);
    /// assert_eq!(tree.parent(child), Some(root));
    /// ```
    pub fn parent(&self, id: ContextId) -> Option<ContextId> {
        self.nodes.get(&id).and_then(|parent| *parent)
    }

    /// Check if a context is a root (has no parent).
    ///
    /// # Arguments
    ///
    /// * `id` - Context ID
    ///
    /// # Returns
    ///
    /// `true` if context is a root, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// let root = tree.create_root(1);
    ///
    /// assert!(tree.is_root(root));
    /// ```
    pub fn is_root(&self, id: ContextId) -> bool {
        matches!(self.nodes.get(&id), Some(None))
    }

    /// Check if a context exists in the tree.
    ///
    /// # Arguments
    ///
    /// * `id` - Context ID
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// let root = tree.create_root(1);
    ///
    /// assert!(tree.contains(root));
    /// assert!(!tree.contains(999));
    /// ```
    pub fn contains(&self, id: ContextId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Get all contexts visible from a given context (self + ancestors).
    ///
    /// Returns contexts in order: self, parent, grandparent, ..., root
    ///
    /// # Arguments
    ///
    /// * `id` - Context ID
    ///
    /// # Returns
    ///
    /// Vector of visible context IDs (empty if context doesn't exist)
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// let global = tree.create_root(1);
    /// let module = tree.create_child(2, global).expect("doc/test fixture: create_child with existing parent");
    /// let function = tree.create_child(3, module).expect("doc/test fixture: create_child with existing parent");
    ///
    /// // Function can see: itself, module, global
    /// let visible = tree.visible_contexts(function);
    /// assert_eq!(visible, vec![function, module, global]);
    /// ```
    pub fn visible_contexts(&self, id: ContextId) -> Vec<ContextId> {
        let mut result = Vec::new();
        let mut current = Some(id);

        while let Some(ctx_id) = current {
            if self.nodes.contains_key(&ctx_id) {
                result.push(ctx_id);
                current = self.parent(ctx_id);
            } else {
                break;
            }
        }

        result
    }

    /// Remove a context and all its descendants.
    ///
    /// # Arguments
    ///
    /// * `id` - Context ID to remove
    ///
    /// # Returns
    ///
    /// `true` if context was removed, `false` if it didn't exist
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// let root = tree.create_root(1);
    /// let child = tree.create_child(2, root).expect("doc/test fixture: create_child with existing parent");
    ///
    /// tree.remove(root);
    /// assert!(!tree.contains(root));
    /// assert!(!tree.contains(child)); // Descendants also removed
    /// ```
    pub fn remove(&mut self, id: ContextId) -> bool {
        if !self.nodes.contains_key(&id) {
            return false;
        }

        // Find all descendants
        let descendants: Vec<ContextId> = self
            .nodes
            .iter()
            .filter_map(|(child_id, _parent)| {
                if self.is_descendant(*child_id, id) {
                    Some(*child_id)
                } else {
                    None
                }
            })
            .collect();

        // Remove the context and all descendants
        self.nodes.remove(&id);
        for desc_id in descendants {
            self.nodes.remove(&desc_id);
        }

        true
    }

    /// Check if a context is a descendant of another.
    ///
    /// # Arguments
    ///
    /// * `child_id` - Potential descendant
    /// * `ancestor_id` - Potential ancestor
    ///
    /// # Returns
    ///
    /// `true` if `child_id` is a descendant of `ancestor_id`
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// let root = tree.create_root(1);
    /// let child = tree.create_child(2, root).expect("doc/test fixture: create_child with existing parent");
    /// let grandchild = tree.create_child(3, child).expect("doc/test fixture: create_child with existing parent");
    ///
    /// assert!(tree.is_descendant(grandchild, root));
    /// assert!(tree.is_descendant(child, root));
    /// assert!(!tree.is_descendant(root, child));
    /// ```
    pub fn is_descendant(&self, child_id: ContextId, ancestor_id: ContextId) -> bool {
        if child_id == ancestor_id {
            return false;
        }

        let mut current = self.parent(child_id);
        while let Some(parent_id) = current {
            if parent_id == ancestor_id {
                return true;
            }
            current = self.parent(parent_id);
        }

        false
    }

    /// Get the depth of a context (distance from root).
    ///
    /// Root contexts have depth 0, their children have depth 1, etc.
    ///
    /// # Arguments
    ///
    /// * `id` - Context ID
    ///
    /// # Returns
    ///
    /// Depth if context exists, `None` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// let root = tree.create_root(1);
    /// let child = tree.create_child(2, root).expect("doc/test fixture: create_child with existing parent");
    /// let grandchild = tree.create_child(3, child).expect("doc/test fixture: create_child with existing parent");
    ///
    /// assert_eq!(tree.depth(root), Some(0));
    /// assert_eq!(tree.depth(child), Some(1));
    /// assert_eq!(tree.depth(grandchild), Some(2));
    /// ```
    pub fn depth(&self, id: ContextId) -> Option<usize> {
        if !self.nodes.contains_key(&id) {
            return None;
        }

        let mut depth = 0;
        let mut current = self.parent(id);

        while current.is_some() {
            depth += 1;
            current = current.and_then(|pid| self.parent(pid));
        }

        Some(depth)
    }

    /// Get the number of contexts in the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// assert_eq!(tree.len(), 0);
    ///
    /// tree.create_root(1);
    /// assert_eq!(tree.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tree is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let tree = ContextTree::new();
    /// assert!(tree.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear all contexts from the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use liblevenshtein::contextual::ContextTree;
    ///
    /// let mut tree = ContextTree::new();
    /// tree.create_root(1);
    /// tree.clear();
    /// assert!(tree.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.nodes.clear();
    }
}

impl Default for ContextTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let tree = ContextTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_create_root() {
        let mut tree = ContextTree::new();
        let root = tree.create_root(1);
        assert_eq!(root, 1);
        assert!(tree.contains(root));
        assert!(tree.is_root(root));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_create_child() {
        let mut tree = ContextTree::new();
        let root = tree.create_root(1);
        let child = tree
            .create_child(2, root)
            .expect("doc/test fixture: create_child with existing parent");

        assert_eq!(child, 2);
        assert!(tree.contains(child));
        assert!(!tree.is_root(child));
        assert_eq!(tree.parent(child), Some(root));
    }

    #[test]
    fn test_create_child_invalid_parent() {
        let mut tree = ContextTree::new();
        let result = tree.create_child(2, 999);
        assert!(result.is_err());
    }

    #[test]
    fn test_visible_contexts() {
        let mut tree = ContextTree::new();
        let global = tree.create_root(1);
        let module = tree
            .create_child(2, global)
            .expect("doc/test fixture: create_child with existing parent");
        let function = tree
            .create_child(3, module)
            .expect("doc/test fixture: create_child with existing parent");

        let visible = tree.visible_contexts(function);
        assert_eq!(visible, vec![function, module, global]);

        let visible_module = tree.visible_contexts(module);
        assert_eq!(visible_module, vec![module, global]);

        let visible_global = tree.visible_contexts(global);
        assert_eq!(visible_global, vec![global]);
    }

    #[test]
    fn test_is_descendant() {
        let mut tree = ContextTree::new();
        let root = tree.create_root(1);
        let child = tree
            .create_child(2, root)
            .expect("doc/test fixture: create_child with existing parent");
        let grandchild = tree
            .create_child(3, child)
            .expect("doc/test fixture: create_child with existing parent");

        assert!(tree.is_descendant(grandchild, root));
        assert!(tree.is_descendant(grandchild, child));
        assert!(tree.is_descendant(child, root));
        assert!(!tree.is_descendant(root, child));
        assert!(!tree.is_descendant(child, grandchild));
        assert!(!tree.is_descendant(root, root)); // Not descendant of self
    }

    #[test]
    fn test_depth() {
        let mut tree = ContextTree::new();
        let root = tree.create_root(1);
        let child = tree
            .create_child(2, root)
            .expect("doc/test fixture: create_child with existing parent");
        let grandchild = tree
            .create_child(3, child)
            .expect("doc/test fixture: create_child with existing parent");

        assert_eq!(tree.depth(root), Some(0));
        assert_eq!(tree.depth(child), Some(1));
        assert_eq!(tree.depth(grandchild), Some(2));
        assert_eq!(tree.depth(999), None);
    }

    #[test]
    fn test_remove() {
        let mut tree = ContextTree::new();
        let root = tree.create_root(1);
        let child = tree
            .create_child(2, root)
            .expect("doc/test fixture: create_child with existing parent");
        let grandchild = tree
            .create_child(3, child)
            .expect("doc/test fixture: create_child with existing parent");

        // Remove child also removes grandchild
        assert!(tree.remove(child));
        assert!(!tree.contains(child));
        assert!(!tree.contains(grandchild));
        assert!(tree.contains(root));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut tree = ContextTree::new();
        assert!(!tree.remove(999));
    }

    #[test]
    fn test_clear() {
        let mut tree = ContextTree::new();
        tree.create_root(1);
        tree.create_root(2);
        tree.clear();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_multiple_roots() {
        let mut tree = ContextTree::new();
        let root1 = tree.create_root(1);
        let root2 = tree.create_root(2);

        assert!(tree.is_root(root1));
        assert!(tree.is_root(root2));
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_complex_hierarchy() {
        let mut tree = ContextTree::new();

        // Build tree:
        //     1 (global)
        //    / \
        //   2   3 (modules)
        //  /     \
        // 4       5 (functions)

        let global = tree.create_root(1);
        let mod1 = tree
            .create_child(2, global)
            .expect("doc/test fixture: create_child with existing parent");
        let mod2 = tree
            .create_child(3, global)
            .expect("doc/test fixture: create_child with existing parent");
        let func1 = tree
            .create_child(4, mod1)
            .expect("doc/test fixture: create_child with existing parent");
        let func2 = tree
            .create_child(5, mod2)
            .expect("doc/test fixture: create_child with existing parent");

        assert_eq!(tree.visible_contexts(func1), vec![func1, mod1, global]);
        assert_eq!(tree.visible_contexts(func2), vec![func2, mod2, global]);
        assert!(!tree.is_descendant(func1, mod2));
        assert!(!tree.is_descendant(func2, mod1));
    }
}
