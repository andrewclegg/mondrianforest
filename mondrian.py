import random
import os
import numpy as np

from collections import deque
from numba import jit, njit


def scorer(name):
    """
    Initialize a scorer of the specified type (currently 'nsp' or 'simple') and return it.
    """
    if name == 'simple':
        scorer = SimpleScorer()
    elif name == 'nsp':
        scorer = NSPScorer()
    else:
        raise ValueError('Unknown scorer ' + name)
    return scorer


def init_tree(n_dims, n_labels, budget, scoring, lib_path, global_name):
    """
    Helper function for remote engines.
    """
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader('mondrian', lib_path)
    mondrian = loader.load_module()
    globals()[global_name] = mondrian.MondrianTree(n_dims, n_labels, budget, scoring)


def extend(data, labels, global_name):
    """
    Helper function for remote engines.
    """
    globals()[global_name].extend(data, labels)


def predict(row, global_name):
    """
    Helper function for remote engines.
    """
    tree = globals()[global_name]
    pred = tree.predict(row)
    return pred


def combine_predictions(results, aggregate=True):
    stacked = np.vstack(results)
    if aggregate:
        return stacked.sum(axis=0) / stacked.sum()
    else:
        return stacked


@jit('f8[:](f8[:,:])')
def get_colwise_min(data):
    return np.amin(data, axis=0)


@jit('f8[:](f8[:,:])')
def get_colwise_max(data):
    return np.amax(data, axis=0)


@jit('f8[:](f8[:], f8[:])')
def get_data_range(min_d, max_d):
    return max_d - min_d


@jit('f8(f8[:])')
def sample_multinomial_scores(scores):
    scores_cumsum = np.cumsum(scores)
    s = scores_cumsum[-1] * np.random.rand(1)
    k = int(np.sum(s > scores_cumsum))
    return k


@jit('f8[:](f8[:])')
def normalize(vec):
    return vec / vec.sum()


class MondrianNode(object):

    
    def __init__(self, n_dims, n_labels, parent, budget, scorer):
        self.min_d = np.empty(n_dims)
        self.max_d = np.empty(n_dims)
        self.range_d = np.empty(n_dims)
        self.sum_range_d = 0
        self.label_counts = np.zeros(n_labels)
        self.budget = budget
        self.max_split_cost = 0
        self.parent = parent
        self.left = None
        self.right = None
        self.split_dim = None
        self.split_point = None
        self._scorer = scorer
        self._scorer.on_create(self)

    
    def update(self, data, labels):
        # Update bounding box and label counts
        self.min_d = get_colwise_min(data)
        self.max_d = get_colwise_max(data)
        self.range_d = get_data_range(self.min_d, self.max_d)
        self.sum_range_d = np.sum(self.range_d)
        self.label_counts += np.bincount(labels, minlength=len(self.label_counts))
        self._scorer.on_update(self, labels)
        

    def apply_split(self, data):
        # Apply this node's existing splitting criterion to some data
        # and return a boolean index (True==goes left, False==goes right)
        return data[:, self.split_dim] <= self.split_point
    
    
    def is_leaf(self):
        if self.left is None:
            assert self.right is None
            return True
        else:
            assert self.right is not None
            return False

        
    def is_pure(self):
        return np.count_nonzero(self.label_counts) < 2


class SimpleScorer(object):


    def on_create(self, node):
        # TODO can we do some clever caching so we do less calculation during prediction?
        if not hasattr(self, '_root'):
            if node.parent is not None:
                raise ValueError('on_create must be called with root node before any others')
            self._root = node


    def on_update(self, node, labels):
        # TODO can we do some clever caching so we do less calculation during prediction?
        pass


    def predict(self, row):

        node = self._root
        last_counts_seen = node.label_counts
        while node.label_counts.sum() > 0 and not node.is_leaf():
            last_counts_seen = node.label_counts
            left = node.apply_split(row)
            node = node.left if left else node.right

        # FIXME why do we sometimes get tiny floats in here, when they should be just counts (ints)?
        if np.allclose(last_counts_seen, 0):
            return np.zeros_like(last_counts_seen)

        # L1 normalize
        return normalize(last_counts_seen)


class NSPScorer(object):


    def on_create(self, leaf_node):

        if not leaf_node.is_leaf():
            raise ValueError('on_create called for non-leaf node')

        if not hasattr(self, '_n_labels'):
            self._tables = {}
            self._pseudocounts = {}
            self._posteriors = {}
            self._n_labels = leaf_node.n_labels
            self._prior = np.ones(self._n_labels) / self._n_labels # Uniform prior

        self._tables[leaf_node] = np.zeros(self._n_labels)
        self._pseudocounts[leaf_node] = np.zeros(self._n_labels)


    def on_update(self, leaf_node, labels):

        if not leaf_node.is_leaf():
            raise ValueError('on_update called for non-leaf node')

        node = leaf_node
        pc = self._pseudocounts
        t = self._tables

        # First, update all the class pseudocounts, from this node up
        # TODO optimize this
        for label in labels:
            pc[node][label] += 1

            while True:
                if t[node][label] == 1:
                    break
                
                if not is_leaf(node):
                    pc[node][label] = t[node.left][label] + t[node.right][label]

                t[node][label] = min(pc[node][label], 1)

                if node.parent is None:
                    root = node
                    break
                node = node.parent

        # Then, update the posterior probabilities, from the root down
        todo = deque([root])
        while todo:
            next_node = todo.pop()
            if next_node == root:
                prior = self._prior
            else:
                prior = self._posteriors[next_node.parent]
            d = self._discounts[node]
            # TODO this doesn't take into account the *real* counts where we've left data in internal nodes
            self._posteriors[node] = (pc[node] - (d * t[node]) + (d * t[node].sum() * prior)) / pc[node].sum()
            if node.left:
                todo.append(node.left)
            if node.right:
                todo.append(node.right)


    def predict(self, row):
        pass # TODO


# TODO add _discounts


class MondrianTree(object):
    
    
    def __init__(self, n_dims, n_labels, budget, scoring):
        self.n_dims = n_dims
        self.n_labels = n_labels
        self.starting_budget = budget
        self._scorer = scorer(scoring)
        self.root = MondrianNode(n_dims, n_labels, None, budget, self._scorer)
        
    
    def extend(self, data, labels):
        self._extend_node(self.root, data, labels)


    def _extend_node(self, node, data, labels):
        
        min_d = get_colwise_min(data)
        max_d = get_colwise_max(data)
        additional_extent_lower = np.fmax(0, node.min_d - min_d)
        additional_extent_upper = np.fmax(0, max_d - node.max_d)
        expo_parameter = np.sum([additional_extent_lower, additional_extent_upper])
        
        is_leaf = node.is_leaf()
        if expo_parameter == 0 or is_leaf:
            # Don't split if (a) none of the new data is outside the existing bounding box,
            # or (b) it's a leaf node (we don't split these, we grow them)
            split_cost = np.inf
        else:
            # The bigger the new bounding box relative to the old one, the more likely we are to split the node
            split_cost = random.expovariate(expo_parameter)
        
        if split_cost < node.max_split_cost and not is_leaf:
            # Stop what we're doing and instigate a node split instead
            self._split_node(node, data, labels, split_cost, min_d, max_d,
                             additional_extent_lower, additional_extent_upper)
            return

        # Otherwise carry on updating the existing tree structure
        
        was_paused = is_leaf and node.is_pure()
        node.update(data, labels)
        is_paused = is_leaf and node.is_pure()

        if was_paused and not is_paused:
            # We've unpaused a leaf node so we need to grow the tree
            self._grow(node)
        elif not is_leaf:
            # Split the data into left portion and right portion, and repeat
            # whole process with node's children
            goes_left = node.apply_split(data)
            if np.any(goes_left):
                l_data = np.compress(goes_left, data, axis=0)
                l_labels = np.compress(goes_left, labels)
                self._extend_node(node.left, l_data, l_labels)
            goes_right = ~goes_left
            if np.any(goes_right):
                r_data = np.compress(goes_right, data, axis=0)
                r_labels = np.compress(goes_right, labels)
                self._extend_node(node.right, r_data, r_labels)


    def _split_node(self, node, data, labels, split_cost, min_d, max_d,
                    additional_extent_lower, additional_extent_upper):
        
        assert not node.is_leaf() # Mutate leaf nodes by growing, not splitting
        
        # Create new parent node which is a near-copy of the one that's splitting
        new_parent = MondrianNode(self.n_dims, self.n_labels, node.parent, node.budget, self._scorer)
        new_parent.min_d = np.fmin(min_d, node.min_d)
        new_parent.max_d = np.fmax(max_d, node.max_d)
        new_parent.range_d = get_data_range(new_parent.min_d, new_parent.max_d)
        new_parent.sum_range_d = np.sum(new_parent.range_d)
        new_parent.label_counts = node.label_counts
        
        # Pick a random dimension to split on
        feat_score = additional_extent_lower + additional_extent_upper
        feat_id = sample_multinomial_scores(feat_score)
        
        # Pick a random split point between previous bounding box and new one
        draw_from_lower = np.random.rand() <= (additional_extent_lower[feat_id] / feat_score[feat_id])
        if draw_from_lower:
            split = random.uniform(min_d[feat_id], node.min_d[feat_id])
        else:
            split = random.uniform(node.max_d[feat_id], max_d[feat_id])
        
        assert (split < node.min_d[feat_id]) or (split > node.max_d[feat_id])
        
        # Set up the new parent node to use this split
        new_parent.split_dim = feat_id
        new_parent.split_point = split
        
        # Now create new child node which is initially empty -- new sibling for original node
        new_budget = node.budget - split_cost
        new_sibling = MondrianNode(self.n_dims, self.n_labels, new_parent, new_budget, self._scorer)
        
        # This bit's clever -- since the split point is outside the original node's bounding box,
        # we might need to give some new data to that node, but we won't need to take any away
        # from the existing node. By definition, none of its data points can be in there.
        
        # Use the new parent's split criterion to decide which child gets which subset of data,
        # and adjust tree to match
        goes_left = new_parent.apply_split(data)
        original_node_goes_left = split > node.max_d[feat_id]
        if original_node_goes_left:
            new_parent.left = node
            new_parent.right = new_sibling
            data_for_original = goes_left
        else:
            new_parent.left = new_sibling
            new_parent.right = node
            data_for_original = ~goes_left

        # Figure out whether new parent is on the left or right of *its* parent (unless it's the root)
        if node == self.root:
            self.root = new_parent
        else:
            if node.parent.left == node:
                node.parent.left = new_parent
            elif node.parent.right == node:
                node.parent.right = new_parent
            else:
                assert False
        node.parent = new_parent
        
        # Update budgets and costs associated with the nodes
        node.budget = new_budget
        node.max_split_cost = node.max_split_cost - split_cost
        new_parent.max_split_cost = split_cost
        
        # Update the bounding boxes and label counts for the left and right sides of the new split
        # (the new node will be grown automatically when needed, filling in max_split_cost)
        if np.any(data_for_original):
            l_data = np.compress(data_for_original, data, axis=0)
            l_labels = np.compress(data_for_original, labels)
            self._extend_node(node.left, l_data, l_labels)
        data_for_new = ~data_for_original
        if np.any(data_for_new):
            r_data = np.compress(data_for_new, data, axis=0)
            r_labels = np.compress(data_for_new, labels)
            self._extend_node(node.right, r_data, r_labels)
        
        # TODO ensure the new node behaves correctly (i.e. gets grown when it needs to)


    def _grow(self, node):
        
        assert node.is_leaf()
        
        # Is node paused, empty or effectively empty? If so, don't split it
        if node.is_pure() or node.sum_range_d == 0:
            node.max_split_cost = node.budget
            return

        split_cost = random.expovariate(node.sum_range_d)
        node.max_split_cost = split_cost

        if node.budget > split_cost:
            node.split_dim = sample_multinomial_scores(node.range_d)
            node.split_point = random.uniform(node.min_d[node.split_dim], node.max_d[node.split_dim])
            
            # TODO use the default budget here, or inherit from parent? Paper says default
            node.left = MondrianNode(self.n_dims, self.n_labels, node, self.starting_budget, self._scorer)
            node.right = MondrianNode(self.n_dims, self.n_labels, node, self.starting_budget, self._scorer)
            # No point growing these as they start off empty (unlike in original paper)


    def predict(self, row):
        return self._scorer.predict(row)


class MondrianForest(object):

    
    def __init__(self, n_trees, n_dims, n_labels, budget, scoring):
        self.trees = [MondrianTree(n_dims, n_labels, budget, scoring) for k in range(n_trees)]
    

    def update(self, data, labels):
        for tree in self.trees:
            tree.extend(data, labels)


    def predict(self, row, aggregate):
        results = [tree.predict(row) for tree in self.trees]
        return combine_predictions(results, aggregate)


class ParallelMondrianForest(object):


    def __init__(self, ipy_view, n_dims, n_labels, budget, scoring):
        self._view = ipy_view
        self._remote_name = 'mondrian_worker'
        self._view.apply_sync(init_tree, n_dims, n_labels, budget, scoring, os.path.realpath(__file__), self._remote_name)


    def update(self, data, labels):
        self._view.apply_sync(extend, data, labels, self._remote_name)


    def predict(self, row, aggregate):
        results = self._view.apply_sync(predict, row, self._remote_name)
        return combine_predictions(results, aggregate)

