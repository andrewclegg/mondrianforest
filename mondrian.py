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


def predict(x, global_name):
    """
    Helper function for remote engines.
    """
    tree = globals()[global_name]
    pred = tree.predict(x)
    return pred


def combine_predictions(results):
    """
    Helper function for combining predictions from multiple trees.

    The results param is a list of numpy arrays, one from each tree.
    Each one must have one row per instance under test, and one column per class.
    Each cell is the probability of that instance belonging to that class.

    The output has the same shape as one of these input arrays. The class probabilities
    for each instance are summed across all the input arrays, and renormalized.
    """
    summed = np.add.reduce(results)
    return normalize(summed)


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
def normalize(array):
    if array.ndim == 1:
        return array / array.sum()
    else:
        row_totals = array.sum(axis=1)[:, np.newaxis]
        return array / row_totals


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
        # Apply this node's existing splitting criterion to some data (vector or array)
        # and return a boolean index (True==goes left, False==goes right)
        data_array = np.atleast_2d(data)
        return data_array[:, self.split_dim] <= self.split_point
    
    
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


    def _predict(self, vector):

        node = self._root
        last_counts_seen = node.label_counts
        while node.label_counts.sum() > 0 and not node.is_leaf():
            last_counts_seen = node.label_counts
            left = node.apply_split(vector)
            node = node.left if left else node.right

        # FIXME why do we sometimes get tiny floats in here, when they should be just counts (ints)?
        if np.allclose(last_counts_seen, 0):
            return np.zeros_like(last_counts_seen)
        else:
            # L1 normalize
            return normalize(last_counts_seen)


    def predict(self, x):

        x_array = np.atleast_2d(x)
        return np.apply_along_axis(self._predict, 1, x_array)


class NSPScorer(object):


    DISCOUNT_PARAM = 10 # TODO un-hard-code me


    def _discount(self, node):
        return np.expm1(-self.DISCOUNT_PARAM * node.max_split_cost)


    def on_create(self, leaf_node):

        if not leaf_node.is_leaf():
            raise ValueError('on_create called for non-leaf node')

        if not hasattr(self, '_pseudocounts'):
            self._tables = {}
            self._pseudocounts = {}
            self._posteriors = {}
            self._prior = normalize(np.ones_like(leaf_node.label_counts)) # Uniform prior

        self._tables[leaf_node] = np.zeros_like(self._prior)

        if leaf_node.parent is None:
            if hasattr(self, '_root'):
                raise ValueError('on_create called with a root node twice')
            self._root = leaf_node

        # For leaf nodes, pseudocounts are actually real counts -- these are stored
        # by the MondrianNode as they're used for other purposes too
        self._pseudocounts[leaf_node] = leaf_node.label_counts


    def on_update(self, leaf_node, labels):

        if not leaf_node.is_leaf():
            raise ValueError('on_update called for non-leaf node')

        node = leaf_node
        pc = self._pseudocounts
        t = self._tables

        # First, update all the class pseudocounts on ancestors of this node
        # TODO optimize this
        for label in labels:

            while True:
                if t[node][label] == 1:
                    break
                
                if not node.is_leaf():
                    # Kneser-Ney approximation, extended to included
                    # data points we've left attached to internal nodes
                    pc[node][label] = t[node.left][label] + t[node.right][label] + min(node.label_counts, 1)

                t[node][label] = min(pc[node][label], 1)

                if node == self._root:
                    break
                node = node.parent

        # Then, update the posterior probabilities, from the root down
        # TODO rewrite this using a generator (i.e. yield)
        # TODO move this into an on_batch_complete method, so we don't do it on EVERY leaf node change
        todo = deque([self._root])
        while todo:
            next_node = todo.pop()

            if next_node == self._root:
                prior = self._prior
            else:
                prior = self._posteriors[next_node.parent]

            d = self._discount(node)
            self._posteriors[node] = (pc[node] - (d * t[node]) + (d * t[node].sum() * prior)) / pc[node].sum()
            if node.left:
                todo.append(node.left)
            if node.right:
                todo.append(node.right)


    def predict(self, x_test):
        """
        predict new label (for classification tasks)
        """
        pred_prob = np.zeros((x_test.shape[0], len(self._prior)))
        prob_not_separated_yet = np.ones(x_test.shape[0])
        prob_separated = np.zeros(x_test.shape[0])
        d_idx_test = {self._root: np.arange(x_test.shape[0])}

        todo = deque([self._root])
        while todo:

            node = todo.pop()
            idx_test = d_idx_test[node]
            if len(idx_test) == 0:
                continue

            if node == self._root:
                print('Inspecting root node')
            elif node.is_leaf():
                print('Inspecting root node')
            else:
                print('Inspecting internal node')

            print(node)
            print('Prior:', self._prior if node == self._root else self._posteriors[node.parent])
            print('Class counts:', node.label_counts)
            print('Pseudocounts:', self._pseudocounts[node])
            print('Tables:', self._tables[node])
            print('Posterior:', self._posteriors[node])

            x = x_test[idx_test, :]
            distance_lower = np.fmax(0, node.min_d - x).sum(1)
            distance_upper = np.fmax(0, x - node.max_d).sum(1)
            expo_parameter = distance_lower + distance_upper
            prob_not_separated_now = np.expm1(-expo_parameter * node.max_split_cost)
            prob_separated_now = 1 - prob_not_separated_now

            if np.isinf(node.max_split_cost):
                idx_zero = expo_parameter == 0          # rare scenario where test point overlaps exactly with a training data point
                prob_not_separated_now[idx_zero] = 1   # to prevent nan in computation above when test point overlaps with training data point
                prob_separated_now[idx_zero] = 0

            # predictions for idx_test_zero
            # data dependent discounting (depending on how far test data point is from the mondrian block)
            idx_non_zero = expo_parameter > 0
            idx_test_non_zero = idx_test[idx_non_zero]
            expo_parameter_non_zero = expo_parameter[idx_non_zero]
            base = self._prior if node == self._root else self._posteriors[node.parent]

            if np.any(idx_non_zero):
                num_tables_k = self._tables[node]
                num_tables = num_tables_k.sum()
                num_customers = self._pseudocounts[node].sum()
                # expected discount (averaging over time of cut which is a truncated exponential)
                discount = (expo_parameter_non_zero / (expo_parameter_non_zero + self.DISCOUNT_PARAM)) \
                                * (-np.expm1(-(expo_parameter_non_zero + self.DISCOUNT_PARAM) * node.max_split_cost)) \
                                / (-np.expm1(-expo_parameter_non_zero * node.max_split_cost))
                discount_per_num_customers = discount / num_customers
                pred_prob_tmp = num_tables * discount_per_num_customers[:, np.newaxis] * base \
                                + self._pseudocounts[node] / num_customers - discount_per_num_customers[:, np.newaxis] * num_tables_k
                pred_prob[idx_test_non_zero, :] += prob_separated_now[idx_non_zero][:, np.newaxis] \
                                                    * prob_not_separated_yet[idx_test_non_zero][:, np.newaxis] * pred_prob_tmp
                prob_not_separated_yet[idx_test] *= prob_not_separated_now

            # predictions for idx_test_zero
            if np.isinf(node.max_split_cost) and np.any(idx_zero):
                idx_test_zero = idx_test[idx_zero]
                pred_prob_node = self._posteriors(node)
                pred_prob[idx_test_zero, :] += prob_not_separated_yet[idx_test_zero][:, np.newaxis] * pred_prob_node

            # try:
            if node.split_point is not None:
                cond = x[:, node.split_dim] <= node.split_point
                d_idx_test[node.left], d_idx_test[node.right] = idx_test[cond], idx_test[~cond]
                todo.append(node.left)
                todo.append(node.right)
            # except KeyError:
            #     pass

        return pred_prob


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


    def predict(self, x):
        return self._scorer.predict(x)


class MondrianForest(object):

    
    def __init__(self, n_trees, n_dims, n_labels, budget, scoring):
        self.trees = [MondrianTree(n_dims, n_labels, budget, scoring) for k in range(n_trees)]
    

    def update(self, data, labels):
        for tree in self.trees:
            tree.extend(data, labels)


    def predict(self, x):
        results = [tree.predict(x) for tree in self.trees]
        return combine_predictions(results)


class ParallelMondrianForest(object):


    def __init__(self, ipy_view, n_dims, n_labels, budget, scoring):
        self._view = ipy_view
        self._remote_name = 'mondrian_worker'
        self._view.apply_sync(init_tree, n_dims, n_labels, budget, scoring, os.path.realpath(__file__), self._remote_name)


    def update(self, data, labels):
        self._view.apply_sync(extend, data, labels, self._remote_name)


    def predict(self, x):
        results = self._view.apply_sync(predict, x, self._remote_name)
        return combine_predictions(results)

