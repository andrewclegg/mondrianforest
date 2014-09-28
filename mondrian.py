import random
import os

import numpy as np

from collections import deque

from functools import lru_cache

from numba import jit, njit

from scipy.stats import truncexpon


def scorer(name):
    """
    Initialize a scorer of the specified type (currently 'nsp' or 'simple') and return it.
    """
    if name == 'simple':
        scorer = SimpleScorer()
    elif name == 'nsp':
        scorer = NSPScorerNaive()
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


def depth_first(node):
    """
    Depth-first node iterator, not currently using this but it
    might come in handy.
    """
    if node is None:
        return
    yield node
    for kid in depth_first(node.left):
        yield node
    for kid in depth_first(node.right):
        yield node


def get_tables(node):
    """
    Get the IKN approximation of the label counts at this node.
    More specifically, this returns a boolean vector containing
    True where this node (or a child) *does* contain that label,
    and False otherwise.

    We use a small modification of the formulation in the paper.
    Specifically, non-leaf nodes can have counts associated with
    them, which they observed before getting split. These are
    taken into account along with the counts on child nodes.
    """
    my_tables = node.label_counts.astype(bool)
    my_tables = my_tables | (node.left is not None and get_tables(node.left))
    my_tables = my_tables | (node.right is not None and get_tables(node.right))
    return my_tables


def get_prior_class_probs(node, discount_param):
    """
    Get the IKN prior for the class probabilities at this node,
    with the specified discount rate.
    """
    if node.parent is not None:
        return get_posterior_class_probs(node.parent, discount_param)
    else:
        return normalize(np.ones_like(node.label_counts))


def get_posterior_class_probs(node, discount_param):
    """
    Get the posterior class probabilities for a node, taking into
    account its IKN prior along with any counts observed at (or
    below) this node.
    """
    prior = get_prior_class_probs(node, discount_param)
    if np.any(node.label_counts):
        sum_counts = np.sum(node.label_counts)
        discount = np.exp(-discount_param * node.max_split_cost)
        tables = get_tables(node)
        return calc_posterior(node.label_counts, discount, tables, prior)
    else:
        return prior


def calc_posterior(label_counts, discount, tables, prior):
    """
    Calculate the posterior class probabilities for a node.
    """
    return (label_counts
            - discount * tables
            + discount * tables.sum() * prior) / label_counts.sum()


@lru_cache(maxsize=10000)
def calc_discount(discount_factor, split_cost):
    """
    Calculate the discount parameter for a given split.

    discount_factor is a user-supplied constant, defaulting
    to 10. This is γ in the paper.

    split_cost reflects time since the parent node was split.
    This is ∆_j == τ_j - τ_parent(j) in the paper.

    A higher discount_factor or split cost (age) will lead to
    a smaller result, which will have the effect of weighting
    the node's own data more strongly compared to the priors
    inherited from its parent.
    """
    return np.exp(-discount_factor * split_cost)


@lru_cache(maxsize=10000)
def expected_discount(discount_factor, exp_rate, upper_bound):
    """
    Calculate the expected discount for a new instance which we want
    to make a prediction about, with regard to a node in a tree.
    This is based on an exponential distribution, truncated to the
    interval [0, upper_bound].

    exp_rate is the rate parameter on the distribution -- η_j(x)
    in the paper. This reflects how far outside the node's bounding
    box the new point falls. The further this is, the more likely a
    split is. The split costs described above are drawn from this
    distribution, and also provide the upper bound (not sure I follow
    this yet).
    
    The discount factor is also applied as before.
    """
    func = lambda d: np.exp(-discount_factor * d)
    trunc_exp = truncexpon(exp_rate)
    return trunc_exp.dist.expect(func, trunc_exp.args, lb=0, ub=upper_bound)


@jit('f8[:](f8[:,:])')
def colwise_max(data):
    n_rows, n_cols = data.shape
    res = np.empty(n_cols, dtype=np.float64)
    colwise_max__(data, n_rows, n_cols, res)
    return res


@njit('void(f8[:,:],u4,u4,f8[:])')
def colwise_max__(data, n_rows, n_cols, res):
    for j in range(n_cols):
        curr_max = data[0, j]
        for i in range(1, n_rows):
            if data[i, j] > curr_max:
                curr_max = data[i, j]
        res[j] = curr_max


@jit('f8[:](f8[:,:])')
def colwise_min(data):
    n_rows, n_cols = data.shape
    res = np.empty(n_cols, dtype=np.float64)
    colwise_min__(data, n_rows, n_cols, res)
    return res


@njit('void(f8[:,:],u4,u4,f8[:])')
def colwise_min__(data, n_rows, n_cols, res):
    for j in range(n_cols):
        curr_min = data[0, j]
        for i in range(1, n_rows):
            if data[i, j] < curr_min:
                curr_min = data[i, j]
        res[j] = curr_min


def sample_multinomial_scores(scores):
    scores_cumsum = np.cumsum(scores)
    s = scores_cumsum[-1] * np.random.rand(1)
    k = int(np.sum(s > scores_cumsum))
    return k


@jit
def normalize(array):
    res = np.empty_like(array)
    if array.ndim > 1:
        normalize__(array, res)
    else:
        normalize_(array, res)
    return res


@njit('void(f8[:],f8[:])')
def normalize_(vec, res):
    total = vec[0]
    length = len(vec)
    for i in range(1, length):
        total += vec[i]
    for i in range(length):
        res[i] = vec[i] / total


@njit('void(f8[:,:],f8[:,:])')
def normalize__(array, res):
    n_rows, n_cols = array.shape
    for i in range(n_rows):
        total = array[i, 0]
        for j in range(1, n_cols):
            total += array[i, j]
        for j in range(n_cols):
            res[i, j] = array[i, j] / total


@njit('void(f8[:,:],i4,i4,f8,b1[:])')
def split__(array, length, dim, threshold, res):
    for i in range(length):
        if array[i, dim] <= threshold:
            res[i] = True


@jit('b1[:](f8[:,:],i4,f8)')
def split(array, dim, threshold):
    length = len(array)
    res = np.zeros((length), dtype=bool)
    split__(array, length, dim, threshold, res)
    return res


@njit('f8(f8[:,:],f8[:],f8[:])')
def calc_bbox_growth(data, node_min_d, node_max_d):
    """
    Calculate the difference in linear dimension between the
    current node, and the incoming data. Roughly, this means
    calculating how much the bounding box would have to grow
    to accommodate all the new points, in each dimension, and
    then summing across these dimensions.
    """
    n_rows, n_cols = data.shape
    total = 0
    for j in range(n_cols):

        # Keep track of maximum extension required for lower
        # and upper bound, respectively, in this dimension
        l_extension = 0
        u_extension = 0

        # Loop over data points
        for i in range(0, n_rows):

            # Does this data point exceed the lower bound by more
            # than previous furthest example?
            e = node_min_d[j] - data[i, j]
            if e > l_extension:
                l_extension = e

            # Does this data point exceed the upper bound by more
            # than previous furthest example?
            e = data[i, j] - node_max_d[j]
            if e > u_extension:
                u_extension = e

        # Add furthest extensions in this dimension to running total
        total += l_extension + u_extension
        
    return total 


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
        self.min_d = colwise_min(data)
        self.max_d = colwise_max(data)
        self.range_d = self.max_d - self.min_d
        self.sum_range_d = np.sum(self.range_d)
        self.label_counts += np.bincount(labels, minlength=len(self.label_counts))
        self._scorer.on_update(self, labels)
        

    def apply_split(self, data):
        # Apply this node's existing splitting criterion to some data (vector or array)
        # and return a boolean index (True==goes left, False==goes right)
        dim = self.split_dim
        threshold = self.split_point
        if data.ndim > 1:
            return split(data, dim, threshold)
        else:
            return data[dim] <= threshold
    
    
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


    def __init__(self):
        self._root = None


    def on_create(self, node):
        # TODO can we do some clever caching so we do less calculation during prediction?
        if self._root is None:
            if node.parent is not None:
                raise ValueError('Error: called on_create with a non-root node the first time')
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


class NSPScorerNaive(object):


    DISCOUNT_FACTOR = 10 # TODO un-hard-code me

    
    def __init__(self):
        self._root = None


    # N.B. This is directly copied from SimpleScorer -- temporary measure
    def on_create(self, node):
        if self._root is None:
            if node.parent is not None:
                raise ValueError('Error: called on_create with a non-root node the first time')
            self._root = node


    def on_update(self, node, labels):
        pass


    def _predict(self, x):
        
#        print('Predicting for one row:', x)

        previous_posterior = None
        node = self._root
        p_not_separated_yet = 1
        s = np.zeros_like(self._root.label_counts)

        while True:

#            if node == self._root:
#                print('Inspecting root node')
#            elif node.is_leaf():
#                print('Inspecting leaf node')
#            else:
#                print('Inspecting internal node')

            discount_ubound = node.max_split_cost
            discount_rate_param = np.sum(np.fmax(x - node.max_d, 0)
                                         + np.fmax(node.min_d - x, 0))
            p_split = 1 - np.exp(-discount_ubound * discount_rate_param)

            if previous_posterior is None:
                prior = normalize(np.ones_like(node.label_counts))
            else:
                prior = previous_posterior

#            print('p_not_separated_yet', p_not_separated_yet)
#            print('discount_ubound', discount_ubound)
#            print('discount_rate_param', discount_rate_param)
#            print('p_split', p_split)
#            print('previous_posterior', previous_posterior)
#            print('prior', prior)

            if p_split > 0:

                expected_d = expected_discount(
                        self.DISCOUNT_FACTOR,
                        discount_rate_param,
                        discount_ubound)

                tab_new_node = np.fmin(node.label_counts, 1)
                counts_new_node = tab_new_node
                posterior_new_node = calc_posterior(counts_new_node, expected_d,
                                                    tab_new_node, prior)

                s = s + p_not_separated_yet * p_split * posterior_new_node

#                print('expected_d', expected_d)
#                print('tab_new_node', tab_new_node)
#                print('posterior_new_node', posterior_new_node)
#                print('s', s)

            if node.is_leaf():
                s = s + p_not_separated_yet * (1 - p_split) * previous_posterior
#                print('s', s)
                return s
            else:
                p_not_separated_yet = p_not_separated_yet * (1 - p_split)
                if x[node.split_dim] <= node.split_point:
                    node = node.left
                else:
                    node = node.right
                previous_posterior = get_posterior_class_probs(node, self.DISCOUNT_FACTOR)


    def predict(self, x):

        x_array = np.atleast_2d(x)
        return np.apply_along_axis(self._predict, 1, x_array)


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
        
#        min_d = colwise_min(data)
#        max_d = colwise_max(data)
#        additional_extent_lower = np.fmax(0, node.min_d - min_d)
#        additional_extent_upper = np.fmax(0, max_d - node.max_d)
#        expo_parameter = np.sum([additional_extent_lower, additional_extent_upper])
        expo_parameter = calc_bbox_growth(data, node.min_d, node.max_d)
        
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
            self._split_node(node, data, labels, split_cost)
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


    def _split_node(self, node, data, labels, split_cost):
        
        assert not node.is_leaf() # Mutate leaf nodes by growing, not splitting
        
        min_d = colwise_min(data)
        max_d = colwise_max(data)
        additional_extent_lower = np.fmax(0, node.min_d - min_d)
        additional_extent_upper = np.fmax(0, max_d - node.max_d)

        # Create new parent node which is a near-copy of the one that's splitting
        new_parent = MondrianNode(self.n_dims, self.n_labels, node.parent, node.budget, self._scorer)
        new_parent.min_d = np.fmin(min_d, node.min_d)
        new_parent.max_d = np.fmax(max_d, node.max_d)
        new_parent.range_d = new_parent.max_d - new_parent.min_d
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

