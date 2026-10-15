"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    # the caller's distance matrix is never mutated + track of trees built so far for each row/column.
    distances = distances.copy().astype(float)
    nodes = [Node(label, None, 0.0, None, 0.0) for label in labels]

    # Repeat joining the closest pair until only two clusters remain.
    while len(nodes) > 2:
        # Based on curr distance matrix, calculate matrix Q
        n = distances.shape[0]
        row_sums = distances.sum(axis=1)
        Q = np.zeros_like(distances)
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] = (n - 2) * distances[i, j] - row_sums[i] - row_sums[j]
        # Find pair of distinct nodes i and j (i≠j) for which Q(i,j) is smallest
        i, j = np.unravel_index(np.argmin(Q + np.diag([np.inf] * n)), Q.shape)
        # 3. new node that joins the nodes i and j + calc the distance from each of the nodes in the pair to this new node.
        left_distance = (distances[i, j] + (row_sums[i] - row_sums[j]) / (n - 2)) / 2
        right_distance = distances[i, j] - left_distance
        new_node = Node(f"Internal-{i}-{j}", nodes[i], left_distance, nodes[j], right_distance)
        # 5. Calculate distance from each of the nodes outside of this pair to the new node.
        remaining = [m for m in range(n) if m != i and m != j]
        new_distances_to_node = [
            (distances[i, m] + distances[j, m] - distances[i, j]) / 2 for m in remaining
        ]

        # Build reduced distance matrix/node list with the new node appended at the end.
        new_n = n - 1
        new_distances = np.zeros((new_n, new_n))
        for a_idx, a in enumerate(remaining):
            for b_idx, b in enumerate(remaining):
                new_distances[a_idx, b_idx] = distances[a, b]
        for a_idx, d in enumerate(new_distances_to_node):
            new_distances[a_idx, -1] = d
            new_distances[-1, a_idx] = d

        distances = new_distances
        nodes = [nodes[m] for m in remaining] + [new_node]
        # 6. Repeat w/ new node and distances until convergence (loop condition above)
    final_distance = distances[0, 1]
    tree = Node("ROOT", nodes[0], final_distance / 2, nodes[1], final_distance / 2)

    return tree


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    next_leaf_y = [0]

    def _plot_node(node: Node, x: float) -> float:
        if node.left is None and node.right is None:
            y = next_leaf_y[0]
            next_leaf_y[0] += 1
            ax.text(x, y, node.name, ha="left", va="center")
            return y
        left_y = _plot_node(node.left, x + node.left_distance)
        right_y = _plot_node(node.right, x + node.right_distance)
        y = (left_y + right_y) / 2
        # pink plot
        ax.plot(x, y, 'o', color="pink")
        ax.plot([x, x + node.left_distance], [left_y, left_y], color="black")
        ax.plot([x, x + node.right_distance], [right_y, right_y], color="black")
        ax.plot([x, x], [left_y, right_y], color="black")
        return y

    _plot_node(tree, 0)

    # Grow the figure with the number of leaves and widen the x-axis so that
    # leaf labels have room to breathe and never overlap each other or get
    # clipped at the right edge, regardless of the caller's figure size.
    n_leaves = next_leaf_y[0]
    fig_width, fig_height = ax.figure.get_size_inches()
    ax.figure.set_size_inches(max(fig_width, 8), max(fig_height, n_leaves * 0.3))
    ax.margins(x=0.3)

    ax.set_xlabel("Distance")
    ax.set_ylabel("Nodes")
    ax.invert_yaxis()
    ax.set_title("Neighbor Joining Tree")

    

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    if tree is None:
        return

    if tree.left is not None:
        sort_children_by_leaves(tree.left)
    if tree.right is not None:
        sort_children_by_leaves(tree.right)

    left_leaves = count_leaves(tree.left) if tree.left is not None else 0
    right_leaves = count_leaves(tree.right) if tree.right is not None else 0

    if left_leaves < right_leaves:
        tree.left, tree.right = tree.right, tree.left
        tree.left_distance, tree.right_distance = tree.right_distance, tree.left_distance

def count_leaves(node: Node) -> int:
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()