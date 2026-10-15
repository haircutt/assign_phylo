"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


def global_alignment(seq1: str, seq2: str, scoring_function) -> tuple:
    """Global sequence alignment using the Needleman-Wunsch algorithm.

    Indels are denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """
    gap_score = -1  # gap penalty for global alignment - should be adjustable
    rows = len(seq1) + 1
    columns = len(seq2) + 1
    score_matrix = [[0] * columns for _ in range(rows)]

    for i in range(1, rows):
        score_matrix[i][0] = score_matrix[i - 1][0] + gap_score
    for j in range(1, columns):
        score_matrix[0][j] = score_matrix[0][j - 1] + gap_score

    for i in range(1, rows):
        for j in range(1, columns):
            diagonal = score_matrix[i - 1][j - 1] + scoring_function(
                seq1[i - 1], seq2[j - 1]
            )
            up = score_matrix[i - 1][j] + gap_score
            left = score_matrix[i][j - 1] + gap_score
            score_matrix[i][j] = max(diagonal, up, left)

    aligned_seq1 = []
    aligned_seq2 = []
    i = len(seq1)
    j = len(seq2)
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and score_matrix[i][j]
            == score_matrix[i - 1][j - 1]
            + scoring_function(seq1[i - 1], seq2[j - 1])
        ):
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and score_matrix[i][j] == score_matrix[i - 1][j] + gap_score:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1
        else:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1

    return (
        "".join(reversed(aligned_seq1)),
        "".join(reversed(aligned_seq2)),
        float(score_matrix[-1][-1]),
    )


def global_alignment_fast(seq1: str, seq2: str, scoring_function, gap_score: float = -1) -> tuple:
    """Updated Needleman-Wunsch alignment to optimise time spent running in notebook.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable
    gap_score: float
        Linear gap penalty (applied per inserted/deleted character).

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    """
    n, m = len(seq1), len(seq2)
    alphabet = sorted(set(seq1) | set(seq2))
    char_to_code = {c: k for k, c in enumerate(alphabet)}
    score_lookup = np.array(
        [[scoring_function(a, b) for b in alphabet] for a in alphabet], dtype=np.float64
    )
    seq1_codes = np.array([char_to_code[c] for c in seq1])
    seq2_codes = np.array([char_to_code[c] for c in seq2])

    traceback = np.zeros((n + 1, m), dtype=np.int8)
    left_source = np.tile(np.arange(m, dtype=np.uint16), (n + 1, 1))

    prev_row = (np.arange(m + 1) * gap_score).astype(np.float64)
    col_idx = np.arange(m)

    for i in range(1, n + 1):
        match_scores = score_lookup[seq1_codes[i - 1], seq2_codes]
        diag = prev_row[0:m] + match_scores
        up = prev_row[1:m + 1] + gap_score
        cand = np.maximum(diag, up)
        from_diag = diag >= up
        adj = cand - gap_score * col_idx
        running_max = np.maximum.accumulate(adj)
        marker = np.where(adj == running_max, col_idx, -1)
        running_idx = np.maximum.accumulate(marker)

        left_chain_val = running_max + gap_score * col_idx
        use_left_chain = left_chain_val > cand

        new_row = np.where(use_left_chain, left_chain_val, cand)

        traceback[i] = np.where(use_left_chain, 2, np.where(from_diag, 0, 1)).astype(np.int8)
        left_source[i] = np.where(use_left_chain, running_idx, col_idx)

        new_row_full = np.empty(m + 1)
        new_row_full[0] = prev_row[0] + gap_score
        new_row_full[1:] = new_row
        prev_row = new_row_full

    final_score = prev_row[-1]

    aligned_seq1 = []
    aligned_seq2 = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[j - 1])
            j -= 1
            continue
        if j == 0:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1
            continue
        direction = traceback[i, j - 1]
        if direction == 0:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif direction == 1:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append("-")
            i -= 1
        else:
            src = int(left_source[i, j - 1])
            for col in range(j - 1, src, -1):
                aligned_seq1.append("-")
                aligned_seq2.append(seq2[col])
            j = src + 1

    return (
        "".join(reversed(aligned_seq1)),
        "".join(reversed(aligned_seq2)),
        float(final_score),
    )


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

    # Optional {leaf_name: color} mapping (e.g. to highlight coronavirus
    # genera) used to colour each leaf's label; defaults to black.
    label_colors = kwargs.get("label_colors", {})

    next_leaf_y = [0]

    def _plot_node(node: Node, x: float) -> float:
        if node.left is None and node.right is None:
            y = next_leaf_y[0]
            next_leaf_y[0] += 1
            color = label_colors.get(node.name, "black")
            ax.text(x, y, node.name, ha="left", va="center", color=color)
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

    # Grow the figure with the number of leaves so leaf labels have room to
    # breathe and never overlap each other, regardless of the caller's
    # figure size.
    n_leaves = next_leaf_y[0]
    fig_width, fig_height = ax.figure.get_size_inches()
    ax.figure.set_size_inches(max(fig_width, 8), max(fig_height, n_leaves * 0.3))
    ax.margins(x=0.3)

    # Measure the actual rendered width of the leaf labels (in data units)
    # and extend the right x-limit so the longest label is never clipped,
    # no matter how long the species names are.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    text_rights = [
        inv.transform((txt.get_window_extent(renderer=renderer).x1, 0))[0]
        for txt in ax.texts
    ]
    if text_rights:
        xlim_left, xlim_right = ax.get_xlim()
        ax.set_xlim(xlim_left, max(xlim_right, max(text_rights) * 1.02))

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