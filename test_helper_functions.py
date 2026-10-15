import unittest
from numbers import Number

import numpy as np
import pickle


def is_isomorhipic(tree1: "Node", tree2: "Node") -> bool:
    """Check if two trees are isomorphic, checking their structure and distances."""
    if tree1 is None and tree2 is None:
        return True
    if (tree1 is None and tree2 is not None) or (tree1 is not None and tree2 is None):
        return False

    isomorphic_orig_order = (
        is_isomorhipic(tree1.left, tree2.left)
        and is_isomorhipic(tree1.right, tree2.right)
        and np.isclose(tree1.left_distance, tree2.left_distance)
        and np.isclose(tree1.right_distance, tree2.right_distance)
    )
    isomorphic_flip_order = (
        is_isomorhipic(tree1.left, tree2.right)
        and is_isomorhipic(tree1.right, tree2.left)
        and np.isclose(tree1.left_distance, tree2.right_distance)
        and np.isclose(tree1.right_distance, tree2.left_distance)
    )

    if isomorphic_orig_order or isomorphic_flip_order:
        return True

    return False


class TestNeighborJoining(unittest.TestCase):
    def test_neighbor_joining_node_api(self):
        from helper_functions import neighbor_joining, Node

        distances = np.array(
            [
                [0, 14, 14, 12],
                [0, 0, 16, 14],
                [0, 0, 0, 6],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )
        distances = distances + distances.T

        original_distances = distances.copy()

        tree: Node = neighbor_joining(distances, list("ABCD"))

        # Check that the distance matrix remained in-tact
        np.testing.assert_almost_equal(
            distances,
            original_distances,
            err_msg="`neighbor_joining` changed original distance matrix!",
        )

        # Check that NJ returns a Node object
        self.assertIsInstance(tree, Node)
        # Check if the Node object has a valid public API
        self.assertTrue(hasattr(tree, "name"))

        self.assertIsInstance(tree.left, Node)
        self.assertIsInstance(tree.right, Node)

        self.assertIsInstance(tree.left_distance, Number)
        self.assertIsInstance(tree.right_distance, Number)

    def test_textbook_example(self):
        from helper_functions import Node, neighbor_joining

        distances = np.array(
            [
                [0, 5, 4, 9, 8],
                [0, 0, 5, 10, 9],
                [0, 0, 0, 7, 6],
                [0, 0, 0, 0, 7],
                [0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        distances = distances + distances.T

        result = neighbor_joining(distances, list("ABCDE"))
        expected = Node(
            "ROOT",
            left=Node(
                "X",
                left=Node(
                    "W",
                    left=Node("D", None, 0, None, 0),
                    left_distance=4,
                    right=Node("E", None, 0, None, 0),
                    right_distance=3,
                ),
                left_distance=2,
                right=Node("C", None, 0, None, 0),
                right_distance=1,
            ),
            left_distance=0.5,
            right=Node(
                "Y",
                left=Node("A", None, 0, None, 0),
                left_distance=2,
                right=Node("B", None, 0, None, 0),
                right_distance=3,
            ),
            right_distance=0.5,
        )

        self.assertTrue(is_isomorhipic(result, expected))


class TestPlottingDendrogram(unittest.TestCase):
    def setUp(self) -> None:
        from helper_functions import Node

        self.test_case_1_lines = [
            [[0,0],[0.5,2.5]],
            [[2,2],[2,3]],
            [[2,2],[0,1]],
            [[2,8],[0,0]],
            [[0,2],[0.5,0.5]],
            [[2,10],[1,1]],
            [[2,6],[2,2]],
            [[0,2],[2.5,2.5]],
            [[2,4],[3,3]],
            [[-0.2,0],[1.5,1.5]],
        ]

        self.test_case_1_leaf_labels = {
            "A": [8, 0],
            "B": [10, 1],
            "C": [6, 2],
            "D": [4, 3],
        }

        self.test_case_2_lines = [
            [[2.5, 6.5], [0, 0]],
            [[2.5, 5.5], [1, 1]],
            [[0.5, 2.5], [0.5, 0.5]],
            [[0.5, 1.5], [2, 2]],
            [[0, 0.5], [4 / 3, 4 / 3]],
            [[-0.2, 0], [12 / 5, 12 / 5]],
            [[0.5, 2.5], [3, 3]],
            [[0, 0.5], [3.5, 3.5]],
            [[0.5, 3.5], [4, 4]],
            [[2.5, 2.5], [0, 1]],
            [[0.5, 0.5], [0.5, 2]],
            [[0, 0], [4 / 3, 3.5]],
            [[0.5, 0.5], [3, 4]],
        ]
        self.test_case_2_leaf_labels = {
            "D": [6.5, 0],
            "E": [5.5, 1],
            "C": [1.5, 2],
            "A": [2.5, 3],
            "B": [3.5, 4],
        }
        self.tree_1 = Node(
            "ROOT",
            left=Node(
                "X",
                left=Node(
                    "A",
                    left=None,
                    left_distance=0,
                    right=None,
                    right_distance=0,
                ),
                left_distance=6,
                right=Node(
                    "B",
                    left=None,
                    left_distance=0,
                    right=None,
                    right_distance=0,
                ),
                right_distance=8,
            ),
            left_distance=2,
            right=Node(
                "Y",
                left=Node(
                    "C",
                    left=None,
                    left_distance=0,
                    right=None,
                    right_distance=0,
                ),
                left_distance=4,
                right=Node(
                    "D",
                    left=None,
                    left_distance=0,
                    right=None,
                    right_distance=0,
                ),
                right_distance=2,
            ),
            right_distance=2,
        )
        self.tree_2 = Node(
            "ROOT",
            left=Node(
                "X",
                left=Node(
                    "W",
                    left=Node(
                        "D",
                        left=None,
                        left_distance=0,
                        right=None,
                        right_distance=0,
                    ),
                    left_distance=4,
                    right=Node(
                        "E",
                        left=None,
                        left_distance=0,
                        right=None,
                        right_distance=0,
                    ),
                    right_distance=3,
                ),
                left_distance=2,
                right=Node(
                    "C",
                    left=None,
                    left_distance=0,
                    right=None,
                    right_distance=0,
                ),
                right_distance=1,
            ),
            left_distance=0.5,
            right=Node(
                "Y",
                left=Node(
                    "A",
                    left=None,
                    left_distance=0,
                    right=None,
                    right_distance=0,
                ),
                left_distance=2,
                right=Node(
                    "B",
                    left=None,
                    left_distance=0,
                    right=None,
                    right_distance=0,
                ),
                right_distance=3,
            ),
            right_distance=0.5,
        )

    def plot_gt_lines(self, lines: list, leaf_labels: dict, ax):

        for line in lines:
            ax.plot(line[0], line[1], color="k", lw=1.5)

        for leaf, loc in leaf_labels.items():
            ax.text(
                loc[0],
                loc[1],
                f"  {leaf}",
                color="k",
                fontsize=12,
                verticalalignment="center",
            )

        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)

    def test_plotting_dendrogram(self):

        from helper_functions import plot_nj_tree
        import matplotlib.pyplot as plt

        f, ax = plt.subplots(2, 2, figsize=(10, 7))

        self.plot_gt_lines(
            self.test_case_1_lines, self.test_case_1_leaf_labels, ax[0][0]
        )
        self.plot_gt_lines(
            self.test_case_2_lines, self.test_case_2_leaf_labels, ax[0][1]
        )

        plot_nj_tree(self.tree_1, ax=ax[1][0])
        plot_nj_tree(self.tree_2, ax=ax[1][1])

        ax[0][0].set_title("Test Case 1", fontsize=12)
        ax[0][1].set_title("Test Case 2", fontsize=12)

        ax[0][0].set_ylabel("Ground truth", fontsize=12)
        ax[1][0].set_ylabel("Your implementation", fontsize=12)

        f.savefig("tests/test_plotting_dendrogram.png", dpi=400, bbox_inches="tight")

def test_sort_children_by_leaves(self):
        from helper_functions import sort_children_by_leaves, count_leaves

        # Example test case
        tree = self.tree_1  # Replace with an actual tree instance
        sort_children_by_leaves(tree)
        left_leaves = count_leaves(tree.left)
        right_leaves = count_leaves(tree.right)
        self.assertGreaterEqual(left_leaves, right_leaves)
if __name__ == "__main__":
    unittest.main()
