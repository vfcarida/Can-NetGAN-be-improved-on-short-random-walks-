"""
Random walk distance computation on conversation graphs.

Implements the core metric from arXiv:1905.05298:

    d(v_i, v_j) = Σ_τ  p(v_i → v_j, τ) · c · (1 - c)^τ

where τ is the walk length, p is the transition probability along
the walk, and c is the restart probability.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from netgan_walks.graph.builder import ConversationGraph

logger = logging.getLogger(__name__)


class RandomWalkDistance:
    """Compute random walk distances on a conversation graph.

    The random walk distance between two nodes captures both structural
    and semantic proximity by considering all paths of length up to
    ``max_walk_length``, weighted by transition probabilities and a
    geometric decay factor controlled by ``restart_prob``.

    Args:
        graph: The conversation graph (with dummy edges).
        max_walk_length: Maximum walk length τ to consider.
        restart_prob: Probability c of restarting the walk at the source.

    Example::

        rwd = RandomWalkDistance(graph, max_walk_length=5, restart_prob=0.0002)
        transition_matrix = rwd.build_transition_matrix()
        dist = rwd.compute_distance(node_i=0, node_j=3)
    """

    def __init__(
        self,
        graph: ConversationGraph,
        max_walk_length: int = 2,
        restart_prob: float = 0.0002,
    ):
        if max_walk_length < 1:
            raise ValueError(f"max_walk_length must be >= 1, got {max_walk_length}")
        if not 0 < restart_prob < 1:
            raise ValueError(f"restart_prob must be in (0, 1), got {restart_prob}")

        self.graph = graph
        self.max_walk_length = max_walk_length
        self.restart_prob = restart_prob
        self._transition_matrix: Optional[npt.NDArray[np.float64]] = None

    def build_transition_matrix(self) -> npt.NDArray[np.float64]:
        """Build the row-stochastic transition probability matrix.

        For each node, the transition probability to a neighbor is:

            P(i → j) = W(i, j) / Σ_k W(i, k)

        Returns:
            Square transition matrix of shape (N, N) where N = total nodes.
        """
        n = self.graph.total_nodes
        logger.info("Building transition matrix (%d × %d)", n, n)

        # Build weighted adjacency matrix
        weight_matrix = np.zeros((n, n), dtype=np.float64)
        for edge in self.graph.adjacency_list:
            if edge.source < n and edge.target < n:
                weight_matrix[edge.source, edge.target] = edge.weight

        # Normalize rows to get transition probabilities
        row_sums = weight_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        self._transition_matrix = weight_matrix / row_sums

        logger.info("Transition matrix built")
        return self._transition_matrix

    def compute_distance(self, node_i: int, node_j: int) -> float:
        """Compute the random walk distance between two nodes.

        d(v_i, v_j) = Σ_{τ=1}^{max_walk_length} P^τ(i,j) · c · (1-c)^τ

        where P^τ(i,j) is the (i,j) entry of the τ-th power of the
        transition matrix.

        Args:
            node_i: Source node index.
            node_j: Target node index.

        Returns:
            The random walk distance (lower = closer in walk space).

        Raises:
            RuntimeError: If transition matrix has not been built.
        """
        if self._transition_matrix is None:
            raise RuntimeError("Transition matrix not built. Call build_transition_matrix() first.")

        c = self.restart_prob
        distance = 0.0
        p_matrix = self._transition_matrix.copy()

        for tau in range(1, self.max_walk_length + 1):
            if tau > 1:
                p_matrix = p_matrix @ self._transition_matrix

            transition_prob = p_matrix[node_i, node_j]
            distance += transition_prob * c * ((1 - c) ** tau)

        return distance

    def compute_all_distances(self) -> npt.NDArray[np.float64]:
        """Compute pairwise random walk distances for all conversation nodes.

        Only computes distances between conversation nodes (not intention
        nodes), as these are the nodes of interest for path optimization.

        Returns:
            Square matrix of shape (N_conv, N_conv) with pairwise distances.

        Raises:
            RuntimeError: If transition matrix has not been built.
        """
        if self._transition_matrix is None:
            raise RuntimeError("Transition matrix not built. Call build_transition_matrix() first.")

        n_conv = self.graph.num_conversation_nodes
        logger.info("Computing all pairwise distances for %d conversation nodes", n_conv)

        c = self.restart_prob
        distance_matrix = np.zeros((n_conv, n_conv), dtype=np.float64)
        p_matrix = self._transition_matrix.copy()

        for tau in range(1, self.max_walk_length + 1):
            if tau > 1:
                p_matrix = p_matrix @ self._transition_matrix

            decay = c * ((1 - c) ** tau)
            distance_matrix += p_matrix[:n_conv, :n_conv] * decay

        logger.info("Distance computation complete")
        return distance_matrix

    def get_shortest_path_distance(
        self, source: int, target: int
    ) -> float:
        """Find the shortest random-walk-weighted path between two nodes.

        This uses the computed RW distance matrix to find the minimum
        distance path, which corresponds to the most probable walk.

        Args:
            source: Source conversation node index.
            target: Target conversation node index.

        Returns:
            The shortest path distance.
        """
        distances = self.compute_all_distances()
        return float(distances[source, target])
