"""
Graph reduction by merging semantically similar nodes.

Implements Step 4 of the methodology (arXiv:1905.05298):
reducing graph size by eliminating redundant nodes that are
semantically similar and share the same classified intention.
"""

from __future__ import annotations

import logging
from copy import deepcopy

from netgan_walks.graph.builder import ConversationGraph, Edge

logger = logging.getLogger(__name__)


class GraphReducer:
    """Reduce a conversation graph by merging semantically equivalent nodes.

    Args:
        graph: The conversation graph to reduce.
        node_intentions: Mapping from node index to intention name.
        alpha: Similarity threshold for merging (0 < alpha < 1).
    """

    def __init__(
        self,
        graph: ConversationGraph,
        node_intentions: dict[int, str],
        alpha: float = 0.3,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.graph = deepcopy(graph)
        self.node_intentions = node_intentions
        self.alpha = alpha

    def reduce(self) -> ConversationGraph:
        """Execute the full graph reduction pipeline.

        Returns:
            A new ConversationGraph with redundant nodes merged.
        """
        logger.info("Starting reduction (alpha=%.3f, edges=%d)", self.alpha, len(self.graph.adjacency_list))
        merge_candidates = self._find_merge_candidates()

        if not merge_candidates:
            return self.graph

        removed_nodes: set[int] = set()
        for target_node, absorbing_node in merge_candidates:
            if target_node in removed_nodes or absorbing_node in removed_nodes:
                continue
            self._merge_nodes(absorbing_node, target_node)
            removed_nodes.add(target_node)

        self.graph.adjacency_list = [
            e for e in self.graph.adjacency_list
            if e.source not in removed_nodes and e.target not in removed_nodes
        ]

        logger.info("Removed %d nodes, %d edges remaining", len(removed_nodes), len(self.graph.adjacency_list))
        return self.graph

    def _find_merge_candidates(self) -> list[tuple[int, int]]:
        """Identify (target_to_remove, absorbing_node) pairs for merging."""
        candidates: list[tuple[int, int]] = []
        for edge in self.graph.adjacency_list:
            if not edge.is_dummy:
                continue
            src_intent = self.node_intentions.get(edge.source)
            tgt_intent = self.node_intentions.get(edge.target)
            if src_intent and tgt_intent and src_intent == tgt_intent and edge.weight < self.alpha:
                candidates.append((edge.target, edge.source))
        return candidates

    def _merge_nodes(self, absorbing: int, target: int) -> None:
        """Merge target into absorbing by redirecting edges."""
        bridge_weight = self._get_edge_weight(absorbing, target)
        new_edges: list[Edge] = []

        for edge in self.graph.adjacency_list:
            if edge.target == target and edge.source != absorbing:
                new_edges.append(Edge(source=edge.source, target=absorbing, weight=edge.weight, is_dummy=edge.is_dummy))
            if edge.source == target and edge.target != absorbing:
                if edge.target >= self.graph.num_conversation_nodes:
                    continue
                new_edges.append(Edge(source=absorbing, target=edge.target, weight=bridge_weight * edge.weight, is_dummy=True))

        self.graph.adjacency_list.extend(new_edges)

    def _get_edge_weight(self, source: int, target: int) -> float:
        """Look up the weight of a specific edge."""
        for edge in self.graph.adjacency_list:
            if edge.source == source and edge.target == target:
                return edge.weight
        return 0.0
