"""
Graph construction from conversation data.

This module builds graph representations (adjacency list and adjacency
matrix) from preprocessed conversation data. It implements the tree-based
conversation structure and "dummy edge" mechanism described in the paper
(arXiv:1905.05298, Section 3).

Dummy edges connect semantically similar nodes across different
conversation trees, with weights computed as:

    W(i, j) = 1 - d(i, j) / D

where d(i,j) is the semantic distance and D is the maximum distance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """A single directed edge in the conversation graph.

    Attributes:
        source: Index of the source node.
        target: Index of the target node.
        weight: Edge weight (1.0 for structural edges, computed for dummy edges).
        is_dummy: Whether this is a dummy (semantic) edge.
    """

    source: int
    target: int
    weight: float = 1.0
    is_dummy: bool = False


@dataclass
class ConversationGraph:
    """Container for the conversation graph structure.

    Attributes:
        adjacency_list: List of Edge objects.
        num_conversation_nodes: Number of nodes corresponding to conversations.
        num_intention_nodes: Number of nodes corresponding to unique intentions.
        intentions: Ordered list of intention names.
    """

    adjacency_list: list[Edge] = field(default_factory=list)
    num_conversation_nodes: int = 0
    num_intention_nodes: int = 0
    intentions: list[str] = field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        """Total number of nodes in the graph."""
        return self.num_conversation_nodes + self.num_intention_nodes


class GraphBuilder:
    """Build conversation graphs from preprocessed DataFrames.

    This builder constructs graph representations by:
    1. Creating tree structures for individual conversations (parent → child).
    2. Linking each message node to its classified intention node.
    3. Optionally adding dummy edges based on semantic distances.

    Args:
        conversations: Cleaned conversation DataFrame with columns
            ``session_id``, ``intention_name``, ``user_message``.
        intentions: Ordered list of unique intention names.
        max_conversations: Maximum number of conversations to process.
            Defaults to processing all available data.

    Example::

        builder = GraphBuilder(conversations_df, intentions_list, max_conversations=10000)
        graph = builder.build_adjacency_list()
        builder.add_dummy_edges(graph, semantic_distances)
    """

    def __init__(
        self,
        conversations: pd.DataFrame,
        intentions: list[str],
        max_conversations: Optional[int] = None,
    ):
        self.conversations = conversations
        self.intentions = intentions
        self.max_conversations = max_conversations or len(conversations)

    def build_adjacency_list(self) -> ConversationGraph:
        """Build the graph as an adjacency list of Edge objects.

        Each conversation is modeled as a tree:
        - Sequential messages within the same session form parent → child edges.
        - Each message node links to its corresponding intention node.
        - Self-loops are added to each node (for random walk computations).

        Returns:
            A :class:`ConversationGraph` containing the adjacency list.
        """
        logger.info(
            "Building adjacency list for %d conversations", self.max_conversations
        )

        graph = ConversationGraph(
            intentions=self.intentions,
            num_intention_nodes=len(self.intentions),
        )

        subset = self.conversations.iloc[: self.max_conversations]
        conversation_count = len(subset)
        graph.num_conversation_nodes = conversation_count

        node_index = 0
        previous_session_id: Optional[str] = None

        for _, row in subset.iterrows():
            current_session_id = row["session_id"]
            intention_name = row["intention_name"]

            # --- Link node to its intention ---
            intention_index = self.intentions.index(intention_name)
            graph.adjacency_list.append(
                Edge(
                    source=node_index,
                    target=conversation_count + intention_index,
                    weight=1.0,
                )
            )

            # --- Self-loop (required for random walk distance computation) ---
            graph.adjacency_list.append(
                Edge(source=node_index, target=node_index, weight=1.0)
            )

            # --- Sequential edge within same session (tree structure) ---
            if previous_session_id == current_session_id:
                graph.adjacency_list.append(
                    Edge(source=node_index - 1, target=node_index, weight=1.0)
                )

            if previous_session_id is None:
                previous_session_id = current_session_id

            previous_session_id = current_session_id
            node_index += 1

        logger.info(
            "Built graph: %d nodes, %d edges",
            graph.total_nodes,
            len(graph.adjacency_list),
        )
        return graph

    def build_adjacency_matrix(self) -> npt.NDArray[np.float64]:
        """Build the graph as a dense adjacency matrix.

        Returns:
            A 2D NumPy array of shape ``(N, N + I)`` where N is the number
            of conversation nodes and I is the number of intention nodes.

        Note:
            This representation is memory-intensive for large datasets.
            Prefer :meth:`build_adjacency_list` for > 50k conversations.
        """
        logger.info("Building adjacency matrix (dense representation)")

        subset = self.conversations.iloc[: self.max_conversations]
        conversation_count = len(subset)
        total_cols = conversation_count + len(self.intentions)

        adj_matrix = np.zeros((conversation_count, total_cols), dtype=np.float64)

        node_index = 0
        previous_session_id: Optional[str] = None

        for _, row in subset.iterrows():
            current_session_id = row["session_id"]
            intention_name = row["intention_name"]

            # Link node to intention
            intention_index = self.intentions.index(intention_name)
            adj_matrix[node_index, conversation_count + intention_index] = 1.0

            # Sequential edge within same session
            if previous_session_id is None:
                previous_session_id = current_session_id

            if previous_session_id == current_session_id and node_index > 0:
                adj_matrix[node_index - 1, node_index] = 1.0

            previous_session_id = current_session_id
            node_index += 1

        logger.info("Built adjacency matrix: shape %s", adj_matrix.shape)
        return adj_matrix

    @staticmethod
    def add_dummy_edges(
        graph: ConversationGraph,
        semantic_distances: dict[tuple[int, int], float],
    ) -> ConversationGraph:
        """Add dummy (semantic) edges between nodes of different conversations.

        Dummy edges connect semantically similar questions across different
        conversation trees. Their weights are inversely proportional to
        semantic distance:

            W(i, j) = 1 - d(i, j) / D

        where D is the maximum pairwise distance observed.

        Args:
            graph: The graph to augment with dummy edges.
            semantic_distances: Dictionary mapping ``(node_i, node_j)``
                pairs to their semantic (WMD) distance.

        Returns:
            The same graph, augmented with dummy edges.
        """
        if not semantic_distances:
            logger.warning("No semantic distances provided — skipping dummy edges")
            return graph

        max_distance = max(semantic_distances.values())
        if max_distance == 0:
            logger.warning("Max semantic distance is 0 — skipping dummy edges")
            return graph

        logger.info(
            "Adding dummy edges from %d distance pairs (D_max=%.4f)",
            len(semantic_distances),
            max_distance,
        )

        dummy_count = 0
        for (node_i, node_j), distance in semantic_distances.items():
            weight = 1.0 - (distance / max_distance)

            # Only add edges with positive weight (semantically close nodes)
            if weight > 0:
                graph.adjacency_list.append(
                    Edge(
                        source=node_i,
                        target=node_j,
                        weight=weight,
                        is_dummy=True,
                    )
                )
                dummy_count += 1

        logger.info("Added %d dummy edges", dummy_count)
        return graph
