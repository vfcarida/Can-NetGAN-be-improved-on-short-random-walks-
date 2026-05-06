"""Tests for the GraphReducer module."""

from __future__ import annotations

import pytest

from netgan_walks.graph.builder import ConversationGraph, Edge
from netgan_walks.graph.reducer import GraphReducer


@pytest.fixture
def simple_graph() -> ConversationGraph:
    """Create a simple graph with dummy edges for reduction testing.

    Graph structure:
        Node 0 (intent=A) → Node 1 (intent=B) → Node 2 (intent=A)
        Dummy edge: 0 → 2, weight=0.1 (below alpha=0.3, same intent → merge)
        Dummy edge: 0 → 1, weight=0.8 (different intent → no merge)
    """
    return ConversationGraph(
        adjacency_list=[
            Edge(source=0, target=1, weight=1.0),
            Edge(source=1, target=2, weight=1.0),
            Edge(source=0, target=3, weight=1.0),  # 0 → intention A
            Edge(source=1, target=4, weight=1.0),  # 1 → intention B
            Edge(source=2, target=3, weight=1.0),  # 2 → intention A
            Edge(source=0, target=2, weight=0.1, is_dummy=True),  # merge candidate
            Edge(source=0, target=1, weight=0.8, is_dummy=True),  # no merge
        ],
        num_conversation_nodes=3,
        num_intention_nodes=2,
        intentions=["A", "B"],
    )


@pytest.fixture
def node_intentions() -> dict[int, str]:
    """Map node indices to their intentions."""
    return {0: "A", 1: "B", 2: "A"}


class TestGraphReducer:
    """Tests for the GraphReducer."""

    def test_invalid_alpha(self, simple_graph: ConversationGraph, node_intentions: dict[int, str]) -> None:
        """Test that invalid alpha values raise ValueError."""
        with pytest.raises(ValueError):
            GraphReducer(simple_graph, node_intentions, alpha=0.0)
        with pytest.raises(ValueError):
            GraphReducer(simple_graph, node_intentions, alpha=1.0)
        with pytest.raises(ValueError):
            GraphReducer(simple_graph, node_intentions, alpha=-0.5)

    def test_reduce_merges_similar_nodes(
        self, simple_graph: ConversationGraph, node_intentions: dict[int, str]
    ) -> None:
        """Test that nodes with same intent and low dummy weight are merged."""
        reducer = GraphReducer(simple_graph, node_intentions, alpha=0.3)
        reduced = reducer.reduce()

        # Node 2 should be removed (merged into node 0)
        all_nodes = set()
        for edge in reduced.adjacency_list:
            all_nodes.add(edge.source)
            all_nodes.add(edge.target)

        assert 2 not in all_nodes

    def test_reduce_preserves_different_intents(
        self, simple_graph: ConversationGraph, node_intentions: dict[int, str]
    ) -> None:
        """Test that nodes with different intents are NOT merged."""
        reducer = GraphReducer(simple_graph, node_intentions, alpha=0.3)
        reduced = reducer.reduce()

        all_nodes = set()
        for edge in reduced.adjacency_list:
            all_nodes.add(edge.source)
            all_nodes.add(edge.target)

        # Node 1 (intent B) should survive
        assert 1 in all_nodes

    def test_reduce_no_candidates(self) -> None:
        """Test that graph with no merge candidates is unchanged."""
        graph = ConversationGraph(
            adjacency_list=[
                Edge(source=0, target=1, weight=1.0),
                Edge(source=0, target=2, weight=1.0),
            ],
            num_conversation_nodes=2,
            num_intention_nodes=1,
        )
        node_intentions = {0: "A", 1: "B"}
        reducer = GraphReducer(graph, node_intentions, alpha=0.3)
        reduced = reducer.reduce()

        assert len(reduced.adjacency_list) == 2

    def test_does_not_mutate_original(
        self, simple_graph: ConversationGraph, node_intentions: dict[int, str]
    ) -> None:
        """Test that the original graph is not modified."""
        original_edge_count = len(simple_graph.adjacency_list)
        reducer = GraphReducer(simple_graph, node_intentions, alpha=0.3)
        reducer.reduce()

        assert len(simple_graph.adjacency_list) == original_edge_count
