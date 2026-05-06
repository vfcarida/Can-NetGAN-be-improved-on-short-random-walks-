"""Tests for the GraphBuilder module."""

from __future__ import annotations

import pandas as pd
import pytest

from netgan_walks.graph.builder import ConversationGraph, Edge, GraphBuilder


@pytest.fixture
def sample_conversations() -> pd.DataFrame:
    """Create a minimal conversation DataFrame for testing."""
    return pd.DataFrame({
        "session_id": ["s1", "s1", "s1", "s2", "s2"],
        "interaction_id": ["i1", "i2", "i3", "i4", "i5"],
        "user_message": [
            "qual meu saldo",
            "quero ver extrato",
            "mais detalhes",
            "preciso de emprestimo",
            "qual a taxa",
        ],
        "intention_name": ["saldo", "extrato", "extrato", "emprestimo", "emprestimo"],
        "system_reply": ["r1", "r2", "r3", "r4", "r5"],
    })


@pytest.fixture
def intentions() -> list[str]:
    """Return the list of unique intentions."""
    return ["emprestimo", "extrato", "saldo"]


class TestEdge:
    """Tests for the Edge dataclass."""

    def test_default_values(self) -> None:
        edge = Edge(source=0, target=1)
        assert edge.weight == 1.0
        assert edge.is_dummy is False

    def test_custom_values(self) -> None:
        edge = Edge(source=0, target=1, weight=0.5, is_dummy=True)
        assert edge.weight == 0.5
        assert edge.is_dummy is True


class TestConversationGraph:
    """Tests for the ConversationGraph dataclass."""

    def test_total_nodes(self) -> None:
        graph = ConversationGraph(
            num_conversation_nodes=10,
            num_intention_nodes=3,
        )
        assert graph.total_nodes == 13


class TestGraphBuilder:
    """Tests for the GraphBuilder."""

    def test_build_adjacency_list(
        self, sample_conversations: pd.DataFrame, intentions: list[str]
    ) -> None:
        """Test that adjacency list is built correctly."""
        builder = GraphBuilder(sample_conversations, intentions)
        graph = builder.build_adjacency_list()

        assert graph.num_conversation_nodes == 5
        assert graph.num_intention_nodes == 3
        assert len(graph.adjacency_list) > 0

        # Each node should have at least a self-loop and an intention edge
        sources = [e.source for e in graph.adjacency_list]
        for node_idx in range(5):
            assert node_idx in sources

    def test_build_adjacency_list_with_limit(
        self, sample_conversations: pd.DataFrame, intentions: list[str]
    ) -> None:
        """Test that max_conversations limits processing."""
        builder = GraphBuilder(sample_conversations, intentions, max_conversations=3)
        graph = builder.build_adjacency_list()
        assert graph.num_conversation_nodes == 3

    def test_sequential_edges_within_session(
        self, sample_conversations: pd.DataFrame, intentions: list[str]
    ) -> None:
        """Test that sequential nodes in same session are connected."""
        builder = GraphBuilder(sample_conversations, intentions)
        graph = builder.build_adjacency_list()

        # Nodes 0→1 and 1→2 should be connected (same session s1)
        sequential_edges = [
            (e.source, e.target)
            for e in graph.adjacency_list
            if not e.is_dummy and e.source != e.target
            and e.target < graph.num_conversation_nodes
        ]
        assert (0, 1) in sequential_edges
        assert (1, 2) in sequential_edges

    def test_build_adjacency_matrix(
        self, sample_conversations: pd.DataFrame, intentions: list[str]
    ) -> None:
        """Test adjacency matrix shape and basic properties."""
        builder = GraphBuilder(sample_conversations, intentions)
        matrix = builder.build_adjacency_matrix()

        assert matrix.shape == (5, 5 + 3)  # 5 conversations + 3 intentions

    def test_add_dummy_edges(
        self, sample_conversations: pd.DataFrame, intentions: list[str]
    ) -> None:
        """Test that dummy edges are added correctly."""
        builder = GraphBuilder(sample_conversations, intentions)
        graph = builder.build_adjacency_list()

        distances = {(0, 3): 0.5, (1, 4): 2.0, (2, 3): 10.0}
        initial_edges = len(graph.adjacency_list)
        GraphBuilder.add_dummy_edges(graph, distances)

        dummy_edges = [e for e in graph.adjacency_list if e.is_dummy]
        assert len(dummy_edges) > 0
        assert len(graph.adjacency_list) > initial_edges

        # Check that weights are in [0, 1)
        for edge in dummy_edges:
            assert 0 <= edge.weight <= 1.0

    def test_add_dummy_edges_empty(
        self, sample_conversations: pd.DataFrame, intentions: list[str]
    ) -> None:
        """Test that empty distances dict is handled gracefully."""
        builder = GraphBuilder(sample_conversations, intentions)
        graph = builder.build_adjacency_list()
        initial_edges = len(graph.adjacency_list)

        GraphBuilder.add_dummy_edges(graph, {})
        assert len(graph.adjacency_list) == initial_edges
