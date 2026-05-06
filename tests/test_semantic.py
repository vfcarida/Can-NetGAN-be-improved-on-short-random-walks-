"""Tests for the SemanticSimilarity module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from netgan_walks.similarity.semantic import SemanticSimilarity


@pytest.fixture
def sample_sentences() -> list[list[str]]:
    """Create sample tokenized sentences for Word2Vec training."""
    return [
        ["qual", "meu", "saldo", "bancario"],
        ["quero", "ver", "meu", "extrato"],
        ["preciso", "de", "um", "emprestimo"],
        ["qual", "a", "taxa", "de", "juros"],
        ["como", "investir", "meu", "dinheiro"],
        ["quero", "fazer", "um", "pix"],
        ["meu", "cartao", "foi", "bloqueado"],
        ["como", "desbloquear", "meu", "cartao"],
        ["qual", "o", "limite", "do", "meu", "cartao"],
        ["preciso", "de", "ajuda", "com", "meu", "saldo"],
    ]


@pytest.fixture
def trained_similarity(sample_sentences: list[list[str]]) -> SemanticSimilarity:
    """Return a SemanticSimilarity instance with a trained model."""
    sim = SemanticSimilarity(min_count=1, vector_size=50)
    sim.train_word2vec(sample_sentences)
    return sim


class TestSemanticSimilarity:
    """Tests for SemanticSimilarity."""

    def test_train_word2vec(self, sample_sentences: list[list[str]]) -> None:
        """Test that Word2Vec model trains successfully."""
        sim = SemanticSimilarity(min_count=1, vector_size=50)
        model = sim.train_word2vec(sample_sentences)

        assert model is not None
        assert sim.model is not None
        assert len(sim.model.wv) > 0

    def test_most_similar_without_model(self) -> None:
        """Test that most_similar raises without training."""
        sim = SemanticSimilarity()
        with pytest.raises(RuntimeError, match="No Word2Vec model"):
            sim.most_similar("saldo")

    def test_most_similar(self, trained_similarity: SemanticSimilarity) -> None:
        """Test that most_similar returns results."""
        results = trained_similarity.most_similar("saldo", topn=3)
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_compute_wmd_without_model(self) -> None:
        """Test that WMD computation raises without training."""
        sim = SemanticSimilarity()
        with pytest.raises(RuntimeError):
            sim.compute_wmd_distances(["test message"])

    def test_save_and_load_distances(self, tmp_path: Path) -> None:
        """Test distance serialization roundtrip."""
        distances = {(0, 1): 0.5, (0, 2): 1.3, (1, 2): 0.8}
        filepath = tmp_path / "distances.txt"

        SemanticSimilarity.save_distances(distances, filepath)
        loaded = SemanticSimilarity.load_distances(filepath)

        assert len(loaded) == 3
        for key, value in distances.items():
            assert key in loaded
            assert abs(loaded[key] - value) < 1e-6
