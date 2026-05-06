"""Tests for the ConversationLoader module."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from netgan_walks.data.loader import ConversationLoader


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a minimal synthetic conversation CSV for testing."""
    filepath = tmp_path / "test_conversations.csv"
    rows = [
        ["session_id", "interaction_id", "user_message", "intention_name", "system_reply"],
        ["s1", "i1", "qual meu saldo", "saldo", "Seu saldo é R$100"],
        ["s1", "i2", "quero ver extrato", "extrato", "Aqui está seu extrato"],
        ["s1", "i3", "obrigado", "boas vindas", "De nada!"],
        ["s2", "i4", "preciso de emprestimo", "emprestimo", "Veja nossas opções"],
        ["s2", "i5", "qual a taxa", "emprestimo", "A taxa é 1.5%"],
        ["s3", "i6", "oi", "boas vindas", "Olá! Como posso ajudar?"],
        ["s3", "i7", "quero investir", "investimento", "Temos várias opções"],
    ]
    with open(filepath, "w", newline="", encoding="ISO-8859-1") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(rows)
    return filepath


class TestConversationLoader:
    """Tests for ConversationLoader."""

    def test_load_success(self, sample_csv: Path) -> None:
        """Test that a valid CSV loads correctly."""
        loader = ConversationLoader(sample_csv)
        df = loader.load()
        assert len(df) == 7
        assert "session_id" in df.columns

    def test_load_file_not_found(self) -> None:
        """Test that a missing file raises FileNotFoundError."""
        loader = ConversationLoader("nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_missing_columns(self, tmp_path: Path) -> None:
        """Test that a CSV with missing columns raises ValueError."""
        filepath = tmp_path / "bad.csv"
        with open(filepath, "w", newline="", encoding="ISO-8859-1") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerows([["col_a", "col_b"], ["x", "y"]])

        loader = ConversationLoader(filepath)
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load()

    def test_clean_removes_greetings(self, sample_csv: Path) -> None:
        """Test that 'boas vindas' intents are filtered out."""
        loader = ConversationLoader(sample_csv)
        loader.load()
        df = loader.clean()
        assert "boas vindas" not in df["intention_name"].values

    def test_clean_removes_duplicates(self, sample_csv: Path) -> None:
        """Test that duplicate interaction_ids are removed."""
        loader = ConversationLoader(sample_csv)
        loader.load()
        df = loader.clean()
        assert df["interaction_id"].is_unique

    def test_clean_without_load_raises(self) -> None:
        """Test that clean() before load() raises RuntimeError."""
        loader = ConversationLoader("dummy.csv")
        with pytest.raises(RuntimeError, match="No data loaded"):
            loader.clean()

    def test_group_sessions(self, sample_csv: Path) -> None:
        """Test that sessions are correctly grouped."""
        loader = ConversationLoader(sample_csv)
        loader.load()
        loader.clean()
        sessions = loader.group_sessions()
        assert len(sessions) >= 1
        assert all(isinstance(s, list) for s in sessions)

    def test_group_sessions_without_clean_raises(self, sample_csv: Path) -> None:
        """Test that group_sessions() before clean() raises RuntimeError."""
        loader = ConversationLoader(sample_csv)
        loader.load()
        with pytest.raises(RuntimeError, match="No cleaned data"):
            loader.group_sessions()

    def test_get_unique_intentions(self, sample_csv: Path) -> None:
        """Test that unique intentions are returned sorted."""
        loader = ConversationLoader(sample_csv)
        loader.load()
        loader.clean()
        intentions = loader.get_unique_intentions()
        assert isinstance(intentions, list)
        assert intentions == sorted(intentions)
        assert "boas vindas" not in intentions

    def test_tokenize_messages(self) -> None:
        """Test that messages are tokenized correctly."""
        messages = ["Qual é o saldo?", "Preciso de ajuda!"]
        tokens = ConversationLoader.tokenize_messages(messages)
        assert len(tokens) == 2
        assert all(isinstance(t, list) for t in tokens)
        # Punctuation should be removed
        for token_list in tokens:
            for word in token_list:
                assert word == word.lower()
                assert "?" not in word
                assert "!" not in word
