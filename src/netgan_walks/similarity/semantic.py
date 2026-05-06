"""
Semantic similarity computation using Word2Vec and Word Mover's Distance.

Replaces the original notebook's threading-based distance computation
with a safe concurrent.futures approach, eliminating race conditions.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from gensim.models import Word2Vec

logger = logging.getLogger(__name__)


class SemanticSimilarity:
    """Compute semantic distances between conversation messages.

    Uses Word2Vec embeddings trained on conversation text combined with
    Word Mover's Distance (WMD) to quantify semantic similarity.

    Args:
        min_count: Minimum word frequency for Word2Vec training.
        vector_size: Dimensionality of word vectors.
        window: Context window size for Word2Vec.
        workers: Number of threads for Word2Vec training.

    Example::

        sim = SemanticSimilarity()
        sim.train_word2vec(tokenized_sentences)
        distances = sim.compute_wmd_distances(messages[:100])
    """

    def __init__(
        self,
        min_count: int = 1,
        vector_size: int = 100,
        window: int = 5,
        workers: int = 4,
    ):
        self.min_count = min_count
        self.vector_size = vector_size
        self.window = window
        self.workers = workers
        self.model: Optional[Word2Vec] = None

    def train_word2vec(self, sentences: list[list[str]]) -> Word2Vec:
        """Train a Word2Vec model on tokenized conversation sentences.

        Args:
            sentences: List of tokenized word lists.

        Returns:
            The trained Word2Vec model.
        """
        logger.info("Training Word2Vec on %d sentences", len(sentences))
        self.model = Word2Vec(
            sentences=sentences,
            min_count=self.min_count,
            vector_size=self.vector_size,
            window=self.window,
            workers=self.workers,
        )
        logger.info("Word2Vec training complete — vocabulary size: %d", len(self.model.wv))
        return self.model

    def compute_wmd_distances(
        self,
        messages: list[str],
        max_workers: int = 8,
        checkpoint_path: Optional[str | Path] = None,
        checkpoint_interval: int = 10_000_000,
    ) -> dict[tuple[int, int], float]:
        """Compute pairwise WMD distances between all messages.

        Uses concurrent.futures.ThreadPoolExecutor instead of raw
        threading to avoid the race conditions in the original code.

        Args:
            messages: List of raw message strings.
            max_workers: Number of parallel workers.
            checkpoint_path: Optional file path to periodically save results.
            checkpoint_interval: Number of distances between checkpoints.

        Returns:
            Dictionary mapping (i, j) node pairs to their WMD distance.

        Raises:
            RuntimeError: If Word2Vec model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("No Word2Vec model. Call train_word2vec() first.")

        num_messages = len(messages)
        logger.info("Computing WMD distances for %d messages (%d pairs)",
                     num_messages, num_messages * (num_messages - 1) // 2)

        distances: dict[tuple[int, int], float] = {}

        def _compute_single(i: int, j: int) -> tuple[int, int, float]:
            dist = self.model.wv.wmdistance(
                messages[i].lower().split(),
                messages[j].lower().split(),
            )
            return i, j, dist

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_single, i, j): (i, j)
                for i in range(num_messages)
                for j in range(i + 1, num_messages)
            }

            completed = 0
            for future in as_completed(futures):
                i, j, dist = future.result()
                distances[(i, j)] = dist
                completed += 1

                if checkpoint_path and completed % checkpoint_interval == 0:
                    self.save_distances(distances, checkpoint_path)
                    logger.info("Checkpoint saved at %d distances", completed)

        logger.info("WMD computation complete — %d distances", len(distances))
        return distances

    @staticmethod
    def save_distances(
        distances: dict[tuple[int, int], float], filepath: str | Path
    ) -> None:
        """Save computed distances to a text file.

        Args:
            distances: Dictionary of (i, j) → distance.
            filepath: Output file path.
        """
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            for (i, j), dist in distances.items():
                f.write(f"({i}, {j}, {dist})\n")
        logger.info("Saved %d distances to %s", len(distances), filepath)

    @staticmethod
    def load_distances(filepath: str | Path) -> dict[tuple[int, int], float]:
        """Load pre-computed distances from a text file.

        Args:
            filepath: Path to the distances file.

        Returns:
            Dictionary of (i, j) → distance.
        """
        filepath = Path(filepath)
        distances: dict[tuple[int, int], float] = {}

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().strip("()")
                parts = line.split(",")
                if len(parts) == 3:
                    i, j, dist = int(parts[0]), int(parts[1]), float(parts[2])
                    distances[(i, j)] = dist

        logger.info("Loaded %d distances from %s", len(distances), filepath)
        return distances

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        """Find the most similar words to a given word.

        Args:
            word: The query word.
            topn: Number of similar words to return.

        Returns:
            List of (word, similarity_score) tuples.

        Raises:
            RuntimeError: If Word2Vec model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("No Word2Vec model. Call train_word2vec() first.")
        return self.model.wv.most_similar(word, topn=topn)
