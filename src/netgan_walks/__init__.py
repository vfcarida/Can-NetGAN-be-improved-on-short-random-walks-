"""
NetGAN Walks — Improving NetGAN Graph Generation via Dense-Vertex-Initialized Random Walks.

This package implements the methodology described in the paper
"Can NetGAN be improved on short random walks?" (arXiv:1905.05298).
It provides tools for building conversation graphs, computing semantic
similarities, and calculating random walk distances for chatbot
conversation optimization.

Modules:
    data     — Conversation data loading and preprocessing.
    graph    — Graph construction (adjacency list/matrix) and reduction.
    similarity — Word2Vec-based semantic distance computation.
    walks    — Random walk distance calculations.
"""

__version__ = "1.0.0"
__author__ = "Vinicius Fernandes Caridá"
__paper__ = "https://arxiv.org/abs/1905.05298"
