# Contributing to NetGAN Walks

Thank you for your interest in contributing! This document provides guidelines
and best practices for contributing to this project.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/Can-NetGAN-be-improved-on-short-random-walks-.git
   cd Can-NetGAN-be-improved-on-short-random-walks-
   ```
3. **Install** in development mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e ".[dev]"
   ```

## 📋 Development Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```
2. Make your changes following the code standards below
3. Run the test suite:
   ```bash
   pytest
   ```
4. Commit with clear, descriptive messages:
   ```bash
   git commit -m "feat: add betweenness centrality to dense vertex selection"
   ```
5. Push and open a Pull Request

## 🎨 Code Standards

### Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Maximum line length: **100 characters**
- Use [ruff](https://github.com/astral-sh/ruff) for linting:
  ```bash
  ruff check src/ tests/
  ```

### Type Hints

- All public functions **must** have type annotations
- Use `from __future__ import annotations` for modern syntax
- Validate with mypy:
  ```bash
  mypy src/netgan_walks/
  ```

### Docstrings

- Use Google-style docstrings for all public classes and functions
- Include `Args`, `Returns`, and `Raises` sections
- Add usage examples for complex APIs

### Testing

- Write tests for all new functionality
- Place tests in `tests/` mirroring the `src/` structure
- Use descriptive test names: `test_reduce_merges_similar_nodes`
- Aim for **≥ 80% coverage**

## 🏷️ Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Description |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or updating tests |
| `refactor:` | Code refactoring (no behavior change) |
| `perf:` | Performance improvement |
| `chore:` | Maintenance tasks |

## 🐛 Reporting Issues

When reporting bugs, please include:

- Python version (`python --version`)
- Operating system
- Steps to reproduce the issue
- Expected vs. actual behavior
- Error traceback (if applicable)

## 📄 License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
