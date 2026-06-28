# Contributing to Foundry

Thank you for your interest in contributing! Foundry is an open-source project and
we welcome contributions at all levels.

## Getting Started

```bash
git clone https://github.com/[org]/agent-foundry
cd agent-foundry
pip install -e ".[dev]"
```

## Development Workflow

```bash
# Run tests
pytest tests/unit/

# Lint + type check
ruff check src/
mypy src/

# Run full suite
pytest
```

## What to Work On

- 🐛 **Bug fixes** — check [open issues](https://github.com/[org]/agent-foundry/issues)
- 🔧 **Good first issues** — labeled [`good-first-issue`](https://github.com/[org]/agent-foundry/issues?q=label%3Agood-first-issue)
- 📦 **New fine-tune backends** — implement `FineTuneBackend` protocol
- 🌍 **New environment connectors** — implement `EnvironmentProtocol`
- 📚 **Documentation + examples** — always welcome

## Pull Request Guidelines

1. One feature or fix per PR
2. Tests required for new functionality
3. Run `ruff check` and `mypy` before submitting
4. Use [conventional commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `docs:`, `test:`, `chore:`

## Code of Conduct

Be kind. Be constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
