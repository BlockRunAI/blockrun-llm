# Contributing to blockrun-llm

## Setup

```bash
git clone https://github.com/BlockRunAI/blockrun-llm
cd blockrun-llm
pip install -e ".[dev,solana]"
```

## Development

```bash
pytest                            # Run tests
black blockrun_llm/               # Format
ruff check blockrun_llm/          # Lint
mypy blockrun_llm/                # Type check
```

## Code Standards

- Python >= 3.9
- Black formatting (line-length 100)
- Ruff linting (line-length 100)
- mypy strict mode
- All tests must pass

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Run `pytest`, `black`, `ruff`, `mypy`
4. Submit PR with clear description

## License

MIT
