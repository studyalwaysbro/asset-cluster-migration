# Contributing to Asset Cluster Migration

Thank you for your interest in contributing! This project is an open academic research effort.

## Authors

- **Nicholas Tavares** - Lead Researcher
- **Amjad Hanini** - Contributor
- **Brandon DaSilva** - Contributor

## How to Contribute

### 1. Fork & Branch
```bash
git fork https://github.com/YOUR_USERNAME/asset-cluster-migration.git
git checkout -b feature/your-feature-name
```

### 2. Development Setup
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -e ".[dev]"
```

### 3. Areas for Contribution

**High Priority:**
- Walk-forward backtesting framework (Phase 4.4 in roadmap)
- Streamlit/Dash real-time dashboard
- Bootstrap confidence intervals for transfer entropy
- Sensitivity analysis (window size, threshold, resolution parameters)

**Research Extensions:**
- Alternative clustering algorithms (Infomap, Stochastic Block Model)
- Higher-frequency data analysis (intraday)
- NLP-based geopolitical event layer
- Formal theoretical proofs for topology crystallization

**Infrastructure:**
- Unit tests for all metric functions
- CI/CD pipeline (GitHub Actions)
- Docker containerization
- Documentation improvements

### 4. Code Style
- Python 3.10+
- Type hints on all public functions
- Docstrings (NumPy style)
- Black formatting, Ruff linting

### 5. Pull Request Process
1. Create a feature branch from `main`
2. Add tests for new functionality
3. Update README.md if adding new features
4. Submit PR with clear description of changes
5. At least one author must review before merge

### 6. Data & API Keys
- Never commit API keys or `.env` files
- Raw data is gitignored (`data/raw/`)
- Processed data can be committed if small (<10MB per file)
- Large artifacts should be regenerable from the pipeline

## Disclaimer

All contributions must maintain the educational/research nature of this project. Do not add content that could be construed as investment advice or recommendations.

## Questions?

Open an issue or reach out to the authors.
