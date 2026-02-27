# Tajalli

Tajalli is a research project that will progressively migrate its transformer implementation to use **nn-core** as a shared foundation library.

This repository also includes reference source trees:
- `.nn-core/` — shared transformer core library
- `.baseline-transformer/` — baseline project that uses nn-core
- `.tajalli_deprecated/` — legacy tajalli attempts (reference only)

## Setup (local/offline)

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
