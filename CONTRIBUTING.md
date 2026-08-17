# Contributing to viva-openmm

## Development setup

    uv venv .venv && source .venv/bin/activate
    uv pip install -e ".[dev]"
    uv pip install "git+https://github.com/maccallumlab/martini_openmm.git"
    pytest
