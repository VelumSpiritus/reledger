"""Thin entry point — the cascade runs from the deepest level up.

Usage:
    python run.py <data_dir> [--temporal]
    python run.py --download <url>

CC0 1.0 Universal -- public domain.
"""

from cascade.ingest import main

if __name__ == "__main__":
    main()
