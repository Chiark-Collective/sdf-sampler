# ABOUTME: Entry point for running sdf-sampler as a module
# ABOUTME: Enables `python -m sdf_sampler` invocation

"""
Run sdf-sampler as a module.

Usage:
    python -m sdf_sampler --help
    python -m sdf_sampler analyze input.ply -o constraints.json
    python -m sdf_sampler sample input.ply constraints.json -o samples.parquet
    python -m sdf_sampler pipeline input.ply -o samples.parquet
"""

from sdf_sampler.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
