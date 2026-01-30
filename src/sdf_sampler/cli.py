# ABOUTME: Command-line interface for sdf-sampler
# ABOUTME: Provides analyze, sample, and pipeline commands with full parameter control

"""
CLI for sdf-sampler.

Usage:
    python -m sdf_sampler analyze input.ply -o constraints.json
    python -m sdf_sampler sample input.ply constraints.json -o samples.parquet
    python -m sdf_sampler pipeline input.ply -o samples.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def add_analysis_options(parser: argparse.ArgumentParser) -> None:
    """Add analysis configuration options to a parser."""
    group = parser.add_argument_group("Analysis Options")

    group.add_argument(
        "--min-gap-size",
        type=float,
        default=0.10,
        help="Minimum gap size for flood fill in meters (default: 0.10)",
    )
    group.add_argument(
        "--max-grid-dim",
        type=int,
        default=200,
        help="Maximum voxel grid dimension (default: 200)",
    )
    group.add_argument(
        "--cone-angle",
        type=float,
        default=15.0,
        help="Ray propagation cone half-angle in degrees (default: 15.0)",
    )
    group.add_argument(
        "--normal-offset-pairs",
        type=int,
        default=40,
        help="Number of box pairs for normal_offset algorithm (default: 40)",
    )
    group.add_argument(
        "--max-boxes",
        type=int,
        default=30,
        help="Maximum boxes per algorithm (default: 30)",
    )
    group.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.5,
        help="Overlap threshold for box simplification (default: 0.5)",
    )
    group.add_argument(
        "--idw-sample-count",
        type=int,
        default=1000,
        help="Number of IDW samples to generate (default: 1000)",
    )
    group.add_argument(
        "--idw-max-distance",
        type=float,
        default=0.5,
        help="Maximum IDW distance from surface in meters (default: 0.5)",
    )
    group.add_argument(
        "--idw-power",
        type=float,
        default=2.0,
        help="IDW power factor (default: 2.0)",
    )
    group.add_argument(
        "--hull-alpha",
        type=float,
        default=1.0,
        help="Alpha shape parameter for hull filtering (default: 1.0)",
    )
    group.add_argument(
        "--flood-fill-output",
        type=str,
        choices=["boxes", "samples", "both"],
        default="samples",
        help="Output mode for flood fill (default: samples)",
    )
    group.add_argument(
        "--flood-fill-samples",
        type=int,
        default=500,
        help="Number of sample points from flood fill (default: 500)",
    )
    group.add_argument(
        "--voxel-regions-output",
        type=str,
        choices=["boxes", "samples", "both"],
        default="samples",
        help="Output mode for voxel regions (default: samples)",
    )
    group.add_argument(
        "--voxel-regions-samples",
        type=int,
        default=500,
        help="Number of sample points from voxel regions (default: 500)",
    )


def add_sampling_options(parser: argparse.ArgumentParser) -> None:
    """Add sampling configuration options to a parser."""
    group = parser.add_argument_group("Sampling Options")

    group.add_argument(
        "-n", "--total-samples",
        type=int,
        default=10000,
        help="Total number of samples to generate (default: 10000)",
    )
    group.add_argument(
        "-s", "--strategy",
        type=str,
        choices=["constant", "density", "inverse_square"],
        default="inverse_square",
        help="Sampling strategy (default: inverse_square)",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    group.add_argument(
        "--samples-per-primitive",
        type=int,
        default=100,
        help="Samples per primitive for CONSTANT strategy (default: 100)",
    )
    group.add_argument(
        "--inverse-square-base",
        type=int,
        default=100,
        help="Base samples at surface for INVERSE_SQUARE (default: 100)",
    )
    group.add_argument(
        "--inverse-square-falloff",
        type=float,
        default=2.0,
        help="Falloff exponent for INVERSE_SQUARE (default: 2.0)",
    )
    group.add_argument(
        "--near-band",
        type=float,
        default=0.02,
        help="Near-band width around surface (default: 0.02)",
    )


def add_output_options(parser: argparse.ArgumentParser) -> None:
    """Add output configuration options to a parser."""
    group = parser.add_argument_group("Output Options")

    group.add_argument(
        "--include-surface-points",
        action="store_true",
        help="Include original surface points (phi=0) in output",
    )
    group.add_argument(
        "--surface-point-count",
        type=int,
        default=1000,
        help="Number of surface points to include (default: 1000)",
    )
    group.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Mesh file for area-weighted surface sampling (PLY/OBJ/STL). "
             "If provided, surface points are sampled uniformly by surface area "
             "instead of by vertex count. Recommended for meshes with uneven vertex density.",
    )


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sdf-sampler",
        description="Auto-analysis and sampling of point clouds for SDF training data",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version and exit"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze point cloud to detect SOLID/EMPTY regions",
    )
    analyze_parser.add_argument(
        "input",
        type=Path,
        help="Input point cloud file (PLY, LAS, NPZ, CSV, Parquet)",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output constraints JSON file (default: <input>_constraints.json)",
    )
    analyze_parser.add_argument(
        "-a", "--algorithms",
        type=str,
        nargs="+",
        default=None,
        help="Algorithms to run (flood_fill, voxel_regions, normal_offset, normal_idw, pocket)",
    )
    analyze_parser.add_argument(
        "--no-hull-filter",
        action="store_true",
        help="Disable hull filtering",
    )
    analyze_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    add_analysis_options(analyze_parser)

    # sample command
    sample_parser = subparsers.add_parser(
        "sample",
        help="Generate training samples from constraints",
    )
    sample_parser.add_argument(
        "input",
        type=Path,
        help="Input point cloud file",
    )
    sample_parser.add_argument(
        "constraints",
        type=Path,
        help="Constraints JSON file (from analyze command)",
    )
    sample_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output parquet file (default: <input>_samples.parquet)",
    )
    sample_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    add_sampling_options(sample_parser)
    add_output_options(sample_parser)

    # pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Full pipeline: analyze + sample + export",
    )
    pipeline_parser.add_argument(
        "input",
        type=Path,
        help="Input point cloud file",
    )
    pipeline_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output parquet file (default: <input>_samples.parquet)",
    )
    pipeline_parser.add_argument(
        "-a", "--algorithms",
        type=str,
        nargs="+",
        default=None,
        help="Algorithms to run",
    )
    pipeline_parser.add_argument(
        "--no-hull-filter",
        action="store_true",
        help="Disable hull filtering",
    )
    pipeline_parser.add_argument(
        "--save-constraints",
        type=Path,
        default=None,
        help="Also save constraints to JSON file",
    )
    pipeline_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    add_analysis_options(pipeline_parser)
    add_sampling_options(pipeline_parser)
    add_output_options(pipeline_parser)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about a point cloud or constraints file",
    )
    info_parser.add_argument(
        "input",
        type=Path,
        help="Input file (point cloud or constraints JSON)",
    )

    args = parser.parse_args(argv)

    if args.version:
        from sdf_sampler import __version__
        print(f"sdf-sampler {__version__}")
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "sample":
        return cmd_sample(args)
    elif args.command == "pipeline":
        return cmd_pipeline(args)
    elif args.command == "info":
        return cmd_info(args)

    return 0


def build_analysis_options(args: argparse.Namespace):
    """Build AutoAnalysisOptions from CLI args."""
    from sdf_sampler.config import AutoAnalysisOptions

    return AutoAnalysisOptions(
        min_gap_size=args.min_gap_size,
        max_grid_dim=args.max_grid_dim,
        cone_angle=args.cone_angle,
        normal_offset_pairs=args.normal_offset_pairs,
        max_boxes=args.max_boxes,
        overlap_threshold=args.overlap_threshold,
        idw_sample_count=args.idw_sample_count,
        idw_max_distance=args.idw_max_distance,
        idw_power=args.idw_power,
        hull_filter_enabled=not getattr(args, 'no_hull_filter', False),
        hull_alpha=args.hull_alpha,
        flood_fill_output=args.flood_fill_output,
        flood_fill_sample_count=args.flood_fill_samples,
        voxel_regions_output=args.voxel_regions_output,
        voxel_regions_sample_count=args.voxel_regions_samples,
    )


def build_sampler_config(args: argparse.Namespace):
    """Build SamplerConfig from CLI args."""
    from sdf_sampler.config import SamplerConfig

    return SamplerConfig(
        total_samples=args.total_samples,
        samples_per_primitive=args.samples_per_primitive,
        inverse_square_base_samples=args.inverse_square_base,
        inverse_square_falloff=args.inverse_square_falloff,
        near_band=args.near_band,
        seed=args.seed or 0,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run analyze command."""
    from sdf_sampler import SDFAnalyzer, load_point_cloud

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    output = args.output or args.input.with_suffix(".constraints.json")

    if args.verbose:
        print(f"Loading point cloud: {args.input}")

    try:
        xyz, normals = load_point_cloud(str(args.input))
    except Exception as e:
        print(f"Error loading point cloud: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"  Points: {len(xyz):,}")
        print(f"  Normals: {'yes' if normals is not None else 'no'}")

    options = build_analysis_options(args)

    if args.verbose:
        algos = args.algorithms or ["all"]
        print(f"Running analysis: {', '.join(algos)}")

    analyzer = SDFAnalyzer()
    result = analyzer.analyze(
        xyz=xyz,
        normals=normals,
        algorithms=args.algorithms,
        options=options,
    )

    if args.verbose:
        print(f"Generated {len(result.constraints)} constraints")
        print(f"  SOLID: {result.summary.solid_constraints}")
        print(f"  EMPTY: {result.summary.empty_constraints}")

    # Save constraints
    with open(output, "w") as f:
        json.dump(result.constraints, f, indent=2, default=_json_serializer)

    print(f"Saved constraints to: {output}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Run sample command."""
    from sdf_sampler import SDFSampler, load_mesh, load_point_cloud

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if not args.constraints.exists():
        print(f"Error: Constraints file not found: {args.constraints}", file=sys.stderr)
        return 1

    output = args.output or args.input.with_suffix(".samples.parquet")

    if args.verbose:
        print(f"Loading point cloud: {args.input}")

    try:
        xyz, normals = load_point_cloud(str(args.input))
    except Exception as e:
        print(f"Error loading point cloud: {e}", file=sys.stderr)
        return 1

    # Load mesh for area-weighted surface sampling if provided
    mesh = None
    if args.mesh:
        if args.verbose:
            print(f"Loading mesh for area-weighted sampling: {args.mesh}")
        try:
            mesh = load_mesh(str(args.mesh))
            if args.verbose:
                print(f"  Vertices: {len(mesh.vertices):,}, Faces: {len(mesh.faces):,}")
        except Exception as e:
            print(f"Error loading mesh: {e}", file=sys.stderr)
            return 1

    if args.verbose:
        print(f"Loading constraints: {args.constraints}")

    with open(args.constraints) as f:
        constraints = json.load(f)

    if args.verbose:
        print(f"  Constraints: {len(constraints)}")
        print(f"Generating {args.total_samples:,} samples with strategy: {args.strategy}")
        if args.include_surface_points:
            mode = "area-weighted" if mesh else "vertex-based"
            print(f"  Including {args.surface_point_count:,} surface points ({mode})")

    config = build_sampler_config(args)
    sampler = SDFSampler(config=config)
    samples = sampler.generate(
        xyz=xyz,
        normals=normals,
        constraints=constraints,
        total_samples=args.total_samples,
        strategy=args.strategy,
        seed=args.seed,
        include_surface_points=args.include_surface_points,
        surface_point_count=args.surface_point_count,
        mesh=mesh,
    )

    if args.verbose:
        print(f"Generated {len(samples)} samples")

    sampler.export_parquet(samples, str(output))
    print(f"Saved samples to: {output}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run full pipeline: analyze + sample + export."""
    from sdf_sampler import SDFAnalyzer, SDFSampler, load_mesh, load_point_cloud

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    output = args.output or args.input.with_suffix(".samples.parquet")

    if args.verbose:
        print(f"Loading point cloud: {args.input}")

    try:
        xyz, normals = load_point_cloud(str(args.input))
    except Exception as e:
        print(f"Error loading point cloud: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"  Points: {len(xyz):,}")
        print(f"  Normals: {'yes' if normals is not None else 'no'}")

    # Load mesh for area-weighted surface sampling if provided
    mesh = None
    if args.mesh:
        if args.verbose:
            print(f"Loading mesh for area-weighted sampling: {args.mesh}")
        try:
            mesh = load_mesh(str(args.mesh))
            if args.verbose:
                print(f"  Vertices: {len(mesh.vertices):,}, Faces: {len(mesh.faces):,}")
        except Exception as e:
            print(f"Error loading mesh: {e}", file=sys.stderr)
            return 1

    # Analyze
    if args.verbose:
        algos = args.algorithms or ["all"]
        print(f"Running analysis: {', '.join(algos)}")

    options = build_analysis_options(args)
    analyzer = SDFAnalyzer()
    result = analyzer.analyze(
        xyz=xyz,
        normals=normals,
        algorithms=args.algorithms,
        options=options,
    )

    if args.verbose:
        print(f"Generated {len(result.constraints)} constraints")
        print(f"  SOLID: {result.summary.solid_constraints}")
        print(f"  EMPTY: {result.summary.empty_constraints}")

    # Optionally save constraints
    if args.save_constraints:
        with open(args.save_constraints, "w") as f:
            json.dump(result.constraints, f, indent=2, default=_json_serializer)
        if args.verbose:
            print(f"Saved constraints to: {args.save_constraints}")

    # Sample
    if args.verbose:
        print(f"Generating {args.total_samples:,} samples with strategy: {args.strategy}")
        if args.include_surface_points:
            mode = "area-weighted" if mesh else "vertex-based"
            print(f"  Including {args.surface_point_count:,} surface points ({mode})")

    config = build_sampler_config(args)
    sampler = SDFSampler(config=config)
    samples = sampler.generate(
        xyz=xyz,
        normals=normals,
        constraints=result.constraints,
        total_samples=args.total_samples,
        strategy=args.strategy,
        seed=args.seed,
        include_surface_points=args.include_surface_points,
        surface_point_count=args.surface_point_count,
        mesh=mesh,
    )

    if args.verbose:
        print(f"Generated {len(samples)} samples")

    # Export
    sampler.export_parquet(samples, str(output))
    print(f"Saved samples to: {output}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about a file."""
    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1

    suffix = args.input.suffix.lower()

    if suffix == ".json":
        # Constraints file
        with open(args.input) as f:
            constraints = json.load(f)

        print(f"Constraints file: {args.input}")
        print(f"  Total constraints: {len(constraints)}")

        # Count by type and sign
        by_type: dict[str, int] = {}
        by_sign: dict[str, int] = {}
        for c in constraints:
            ctype = c.get("type", "unknown")
            sign = c.get("sign", "unknown")
            by_type[ctype] = by_type.get(ctype, 0) + 1
            by_sign[sign] = by_sign.get(sign, 0) + 1

        print("  By type:")
        for t, count in sorted(by_type.items()):
            print(f"    {t}: {count}")
        print("  By sign:")
        for s, count in sorted(by_sign.items()):
            print(f"    {s}: {count}")

    elif suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(args.input)

        print(f"Parquet file: {args.input}")
        print(f"  Samples: {len(df):,}")
        print(f"  Columns: {', '.join(df.columns)}")

        if "source" in df.columns:
            print("  By source:")
            for source, count in df["source"].value_counts().items():
                print(f"    {source}: {count:,}")

        if "phi" in df.columns:
            print(f"  Phi range: [{df['phi'].min():.4f}, {df['phi'].max():.4f}]")

    else:
        # Point cloud file
        from sdf_sampler import load_point_cloud

        try:
            xyz, normals = load_point_cloud(str(args.input))
        except Exception as e:
            print(f"Error loading file: {e}", file=sys.stderr)
            return 1

        print(f"Point cloud: {args.input}")
        print(f"  Points: {len(xyz):,}")
        print(f"  Normals: {'yes' if normals is not None else 'no'}")
        print(f"  Bounds:")
        print(f"    X: [{xyz[:, 0].min():.4f}, {xyz[:, 0].max():.4f}]")
        print(f"    Y: [{xyz[:, 1].min():.4f}, {xyz[:, 1].max():.4f}]")
        print(f"    Z: [{xyz[:, 2].min():.4f}, {xyz[:, 2].max():.4f}]")

    return 0


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    sys.exit(main())
