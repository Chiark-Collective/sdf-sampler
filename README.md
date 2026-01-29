# sdf-sampler

Auto-analysis and sampling of point clouds for SDF (Signed Distance Field) training data generation.

A lightweight, standalone Python package for generating SDF training hints from point clouds. Automatically detects SOLID (inside) and EMPTY (outside) regions and generates training samples suitable for SDF regression models.

## Installation

```bash
pip install sdf-sampler
```

For additional I/O format support (PLY, LAS/LAZ):

```bash
pip install sdf-sampler[io]
```

## Quick Start

```python
from sdf_sampler import SDFAnalyzer, SDFSampler, load_point_cloud

# 1. Load point cloud (supports PLY, LAS, CSV, NPZ, Parquet)
xyz, normals = load_point_cloud("scan.ply")

# 2. Auto-analyze to detect EMPTY/SOLID regions
analyzer = SDFAnalyzer()
result = analyzer.analyze(xyz=xyz, normals=normals)
print(f"Generated {len(result.constraints)} constraints")

# 3. Generate training samples
sampler = SDFSampler()
samples = sampler.generate(
    xyz=xyz,
    constraints=result.constraints,
    strategy="inverse_square",
    total_samples=50000,
)

# 4. Export to parquet
sampler.export_parquet(samples, "training_data.parquet")
```

## Features

### Auto-Analysis Algorithms

- **flood_fill**: Detects EMPTY (outside) regions by ray propagation from sky
- **voxel_regions**: Detects SOLID (underground) regions
- **normal_offset**: Generates paired SOLID/EMPTY boxes along surface normals
- **normal_idw**: Inverse distance weighted sampling along normals
- **pocket**: Detects interior cavities

### Sampling Strategies

- **CONSTANT**: Fixed number of samples per constraint
- **DENSITY**: Samples proportional to constraint volume
- **INVERSE_SQUARE**: More samples near surface, fewer far away (recommended)

## API Reference

### SDFAnalyzer

```python
from sdf_sampler import SDFAnalyzer, AnalyzerConfig

# With default config
analyzer = SDFAnalyzer()

# With custom config
analyzer = SDFAnalyzer(config=AnalyzerConfig(
    min_gap_size=0.10,      # Minimum gap for flood fill
    max_grid_dim=200,       # Maximum voxel grid dimension
    cone_angle=15.0,        # Ray propagation cone angle
    hull_filter_enabled=True,  # Filter outside X-Y hull
))

# Run analysis
result = analyzer.analyze(
    xyz=xyz,                    # (N, 3) point positions
    normals=normals,            # (N, 3) point normals (optional)
    algorithms=["flood_fill", "voxel_regions"],  # Which algorithms to run
)

# Access results
print(f"Total constraints: {result.summary.total_constraints}")
print(f"SOLID: {result.summary.solid_constraints}")
print(f"EMPTY: {result.summary.empty_constraints}")

# Get constraint dicts for sampling
constraints = result.constraints
```

### SDFSampler

```python
from sdf_sampler import SDFSampler, SamplerConfig

# With default config
sampler = SDFSampler()

# With custom config
sampler = SDFSampler(config=SamplerConfig(
    total_samples=10000,
    inverse_square_base_samples=100,
    inverse_square_falloff=2.0,
    near_band=0.02,
))

# Generate samples
samples = sampler.generate(
    xyz=xyz,                     # Point cloud for distance computation
    constraints=constraints,      # From analyzer.analyze().constraints
    strategy="inverse_square",    # Sampling strategy
    seed=42,                      # For reproducibility
)

# Export
sampler.export_parquet(samples, "output.parquet")

# Or get DataFrame
df = sampler.to_dataframe(samples)
```

### Constraint Types

The analyzer generates various constraint types:

- **BoxConstraint**: Axis-aligned bounding box
- **SphereConstraint**: Spherical region
- **SamplePointConstraint**: Direct point with signed distance
- **PocketConstraint**: Detected cavity region

Each constraint has:
- `sign`: "solid" (negative SDF) or "empty" (positive SDF)
- `weight`: Sample weight (default 1.0)

## Output Format

The exported parquet file contains columns:

| Column | Type | Description |
|--------|------|-------------|
| x, y, z | float | 3D position |
| phi | float | Signed distance (negative=solid, positive=empty) |
| nx, ny, nz | float | Normal vector (if available) |
| weight | float | Sample weight |
| source | string | Sample origin (e.g., "box_solid", "flood_fill_empty") |
| is_surface | bool | Whether sample is on surface |
| is_free | bool | Whether sample is in free space (EMPTY) |

## Configuration Options

### AnalyzerConfig

| Option | Default | Description |
|--------|---------|-------------|
| min_gap_size | 0.10 | Minimum gap size for flood fill (meters) |
| max_grid_dim | 200 | Maximum voxel grid dimension |
| cone_angle | 15.0 | Ray propagation cone half-angle (degrees) |
| normal_offset_pairs | 40 | Number of box pairs for normal_offset |
| idw_sample_count | 1000 | Total IDW samples |
| idw_max_distance | 0.5 | Maximum IDW distance (meters) |
| hull_filter_enabled | True | Filter outside X-Y alpha shape |
| hull_alpha | 1.0 | Alpha shape parameter |

### SamplerConfig

| Option | Default | Description |
|--------|---------|-------------|
| total_samples | 10000 | Default total samples |
| samples_per_primitive | 100 | Samples per constraint (CONSTANT) |
| samples_per_cubic_meter | 10000 | Sample density (DENSITY) |
| inverse_square_base_samples | 100 | Base samples (INVERSE_SQUARE) |
| inverse_square_falloff | 2.0 | Falloff exponent |
| near_band | 0.02 | Near-band width |
| seed | 0 | Random seed |

## License

MIT
