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

## Command-Line Interface

sdf-sampler provides a CLI for common workflows:

```bash
# Run as module
python -m sdf_sampler --help

# Or use the installed command
sdf-sampler --help
```

### Commands

#### `pipeline` - Full workflow (recommended)

Run the complete pipeline: analyze point cloud → generate samples → export.

```bash
# Basic usage
sdf-sampler pipeline scan.ply -o training_data.parquet

# With options
sdf-sampler pipeline scan.ply \
    -o training_data.parquet \
    -n 50000 \
    -s inverse_square \
    --save-constraints constraints.json \
    -v
```

Options:
- `-o, --output`: Output parquet file (default: `<input>_samples.parquet`)
- `-n, --total-samples`: Number of samples to generate (default: 10000)
- `-s, --strategy`: Sampling strategy: `constant`, `density`, `inverse_square` (default: `inverse_square`)
- `-a, --algorithms`: Specific algorithms to run (default: all)
- `--save-constraints`: Also save constraints to JSON
- `--seed`: Random seed for reproducibility
- `-v, --verbose`: Verbose output

#### `analyze` - Detect regions

Analyze a point cloud to detect SOLID/EMPTY regions.

```bash
sdf-sampler analyze scan.ply -o constraints.json -v
```

Options:
- `-o, --output`: Output JSON file (default: `<input>_constraints.json`)
- `-a, --algorithms`: Algorithms to run (see below)
- `--no-hull-filter`: Disable hull filtering
- `-v, --verbose`: Verbose output

#### `sample` - Generate training samples

Generate training samples from a constraints file.

```bash
sdf-sampler sample scan.ply constraints.json -o samples.parquet -n 50000
```

Options:
- `-o, --output`: Output parquet file
- `-n, --total-samples`: Number of samples (default: 10000)
- `-s, --strategy`: Sampling strategy (default: `inverse_square`)
- `--seed`: Random seed
- `-v, --verbose`: Verbose output

#### `info` - Inspect files

Show information about point clouds, constraints, or sample files.

```bash
sdf-sampler info scan.ply
sdf-sampler info constraints.json
sdf-sampler info samples.parquet
```

## Python SDK

### Quick Start

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

### SDFAnalyzer

Analyzes point clouds to detect SOLID and EMPTY regions.

```python
from sdf_sampler import SDFAnalyzer
from sdf_sampler.config import AnalyzerConfig, AutoAnalysisOptions

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

#### Analysis Algorithms

| Algorithm | Description | Output |
|-----------|-------------|--------|
| `flood_fill` | Detects EMPTY (outside) regions by ray propagation from sky | Box or SamplePoint constraints |
| `voxel_regions` | Detects SOLID (underground) regions | Box or SamplePoint constraints |
| `normal_offset` | Generates paired SOLID/EMPTY boxes along surface normals | Box constraints |
| `normal_idw` | Inverse distance weighted sampling along normals | SamplePoint constraints |
| `pocket` | Detects interior cavities | Pocket constraints |

### SDFSampler

Generates training samples from constraints.

```python
from sdf_sampler import SDFSampler
from sdf_sampler.config import SamplerConfig

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

#### Sampling Strategies

| Strategy | Description |
|----------|-------------|
| `constant` | Fixed number of samples per constraint |
| `density` | Samples proportional to constraint volume |
| `inverse_square` | More samples near surface, fewer far away (recommended) |

### Constraint Types

The analyzer generates various constraint types:

- **BoxConstraint**: Axis-aligned bounding box
- **SphereConstraint**: Spherical region
- **SamplePointConstraint**: Direct point with signed distance
- **PocketConstraint**: Detected cavity region

Each constraint has:
- `sign`: "solid" (negative SDF) or "empty" (positive SDF)
- `weight`: Sample weight (default 1.0)

### I/O Helpers

```python
from sdf_sampler import load_point_cloud, export_parquet

# Load various formats
xyz, normals = load_point_cloud("scan.ply")    # PLY (requires trimesh)
xyz, normals = load_point_cloud("scan.las")    # LAS/LAZ (requires laspy)
xyz, normals = load_point_cloud("scan.csv")    # CSV with x,y,z columns
xyz, normals = load_point_cloud("scan.npz")    # NumPy archive
xyz, normals = load_point_cloud("scan.parquet") # Parquet

# Export samples
export_parquet(samples, "output.parquet")
```

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

## Configuration Reference

### AnalyzerConfig

| Option | Default | Description |
|--------|---------|-------------|
| `min_gap_size` | 0.10 | Minimum gap size for flood fill (meters) |
| `max_grid_dim` | 200 | Maximum voxel grid dimension |
| `cone_angle` | 15.0 | Ray propagation cone half-angle (degrees) |
| `normal_offset_pairs` | 40 | Number of box pairs for normal_offset |
| `idw_sample_count` | 1000 | Total IDW samples |
| `idw_max_distance` | 0.5 | Maximum IDW distance (meters) |
| `hull_filter_enabled` | True | Filter outside X-Y alpha shape |
| `hull_alpha` | 1.0 | Alpha shape parameter |

### SamplerConfig

| Option | Default | Description |
|--------|---------|-------------|
| `total_samples` | 10000 | Default total samples |
| `samples_per_primitive` | 100 | Samples per constraint (CONSTANT) |
| `samples_per_cubic_meter` | 10000 | Sample density (DENSITY) |
| `inverse_square_base_samples` | 100 | Base samples (INVERSE_SQUARE) |
| `inverse_square_falloff` | 2.0 | Falloff exponent |
| `near_band` | 0.02 | Near-band width |
| `seed` | 0 | Random seed |

## Integration with Ubik

sdf-sampler is the core analysis engine for [Ubik](https://github.com/Chiark-Collective/ubik), an interactive web application for SDF labeling. Use sdf-sampler directly for:

- Automated batch processing pipelines
- Integration into ML training workflows
- Custom analysis scripts

## License

MIT
