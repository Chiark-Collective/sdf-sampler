# Changelog

All notable changes to sdf-sampler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-01-30

### Added

- **Area-weighted surface sampling** - New sampling mode that distributes surface points uniformly by surface area instead of by vertex count. Essential for meshes with uneven vertex density (e.g., trench floors vs walls).
  - New `load_mesh()` function returns `Mesh` object with vertices, faces, and normals
  - Pass `mesh=` parameter to `sampler.generate()` for area-weighted sampling
  - CLI: `--mesh path/to/mesh.obj` enables area-weighted mode
  - Supports PLY, OBJ, STL, OFF mesh formats
- **OBJ/STL/OFF file support** - `load_point_cloud()` now supports additional mesh formats via trimesh

### Example

```python
from sdf_sampler import SDFSampler, load_mesh, load_point_cloud

# Load mesh for area-weighted sampling
mesh = load_mesh("model.obj")
xyz, normals = load_point_cloud("model.obj")

# Area-weighted gives uniform coverage by surface area
samples = sampler.generate(
    xyz=xyz, constraints=constraints,
    include_surface_points=True,
    surface_point_count=1000,
    mesh=mesh,  # Enables area-weighted sampling
)
```

## [0.4.0] - 2025-01-30

### Changed

- **Default algorithms no longer include `normal_idw`** - The `normal_idw` algorithm is now opt-in only. Default algorithms are: `flood_fill`, `voxel_regions`, `normal_offset`. To use `normal_idw`, explicitly pass `algorithms=["normal_idw"]` or include it in your algorithm list.
- **Surface point count is now a direct count** - Replaced `surface_point_ratio` with `surface_point_count`. Instead of specifying a percentage, you now specify the exact number of surface points to include.
  - CLI: `--surface-point-count 1000` (default: 1000)
  - SDK: `sampler.generate(..., include_surface_points=True, surface_point_count=1000)`

## [0.3.0] - 2025-01-29

### Added

- **Full parameter control** in CLI and SDK
  - All analysis options exposed: `--min-gap-size`, `--cone-angle`, `--idw-sample-count`, etc.
  - All sampling options exposed: `--samples-per-primitive`, `--inverse-square-falloff`, etc.
  - Output mode control: `--flood-fill-output`, `--voxel-regions-output` (boxes/samples/both)
- **Surface point inclusion**
  - `--include-surface-points` flag to include original points with phi=0
  - `--surface-point-count` to specify number of surface points (default 1000)
  - SDK: `sampler.generate(..., include_surface_points=True, surface_point_count=1000)`

## [0.2.0] - 2025-01-29

### Added

- **Command-Line Interface** for batch processing
  - `sdf-sampler pipeline` - Full workflow (analyze + sample + export)
  - `sdf-sampler analyze` - Detect SOLID/EMPTY regions
  - `sdf-sampler sample` - Generate training samples from constraints
  - `sdf-sampler info` - Inspect point clouds, constraints, and sample files
- Support for `python -m sdf_sampler` invocation
- Console script entry point (`sdf-sampler` command)
- Comprehensive README with SDK and CLI documentation

## [0.1.0] - 2025-01-29

### Added

- Initial release extracted from sdf-labeler backend
- **SDFAnalyzer** class for auto-analysis of point clouds
  - `flood_fill` algorithm: EMPTY region detection via ray propagation from sky
  - `voxel_regions` algorithm: SOLID region detection from underground
  - `normal_offset` algorithm: Paired SOLID/EMPTY boxes along surface normals
  - `normal_idw` algorithm: Inverse distance weighted sampling along normals
  - `pocket` algorithm: Interior cavity detection via voxel flood fill
  - Hull filtering to remove constraints outside X-Y alpha shape
  - Configurable via `AnalyzerConfig` dataclass
- **SDFSampler** class for training sample generation
  - Supports multiple constraint types: box, sphere, halfspace, brush_stroke, etc.
  - Three sampling strategies: CONSTANT, DENSITY, INVERSE_SQUARE
  - Export to Parquet and DataFrame
  - Configurable via `SamplerConfig` dataclass
- **I/O helpers**
  - `load_point_cloud()` for PLY, LAS/LAZ, CSV, NPZ, Parquet formats
  - `export_parquet()` for survi-compatible training data export
- Pydantic models for all constraint types and training samples
- Comprehensive test suite (56 tests including equivalence tests vs sdf-labeler backend)

### Dependencies

- pydantic>=2.5.0 (data validation)
- numpy>=1.26.0 (core arrays)
- pandas>=2.1.0 (Parquet I/O)
- scipy>=1.11.0 (KDTree, spatial, ndimage)
- alphashape>=1.3.1 (concave hull filtering)
- pyarrow>=14.0.0 (Parquet export)

### Optional Dependencies

- trimesh>=4.0.0 (PLY loading, `[io]` extra)
- laspy[laszip]>=2.5.0 (LAS/LAZ loading, `[io]` extra)
