# Changelog

All notable changes to sdf-sampler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
