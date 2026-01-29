# ABOUTME: Configuration dataclasses for analyzer and sampler
# ABOUTME: Simple config objects replacing FastAPI Settings dependency

from dataclasses import dataclass, field


@dataclass
class AnalyzerConfig:
    """Configuration for SDFAnalyzer.

    Controls voxel grid parameters, ray propagation, and filtering options.
    """

    # Voxel grid parameters
    min_gap_size: float = 0.10
    """Minimum gap size in meters that flood fill can traverse."""

    max_grid_dim: int = 200
    """Maximum voxel grid dimension (caps at max_grid_dim³ voxels)."""

    # Ray propagation
    cone_angle: float = 15.0
    """Ray propagation cone half-angle in degrees."""

    # Normal offset algorithm
    normal_offset_pairs: int = 40
    """Number of SOLID/EMPTY box pairs for normal_offset algorithm."""

    # Filtering
    max_boxes: int = 30
    """Maximum boxes per algorithm."""

    overlap_threshold: float = 0.5
    """Fraction overlap required to remove redundant boxes."""

    # IDW Normal sampling
    idw_sample_count: int = 1000
    """Total IDW samples to generate."""

    idw_max_distance: float = 0.5
    """Maximum distance from surface in meters."""

    idw_power: float = 2.0
    """IDW power factor (higher = more weight near surface)."""

    # Hull filtering
    hull_filter_enabled: bool = True
    """Filter out constraints outside the X-Y alpha shape of point cloud."""

    hull_alpha: float = 1.0
    """Alpha shape parameter (smaller = tighter fit to concave boundaries)."""

    # Output modes
    flood_fill_output: str = "samples"
    """Output mode for flood fill: 'boxes', 'samples', or 'both'."""

    flood_fill_sample_count: int = 500
    """Number of sample points from empty voxels."""

    voxel_regions_output: str = "samples"
    """Output mode for voxel regions: 'boxes', 'samples', or 'both'."""

    voxel_regions_sample_count: int = 500
    """Number of sample points from solid voxels."""

    # Pocket detection
    pocket_voxel_target: int = 100
    """Target number of voxels along longest axis for pocket detection."""

    pocket_min_voxel_size: float = 0.01
    """Minimum voxel size for pocket detection."""

    pocket_max_voxels_per_axis: int = 200
    """Maximum voxels per axis for pocket grid."""

    pocket_occupancy_dilation: int = 1
    """Dilation iterations for pocket occupancy grid."""

    pocket_min_volume_voxels: int = 10
    """Minimum voxels for a pocket to be considered significant."""


@dataclass
class SamplerConfig:
    """Configuration for SDFSampler.

    Controls sample generation strategy and parameters.
    """

    # Default sampling parameters
    total_samples: int = 10000
    """Default total number of samples to generate."""

    samples_per_primitive: int = 100
    """Samples per primitive constraint (CONSTANT strategy)."""

    samples_per_cubic_meter: float = 10000.0
    """Sample density per cubic meter (DENSITY strategy)."""

    inverse_square_base_samples: int = 100
    """Base samples at surface (INVERSE_SQUARE strategy)."""

    inverse_square_falloff: float = 2.0
    """Falloff exponent for inverse-square sampling."""

    # Band widths
    near_band: float = 0.02
    """Near-band width around surface."""

    # Random seed
    seed: int = 0
    """Random seed for reproducibility."""


@dataclass
class AutoAnalysisOptions:
    """Tunable hyperparameters for auto-analysis algorithms.

    This is a standalone version of the AutoAnalysisOptions from sdf-labeler,
    allowing fine-grained control over each algorithm.
    """

    # Voxel grid parameters
    min_gap_size: float = 0.10
    max_grid_dim: int = 200

    # Ray propagation
    cone_angle: float = 15.0

    # Normal offset
    normal_offset_pairs: int = 40

    # Filtering
    max_boxes: int = 30
    overlap_threshold: float = 0.5

    # IDW Normal sampling
    idw_sample_count: int = 1000
    idw_max_distance: float = 0.5
    idw_power: float = 2.0

    # Hull filtering
    hull_filter_enabled: bool = True
    hull_alpha: float = 1.0

    # Output modes
    flood_fill_output: str = "samples"
    flood_fill_sample_count: int = 500
    voxel_regions_output: str = "samples"
    voxel_regions_sample_count: int = 500

    @classmethod
    def from_analyzer_config(cls, config: AnalyzerConfig) -> "AutoAnalysisOptions":
        """Create options from AnalyzerConfig."""
        return cls(
            min_gap_size=config.min_gap_size,
            max_grid_dim=config.max_grid_dim,
            cone_angle=config.cone_angle,
            normal_offset_pairs=config.normal_offset_pairs,
            max_boxes=config.max_boxes,
            overlap_threshold=config.overlap_threshold,
            idw_sample_count=config.idw_sample_count,
            idw_max_distance=config.idw_max_distance,
            idw_power=config.idw_power,
            hull_filter_enabled=config.hull_filter_enabled,
            hull_alpha=config.hull_alpha,
            flood_fill_output=config.flood_fill_output,
            flood_fill_sample_count=config.flood_fill_sample_count,
            voxel_regions_output=config.voxel_regions_output,
            voxel_regions_sample_count=config.voxel_regions_sample_count,
        )
