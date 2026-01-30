# ABOUTME: SDFAnalyzer class for auto-analysis of point clouds
# ABOUTME: Detects SOLID and EMPTY regions using multiple algorithms

import uuid
from datetime import UTC, datetime

import numpy as np

from sdf_sampler.algorithms.flood_fill import flood_fill_empty_regions
from sdf_sampler.algorithms.normal_idw import generate_idw_normal_samples
from sdf_sampler.algorithms.normal_offset import generate_normal_offset_boxes
from sdf_sampler.algorithms.pocket import detect_pockets
from sdf_sampler.algorithms.voxel_regions import generate_voxel_region_constraints
from sdf_sampler.config import AnalyzerConfig, AutoAnalysisOptions
from sdf_sampler.models.analysis import (
    ALL_ALGORITHMS,
    DEFAULT_ALGORITHMS,
    AlgorithmStats,
    AlgorithmType,
    AnalysisResult,
    AnalysisSummary,
    GeneratedConstraint,
)
from sdf_sampler.models.constraints import SignConvention


class SDFAnalyzer:
    """Automatic SDF region detection from point clouds.

    Generates spatial constraints (boxes, samples) that define SOLID (inside)
    and EMPTY (outside) regions for SDF training data generation.

    Example:
        >>> analyzer = SDFAnalyzer()
        >>> result = analyzer.analyze(xyz=points, normals=normals)
        >>> print(f"Generated {len(result.constraints)} constraints")

    Algorithms:
        - flood_fill: EMPTY regions reachable from sky
        - voxel_regions: SOLID underground regions
        - normal_offset: Paired boxes along surface normals
        - normal_idw: Point samples with inverse distance weighting
        - pocket: Interior cavity detection
    """

    def __init__(self, config: AnalyzerConfig | None = None):
        """Initialize the analyzer.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or AnalyzerConfig()

    def analyze(
        self,
        xyz: np.ndarray,
        normals: np.ndarray | None = None,
        algorithms: list[str] | None = None,
        options: AutoAnalysisOptions | None = None,
    ) -> AnalysisResult:
        """Run analysis algorithms and generate constraints.

        This is the main entry point for auto-analysis. It runs the specified
        algorithms (or all by default) and returns constraints that can be
        used for sample generation.

        Args:
            xyz: Point cloud positions (N, 3) as numpy array
            normals: Point normals (N, 3) or None if not available
            algorithms: List of algorithm names to run (default: all)
            options: Fine-grained algorithm options (default: from config)

        Returns:
            AnalysisResult containing generated constraints and statistics

        Example:
            >>> result = analyzer.analyze(
            ...     xyz=points,
            ...     normals=normals,
            ...     algorithms=["flood_fill", "voxel_regions"],
            ... )
        """
        if options is None:
            options = AutoAnalysisOptions.from_analyzer_config(self.config)

        # Validate input
        xyz = np.asarray(xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")

        if normals is not None:
            normals = np.asarray(normals)
            if normals.shape != xyz.shape:
                raise ValueError(f"normals shape {normals.shape} doesn't match xyz {xyz.shape}")

        # Determine which algorithms to run
        algo_list = algorithms if algorithms else [a.value for a in DEFAULT_ALGORITHMS]
        algo_list = [a for a in algo_list if a in [alg.value for alg in ALL_ALGORITHMS]]

        # Run algorithms and collect constraints
        all_constraints: list[GeneratedConstraint] = []
        algorithm_stats: dict[str, AlgorithmStats] = {}
        algorithms_run: list[str] = []

        for algo_name in algo_list:
            constraints = self._run_algorithm(algo_name, xyz, normals, options)
            if constraints:
                all_constraints.extend(constraints)
                algorithms_run.append(algo_name)
                algorithm_stats[algo_name] = AlgorithmStats(
                    constraints_generated=len(constraints),
                    coverage_description=self._get_algorithm_description(algo_name, len(constraints)),
                )

        # Remove redundant contained boxes
        all_constraints = self._simplify_constraints(all_constraints, options.overlap_threshold)

        # Filter out constraints outside the X-Y alpha shape
        if options.hull_filter_enabled:
            all_constraints = self._filter_outside_hull(all_constraints, xyz, options.hull_alpha)

        # Compute summary
        summary = self._compute_summary(all_constraints, len(algorithm_stats))

        return AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            computed_at=datetime.now(UTC),
            algorithms_run=algorithms_run,
            summary=summary,
            algorithm_stats=algorithm_stats,
            generated_constraints=all_constraints,
        )

    async def analyze_async(
        self,
        xyz: np.ndarray,
        normals: np.ndarray | None = None,
        algorithms: list[str] | None = None,
        options: AutoAnalysisOptions | None = None,
    ) -> AnalysisResult:
        """Async variant of analyze() for heavy workloads.

        Same interface as analyze() but can be awaited.
        """
        # For now, just wrap sync - could be made truly async with executors
        return self.analyze(xyz, normals, algorithms, options)

    def _run_algorithm(
        self,
        name: str,
        xyz: np.ndarray,
        normals: np.ndarray | None,
        options: AutoAnalysisOptions,
    ) -> list[GeneratedConstraint]:
        """Run a single analysis algorithm."""
        if name == AlgorithmType.POCKET.value:
            return detect_pockets(xyz, self.config)
        elif name == AlgorithmType.NORMAL_OFFSET.value:
            return generate_normal_offset_boxes(xyz, normals, options)
        elif name == AlgorithmType.FLOOD_FILL.value:
            return flood_fill_empty_regions(xyz, normals, options)
        elif name == AlgorithmType.VOXEL_REGIONS.value:
            return generate_voxel_region_constraints(xyz, normals, options)
        elif name == AlgorithmType.NORMAL_IDW.value:
            return generate_idw_normal_samples(xyz, normals, options)
        return []

    def _get_algorithm_description(self, algo_name: str, count: int) -> str:
        """Get human-readable description for algorithm results."""
        descriptions = {
            AlgorithmType.POCKET.value: f"Detected {count} interior cavities",
            AlgorithmType.NORMAL_OFFSET.value: f"Generated {count} surface offset constraints",
            AlgorithmType.FLOOD_FILL.value: f"Found {count} sky-reachable exterior regions",
            AlgorithmType.VOXEL_REGIONS.value: f"Found {count} underground solid regions",
            AlgorithmType.NORMAL_IDW.value: f"Generated {count} IDW normal samples",
        }
        return descriptions.get(algo_name, f"Generated {count} constraints")

    def _compute_summary(
        self, constraints: list[GeneratedConstraint], algorithms_contributing: int
    ) -> AnalysisSummary:
        """Compute summary statistics from generated constraints."""
        solid_count = sum(
            1 for c in constraints if c.constraint.get("sign") == SignConvention.SOLID.value
        )
        empty_count = sum(
            1 for c in constraints if c.constraint.get("sign") == SignConvention.EMPTY.value
        )

        return AnalysisSummary(
            total_constraints=len(constraints),
            solid_constraints=solid_count,
            empty_constraints=empty_count,
            algorithms_contributing=algorithms_contributing,
        )

    def _box_intersection_fraction(self, box_a: dict, box_b: dict) -> float:
        """Calculate what fraction of box_b's volume intersects with box_a."""
        a_center = np.array(box_a["center"])
        a_half = np.array(box_a["half_extents"])
        b_center = np.array(box_b["center"])
        b_half = np.array(box_b["half_extents"])

        a_min, a_max = a_center - a_half, a_center + a_half
        b_min, b_max = b_center - b_half, b_center + b_half

        inter_min = np.maximum(a_min, b_min)
        inter_max = np.minimum(a_max, b_max)

        inter_dims = np.maximum(0, inter_max - inter_min)
        intersection_volume = float(np.prod(inter_dims))

        b_dims = b_max - b_min
        b_volume = float(np.prod(b_dims))

        if b_volume <= 0:
            return 0.0

        return intersection_volume / b_volume

    def _simplify_constraints(
        self, constraints: list[GeneratedConstraint], overlap_threshold: float = 0.5
    ) -> list[GeneratedConstraint]:
        """Remove boxes that significantly overlap with larger boxes."""
        boxes: list[tuple[int, GeneratedConstraint, float]] = []
        for i, c in enumerate(constraints):
            if c.constraint.get("type") == "box":
                half = np.array(c.constraint["half_extents"])
                volume = float(np.prod(half * 2))
                boxes.append((i, c, volume))

        remove_indices: set[int] = set()
        for i, (_idx_a, box_a, vol_a) in enumerate(boxes):
            for j, (idx_b, box_b, vol_b) in enumerate(boxes):
                if i == j:
                    continue
                if vol_b < vol_a:
                    fraction = self._box_intersection_fraction(box_a.constraint, box_b.constraint)
                    if fraction > overlap_threshold:
                        remove_indices.add(idx_b)

        return [c for i, c in enumerate(constraints) if i not in remove_indices]

    def _filter_outside_hull(
        self, constraints: list[GeneratedConstraint], xyz: np.ndarray, alpha: float
    ) -> list[GeneratedConstraint]:
        """Filter out constraints whose center falls outside the X-Y alpha shape."""
        if len(constraints) == 0 or len(xyz) < 3:
            return constraints

        xy = xyz[:, :2]

        try:
            import alphashape
            from shapely.geometry import Point

            shape = alphashape.alphashape(xy, alpha)
            if shape is None or shape.is_empty:
                return constraints
        except Exception:
            return constraints

        filtered: list[GeneratedConstraint] = []
        for constraint in constraints:
            c = constraint.constraint
            c_type = c.get("type")

            center_xy = self._get_constraint_center_xy(c, c_type)
            if center_xy is None:
                filtered.append(constraint)
                continue

            point = Point(center_xy[0], center_xy[1])
            if shape.contains(point) or shape.touches(point):
                filtered.append(constraint)

        return filtered

    def _get_constraint_center_xy(self, constraint: dict, c_type: str | None) -> np.ndarray | None:
        """Get the X-Y center of a constraint for hull checking."""
        center = None

        if c_type == "box":
            center = constraint.get("center")
        elif c_type == "sample_point":
            center = constraint.get("position")
        elif c_type == "sphere":
            center = constraint.get("center")
        elif c_type == "pocket":
            center = constraint.get("centroid")
        else:
            for field in ["center", "position", "point", "centroid"]:
                if field in constraint:
                    center = constraint[field]
                    break

        if center is None:
            return None

        return np.array(center[:2])
