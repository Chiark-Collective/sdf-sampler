# ABOUTME: Voxel region algorithm for SOLID region detection
# ABOUTME: Uses ray propagation from underground to identify solid material

import numpy as np

from sdf_sampler.algorithms.flood_fill import (
    _generate_boxes_from_mask,
    _generate_samples_from_mask,
)
from sdf_sampler.algorithms.voxel_grid import (
    build_voxel_grid,
    compute_hull_mask,
    ray_propagation_with_bounces,
)
from sdf_sampler.config import AutoAnalysisOptions
from sdf_sampler.models.analysis import AlgorithmType, GeneratedConstraint
from sdf_sampler.models.constraints import SignConvention


def generate_voxel_region_constraints(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    options: AutoAnalysisOptions,
) -> list[GeneratedConstraint]:
    """Generate SOLID constraints for underground regions.

    Uses directional Z-ray propagation: SOLID propagates up from Z_min
    until hitting the surface. Only voxels inside the 2D convex hull
    are marked SOLID.

    Args:
        xyz: Point cloud positions (N, 3)
        normals: Point normals (N, 3) or None
        options: Algorithm options

    Returns:
        List of GeneratedConstraint objects
    """
    constraints: list[GeneratedConstraint] = []

    grid_result = build_voxel_grid(xyz, options.min_gap_size, options.max_grid_dim)
    if grid_result is None:
        return constraints

    occupied, bbox_min, voxel_size, grid_shape = grid_result
    _nx, _ny, nz = grid_shape

    inside_hull = compute_hull_mask(xyz, bbox_min, voxel_size, grid_shape)

    _, solid_mask = ray_propagation_with_bounces(
        occupied, grid_shape, inside_hull, options.cone_angle
    )

    output_mode = options.voxel_regions_output.lower()

    if output_mode in ("samples", "both"):
        sample_constraints = _generate_samples_from_mask(
            solid_mask,
            bbox_min,
            voxel_size,
            xyz,
            options.voxel_regions_sample_count,
            SignConvention.SOLID,
            AlgorithmType.VOXEL_REGIONS,
        )
        constraints.extend(sample_constraints)

    if output_mode in ("boxes", "both"):
        box_constraints = _generate_boxes_from_mask(
            solid_mask,
            bbox_min,
            voxel_size,
            nz,
            options,
            SignConvention.SOLID,
            AlgorithmType.VOXEL_REGIONS,
        )
        constraints.extend(box_constraints)

    return constraints
