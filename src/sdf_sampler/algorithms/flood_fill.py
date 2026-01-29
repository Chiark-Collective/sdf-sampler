# ABOUTME: Flood fill algorithm for EMPTY region detection
# ABOUTME: Uses ray propagation from sky to identify exterior empty space

import numpy as np
from scipy.spatial import KDTree

from sdf_sampler.algorithms.voxel_grid import (
    build_voxel_grid,
    compute_hull_mask,
    greedy_2d_mesh,
    ray_propagation_with_bounces,
)
from sdf_sampler.config import AutoAnalysisOptions
from sdf_sampler.models.analysis import AlgorithmType, GeneratedConstraint
from sdf_sampler.models.constraints import SignConvention


def flood_fill_empty_regions(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    options: AutoAnalysisOptions,
) -> list[GeneratedConstraint]:
    """Generate EMPTY constraints using ray propagation with bouncing.

    Uses the ray model:
    1. EMPTY rays shine down from +Z (sky)
    2. Rays bounce to fill occluded areas (trenches, overhangs)
    3. Output depends on flood_fill_output option

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

    empty_mask, _ = ray_propagation_with_bounces(
        occupied, grid_shape, inside_hull, options.cone_angle
    )

    output_mode = options.flood_fill_output.lower()

    if output_mode in ("samples", "both"):
        sample_constraints = _generate_samples_from_mask(
            empty_mask,
            bbox_min,
            voxel_size,
            xyz,
            options.flood_fill_sample_count,
            SignConvention.EMPTY,
            AlgorithmType.FLOOD_FILL,
        )
        constraints.extend(sample_constraints)

    if output_mode in ("boxes", "both"):
        box_constraints = _generate_boxes_from_mask(
            empty_mask,
            bbox_min,
            voxel_size,
            nz,
            options,
            SignConvention.EMPTY,
            AlgorithmType.FLOOD_FILL,
        )
        constraints.extend(box_constraints)

    return constraints


def _generate_samples_from_mask(
    mask: np.ndarray,
    bbox_min: np.ndarray,
    voxel_size: float,
    xyz: np.ndarray,
    n_samples: int,
    sign: SignConvention,
    algorithm: AlgorithmType,
) -> list[GeneratedConstraint]:
    """Generate sample_point constraints from a voxel mask.

    Uses inverse-square distance weighting: more samples near the surface,
    fewer samples far away.
    """
    constraints: list[GeneratedConstraint] = []

    marked_indices = np.argwhere(mask)
    if len(marked_indices) == 0:
        return constraints

    tree = KDTree(xyz)
    rng = np.random.default_rng(42)

    voxel_centers = bbox_min + (marked_indices + 0.5) * voxel_size
    distances, _ = tree.query(voxel_centers, k=1)

    epsilon = voxel_size * 0.1
    weights = 1.0 / (distances + epsilon) ** 2
    weights = weights / weights.sum()

    sample_indices = rng.choice(len(marked_indices), size=n_samples, replace=True, p=weights)

    for idx in sample_indices:
        voxel_ijk = marked_indices[idx]
        offset = rng.uniform(0, 1, 3)
        world_pos = bbox_min + (voxel_ijk + offset) * voxel_size

        dist, _ = tree.query(world_pos, k=1)
        signed_dist = float(dist) if sign == SignConvention.EMPTY else -float(dist)

        constraints.append(
            GeneratedConstraint(
                constraint={
                    "type": "sample_point",
                    "sign": sign.value,
                    "position": tuple(world_pos.tolist()),
                    "distance": signed_dist,
                },
                algorithm=algorithm,
                confidence=0.8,
                description=f"Voxel sample at d={signed_dist:.3f}m",
            )
        )

    return constraints


def _generate_boxes_from_mask(
    empty_mask: np.ndarray,
    bbox_min: np.ndarray,
    voxel_size: float,
    nz: int,
    options: AutoAnalysisOptions,
    sign: SignConvention,
    algorithm: AlgorithmType,
) -> list[GeneratedConstraint]:
    """Generate axis-aligned box constraints from a voxel mask using greedy meshing."""
    constraints: list[GeneratedConstraint] = []

    all_boxes: list[tuple[int, int, int, int, int, int]] = []

    for iz in range(nz):
        slice_2d = empty_mask[:, :, iz]
        if not slice_2d.any():
            continue

        rectangles = greedy_2d_mesh(slice_2d)
        for x_min, x_max, y_min, y_max in rectangles:
            all_boxes.append((iz, iz + 1, x_min, x_max, y_min, y_max))

    if not all_boxes:
        return constraints

    all_boxes.sort(key=lambda b: (b[2], b[3], b[4], b[5], b[0]))

    merged_boxes: list[tuple[int, int, int, int, int, int]] = []
    current = all_boxes[0]

    for box in all_boxes[1:]:
        z_start, z_end, x_min, x_max, y_min, y_max = box
        cz_start, cz_end, cx_min, cx_max, cy_min, cy_max = current

        if (
            x_min == cx_min
            and x_max == cx_max
            and y_min == cy_min
            and y_max == cy_max
            and z_start == cz_end
        ):
            current = (cz_start, z_end, cx_min, cx_max, cy_min, cy_max)
        else:
            merged_boxes.append(current)
            current = box

    merged_boxes.append(current)

    min_extent = 3
    merged_boxes = [
        b
        for b in merged_boxes
        if (b[1] - b[0]) >= min_extent
        and (b[3] - b[2]) >= min_extent
        and (b[5] - b[4]) >= min_extent
    ]

    if not merged_boxes:
        return constraints

    max_boxes = options.max_boxes
    if len(merged_boxes) > max_boxes:
        merged_boxes.sort(
            key=lambda b: (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4]),
            reverse=True,
        )
        merged_boxes = merged_boxes[:max_boxes]

    for z_start, z_end, x_min, x_max, y_min, y_max in merged_boxes:
        world_min = bbox_min + np.array([x_min, y_min, z_start]) * voxel_size
        world_max = bbox_min + np.array([x_max, y_max, z_end]) * voxel_size

        center = (world_min + world_max) / 2
        half_extents = (world_max - world_min) / 2

        box_constraint = {
            "type": "box",
            "sign": sign.value,
            "center": tuple(center.tolist()),
            "half_extents": tuple(half_extents.tolist()),
        }

        volume = float(np.prod(half_extents * 2))
        n_voxels = (z_end - z_start) * (x_max - x_min) * (y_max - y_min)
        constraints.append(
            GeneratedConstraint(
                constraint=box_constraint,
                algorithm=algorithm,
                confidence=0.85,
                description=f"Sky-reachable region ({n_voxels} voxels, {volume:.2f}m³)",
            )
        )

    return constraints
