# ABOUTME: Normal offset algorithm for surface-relative constraints
# ABOUTME: Generates paired SOLID/EMPTY boxes offset along surface normals

import numpy as np
from scipy.spatial import KDTree

from sdf_sampler.algorithms.voxel_grid import estimate_mean_spacing
from sdf_sampler.config import AutoAnalysisOptions
from sdf_sampler.models.analysis import AlgorithmType, GeneratedConstraint
from sdf_sampler.models.constraints import SignConvention


def generate_normal_offset_boxes(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    options: AutoAnalysisOptions,
) -> list[GeneratedConstraint]:
    """Generate paired SOLID/EMPTY boxes offset along surface normals.

    For a surface point with normal N:
    - Box offset in +N direction (outward) -> EMPTY
    - Box offset in -N direction (inward) -> SOLID

    Args:
        xyz: Point cloud positions (N, 3)
        normals: Point normals (N, 3) - required for this algorithm
        options: Algorithm options

    Returns:
        List of GeneratedConstraint objects
    """
    constraints: list[GeneratedConstraint] = []

    if normals is None or len(normals) != len(xyz):
        return constraints

    tree = KDTree(xyz)
    mean_spacing = estimate_mean_spacing(xyz, tree)

    sample_indices = _farthest_point_sample(xyz, options.normal_offset_pairs)
    offset_distance = mean_spacing * 3
    box_size = mean_spacing * 2.5

    for idx in sample_indices:
        point = xyz[idx]
        normal = normals[idx]
        normal_norm = np.linalg.norm(normal)

        if normal_norm < 0.1:
            continue

        normal = normal / normal_norm

        # EMPTY box in +normal direction (outward)
        empty_center = point + normal * offset_distance
        box_constraint_empty = {
            "type": "box",
            "sign": SignConvention.EMPTY.value,
            "center": tuple(empty_center.tolist()),
            "half_extents": (box_size, box_size, box_size),
        }

        constraints.append(
            GeneratedConstraint(
                constraint=box_constraint_empty,
                algorithm=AlgorithmType.NORMAL_OFFSET,
                confidence=0.75,
                description=f"Exterior offset from surface at ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})",
            )
        )

        # SOLID box in -normal direction (inward)
        solid_center = point - normal * offset_distance
        box_constraint_solid = {
            "type": "box",
            "sign": SignConvention.SOLID.value,
            "center": tuple(solid_center.tolist()),
            "half_extents": (box_size, box_size, box_size),
        }

        constraints.append(
            GeneratedConstraint(
                constraint=box_constraint_solid,
                algorithm=AlgorithmType.NORMAL_OFFSET,
                confidence=0.75,
                description=f"Interior offset from surface at ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})",
            )
        )

    return constraints


def _farthest_point_sample(xyz: np.ndarray, n_samples: int) -> list[int]:
    """Select well-distributed points using farthest point sampling."""
    n_points = len(xyz)
    if n_samples >= n_points:
        return list(range(n_points))

    rng = np.random.default_rng(42)
    selected: list[int] = [int(rng.integers(n_points))]
    min_distances = np.full(n_points, np.inf)

    for _ in range(n_samples - 1):
        last_selected = xyz[selected[-1]]
        distances = np.linalg.norm(xyz - last_selected, axis=1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[selected] = -1
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)

    return selected
