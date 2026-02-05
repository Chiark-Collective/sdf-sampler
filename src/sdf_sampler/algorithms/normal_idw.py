# ABOUTME: IDW normal sampling algorithm
# ABOUTME: Generates sample constraints along normals with inverse distance weighting

import numpy as np

from sdf_sampler.algorithms.normal_offset import _farthest_point_sample
from sdf_sampler.config import AutoAnalysisOptions
from sdf_sampler.models.analysis import AlgorithmType, GeneratedConstraint


def generate_idw_normal_samples(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    options: AutoAnalysisOptions,
) -> list[GeneratedConstraint]:
    """Generate sample constraints along normals with inverse distance weighting.

    Creates point samples at varying distances along surface normals, with
    more samples concentrated near the surface (IDW = 1/distance^power).

    For outdoor scenes (aerial/lidar capture), uses the global floor heuristic:
    - The lowest scanned surface defines the "floor"
    - SOLID samples are only generated below the floor level
    - Samples at or above the floor are marked EMPTY (reachable from sky)

    This prevents incorrect SOLID labeling inside open voids like trenches,
    where the normal direction from embedded objects (pipes) would otherwise
    point into the void.

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

    oriented_normals = _orient_normals_outward(xyz, normals)

    # Global floor heuristic: the lowest scanned surface is the floor.
    # SOLID samples should only exist below this level.
    global_floor = float(xyz[:, 2].min())

    n_surface_pts = min(options.idw_sample_count // 10, len(xyz))
    if n_surface_pts < 1:
        return constraints

    surface_indices = _farthest_point_sample(xyz, n_surface_pts)
    samples_per_point = options.idw_sample_count // len(surface_indices)

    if samples_per_point < 1:
        samples_per_point = 1

    rng = np.random.default_rng(42)

    for idx in surface_indices:
        point = xyz[idx]
        normal = oriented_normals[idx]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 0.1:
            continue
        normal = normal / normal_norm

        # Generate distances with IDW distribution
        u = rng.random(samples_per_point)
        distances = options.idw_max_distance * (1 - u ** (1 / options.idw_power))

        for dist in distances:
            sign = rng.choice([-1, 1])
            sample_pos = point + sign * dist * normal

            # Apply global floor heuristic:
            # - If sample is at or above the global floor, mark as EMPTY
            # - Only mark as SOLID if sample is below the floor
            if sample_pos[2] >= global_floor:
                sample_sign = "empty"
                # Adjust signed distance: positive for EMPTY
                signed_dist = abs(dist)
            else:
                sample_sign = "solid"
                # Negative signed distance for SOLID
                signed_dist = -abs(dist)

            constraints.append(
                GeneratedConstraint(
                    constraint={
                        "type": "sample_point",
                        "sign": sample_sign,
                        "position": tuple(sample_pos.tolist()),
                        "distance": float(signed_dist),
                    },
                    algorithm=AlgorithmType.NORMAL_IDW,
                    confidence=0.8,
                    description=f"IDW sample at d={signed_dist:.3f}m",
                )
            )

    return constraints


def _orient_normals_outward(xyz: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Orient normals to point outward using a viewpoint heuristic.

    Uses assumption that viewpoint is above the scene (outdoor scenes).
    """
    centroid = xyz.mean(axis=0)
    z_range = xyz[:, 2].max() - xyz[:, 2].min()
    viewpoint = centroid.copy()
    viewpoint[2] = xyz[:, 2].max() + z_range * 0.5

    to_viewpoint = viewpoint - xyz
    dot_products = np.sum(normals * to_viewpoint, axis=1)

    oriented = normals.copy()
    flip_mask = dot_products < 0
    oriented[flip_mask] = -oriented[flip_mask]

    return oriented
