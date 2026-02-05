# ABOUTME: IDW normal sampling algorithm
# ABOUTME: Generates sample constraints along normals with inverse distance weighting

from typing import Any

import numpy as np

from sdf_sampler.algorithms.normal_offset import _farthest_point_sample
from sdf_sampler.config import AutoAnalysisOptions
from sdf_sampler.models.analysis import AlgorithmType, GeneratedConstraint


def generate_idw_normal_samples(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    options: AutoAnalysisOptions,
    flood_fill_state: Any | None = None,
) -> list[GeneratedConstraint]:
    """Generate sample constraints along normals with inverse distance weighting.

    Creates point samples at varying distances along surface normals, with
    more samples concentrated near the surface (IDW = 1/distance^power).

    When flood_fill_state is provided, uses the voxel grid to determine sign
    instead of relying on the normal orientation heuristic. This produces
    much more accurate signs for non-convex geometries like trenches.

    Args:
        xyz: Point cloud positions (N, 3)
        normals: Point normals (N, 3) - required for this algorithm
        options: Algorithm options
        flood_fill_state: Optional FloodFillState for voxel-based sign determination

    Returns:
        List of GeneratedConstraint objects
    """
    constraints: list[GeneratedConstraint] = []

    if normals is None or len(normals) != len(xyz):
        return constraints

    oriented_normals = _orient_normals_outward(xyz, normals)

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
            # Sample on both sides of the surface
            side = rng.choice([-1, 1])
            sample_pos = point + side * dist * normal

            # Determine sign from voxel grid if available
            if flood_fill_state is not None:
                sample_sign = _classify_point_sign(sample_pos, flood_fill_state)
            else:
                # Fallback: use normal direction heuristic
                sample_sign = "empty" if side > 0 else "solid"

            signed_dist = float(dist) if sample_sign == "empty" else float(-dist)

            constraints.append(
                GeneratedConstraint(
                    constraint={
                        "type": "sample_point",
                        "sign": sample_sign,
                        "position": tuple(sample_pos.tolist()),
                        "distance": signed_dist,
                        "algorithm": AlgorithmType.NORMAL_IDW.value,
                    },
                    algorithm=AlgorithmType.NORMAL_IDW,
                    confidence=0.8,
                    description=f"IDW sample at d={signed_dist:.3f}m",
                )
            )

    return constraints


def _classify_point_sign(pos: np.ndarray, state: Any) -> str:
    """Classify a point as 'empty' or 'solid' using the flood fill voxel grid.

    Uses empty_mask and solid_mask from FloodFillState to determine the sign.
    Falls back to 'solid' for points in unclassified or occupied voxels
    (conservative assumption: near-surface unresolved points are more likely
    to be solid material than open space).
    """
    voxel_idx = ((pos - state.bbox_min) / state.voxel_size).astype(int)
    nx, ny, nz = state.grid_shape

    # Out of bounds in XY: can't classify
    if voxel_idx[0] < 0 or voxel_idx[0] >= nx or voxel_idx[1] < 0 or voxel_idx[1] >= ny:
        return "empty"

    # Above grid: definitely empty (sky)
    if voxel_idx[2] >= nz:
        return "empty"

    # Below grid: definitely solid (underground)
    if voxel_idx[2] < 0:
        return "solid"

    ix, iy, iz = int(voxel_idx[0]), int(voxel_idx[1]), int(voxel_idx[2])

    if bool(state.empty_mask[ix, iy, iz]):
        return "empty"
    elif bool(state.solid_mask[ix, iy, iz]):
        return "solid"
    else:
        # Occupied or unclassified voxel — near the surface.
        # Use 'solid' as conservative default since IDW samples
        # that land inside geometry are more harmful than samples
        # that land just outside it.
        return "solid"


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
