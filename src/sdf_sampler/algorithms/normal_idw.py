# ABOUTME: IDW normal sampling algorithm
# ABOUTME: Generates sample constraints along normals with inverse distance weighting

import logging
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from sdf_sampler.algorithms.normal_offset import _farthest_point_sample
from sdf_sampler.config import AutoAnalysisOptions
from sdf_sampler.models.analysis import AlgorithmType, GeneratedConstraint

logger = logging.getLogger(__name__)


def generate_idw_normal_samples(
    xyz: np.ndarray,
    normals: np.ndarray | None,
    options: AutoAnalysisOptions,
    flood_fill_state: Any | None = None,
) -> list[GeneratedConstraint]:
    """Generate sample constraints along normals with inverse distance weighting.

    Creates point samples at varying distances along surface normals, with
    more samples concentrated near the surface (IDW = 1/distance^power).

    Sign determination uses a KNN-based approach: for each sample point, the K
    nearest surface points vote on inside/outside based on their normals. This
    is more robust than single-normal heuristics for non-convex geometries.

    Args:
        xyz: Point cloud positions (N, 3)
        normals: Point normals (N, 3) - required for this algorithm
        options: Algorithm options
        flood_fill_state: Optional FloodFillState (unused, reserved for future use)

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

    # Build KDTree for KNN sign classification
    tree = KDTree(xyz)
    knn_k = min(8, len(xyz))

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
            side = rng.choice([-1, 1])
            sample_pos = point + side * dist * normal

            # Determine sign using KNN voting
            sample_sign = _classify_sign_knn(
                sample_pos, xyz, oriented_normals, tree, knn_k
            )
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


def _classify_sign_knn(
    sample_pos: np.ndarray,
    xyz: np.ndarray,
    oriented_normals: np.ndarray,
    tree: KDTree,
    k: int,
) -> str:
    """Classify a sample point as 'empty' or 'solid' using KNN normal voting.

    For each of the K nearest surface points, computes the dot product of
    (sample - surface_point) with the surface normal. Positive dot products
    indicate the sample is on the outward (empty) side; negative indicates
    the inward (solid) side. The majority vote determines the sign.

    Uses inverse-distance weighting so closer surface points have more
    influence on the vote.
    """
    sign, _ = _classify_sign_knn_scored(sample_pos, xyz, oriented_normals, tree, k)
    return sign


def _classify_sign_knn_scored(
    sample_pos: np.ndarray,
    xyz: np.ndarray,
    oriented_normals: np.ndarray,
    tree: KDTree,
    k: int,
) -> tuple[str, float]:
    """Classify a sample point with confidence score.

    Returns:
        Tuple of (sign_str, avg_score) where avg_score is the inverse-distance-
        weighted average dot product. Positive = empty, negative = solid.
        Larger magnitude = higher confidence.
    """
    dists, indices = tree.query(sample_pos, k=k)

    # Handle single-neighbor case
    if np.isscalar(dists):
        dists = np.array([dists])
        indices = np.array([indices])

    # Compute weighted vote
    weighted_score = 0.0
    total_weight = 0.0
    epsilon = 1e-8

    for d, idx in zip(dists, indices):
        to_sample = sample_pos - xyz[idx]
        dot = np.dot(to_sample, oriented_normals[idx])
        weight = 1.0 / (d + epsilon)
        weighted_score += dot * weight
        total_weight += weight

    if total_weight > 0:
        avg_score = weighted_score / total_weight
    else:
        avg_score = 0.0

    sign = "empty" if avg_score > 0 else "solid"
    return sign, float(avg_score)


def _orient_normals_outward(xyz: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Orient normals to point outward using a viewpoint heuristic.

    Uses assumption that viewpoint is above the scene (outdoor scenes).
    """
    centroid = xyz.mean(axis=0)
    z_range = xyz[:, 2].max() - xyz[:, 2].min()
    viewpoint = centroid.copy()
    # Ensure viewpoint is meaningfully above the surface even for flat planes
    viewpoint[2] = xyz[:, 2].max() + max(z_range * 0.5, 1.0)

    to_viewpoint = viewpoint - xyz
    dot_products = np.sum(normals * to_viewpoint, axis=1)

    oriented = normals.copy()
    flip_mask = dot_products < 0
    oriented[flip_mask] = -oriented[flip_mask]

    return oriented
