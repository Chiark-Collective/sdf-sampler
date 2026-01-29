# ABOUTME: Box constraint sampling functions
# ABOUTME: Generates training samples from axis-aligned box constraints

from typing import Any

import numpy as np

from sdf_sampler.models.constraints import BoxConstraint, SignConvention
from sdf_sampler.models.samples import TrainingSample


def sample_box(
    constraint: BoxConstraint,
    rng: np.random.Generator,
    near_band: float,
    n_samples: int,
) -> list[TrainingSample]:
    """Generate samples from a box constraint.

    Samples points near the box surfaces with appropriate SDF values.

    Args:
        constraint: Box constraint to sample
        rng: Random number generator
        near_band: Near-band width for offset
        n_samples: Number of samples to generate

    Returns:
        List of TrainingSample objects
    """
    samples = []
    center = np.array(constraint.center)
    half = np.array(constraint.half_extents)

    for _ in range(n_samples):
        # Random point near box surface
        face = rng.integers(0, 6)
        point = center + rng.uniform(-1, 1, 3) * half

        # Clamp to face
        axis = face // 2
        sign = 1 if face % 2 else -1
        point[axis] = center[axis] + sign * half[axis]

        # Offset based on sign convention
        offset = near_band if constraint.sign == SignConvention.EMPTY else -near_band
        normal = np.zeros(3)
        normal[axis] = sign
        point = point + offset * normal

        phi = offset

        samples.append(
            TrainingSample(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
                phi=phi,
                nx=float(normal[0]),
                ny=float(normal[1]),
                nz=float(normal[2]),
                weight=constraint.weight,
                source=f"box_{constraint.sign.value}",
                is_surface=False,
                is_free=constraint.sign == SignConvention.EMPTY,
            )
        )

    return samples


def sample_box_inverse_square(
    constraint: BoxConstraint,
    rng: np.random.Generator,
    near_band: float,
    n_samples: int,
    surface_tree: Any,
    falloff: float = 2.0,
) -> list[TrainingSample]:
    """Generate samples from a box with inverse-square density distribution.

    Samples more points near the surface (point cloud) and fewer far away.

    Args:
        constraint: Box constraint to sample
        rng: Random number generator
        near_band: Near-band width for offset
        n_samples: Number of samples to generate
        surface_tree: KDTree of surface points for distance computation
        falloff: Falloff exponent (higher = faster falloff)

    Returns:
        List of TrainingSample objects
    """
    samples = []
    center = np.array(constraint.center)
    half = np.array(constraint.half_extents)

    n_candidates = n_samples * 10

    for _ in range(n_candidates):
        if len(samples) >= n_samples:
            break

        point = center + rng.uniform(-1, 1, 3) * half
        dist_to_surface, _ = surface_tree.query(point, k=1)

        min_dist = max(dist_to_surface, near_band * 0.1)
        weight = (near_band / min_dist) ** falloff

        if rng.random() < min(1.0, weight):
            offset = near_band if constraint.sign == SignConvention.EMPTY else -near_band
            phi = offset

            samples.append(
                TrainingSample(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=float(point[2]),
                    phi=phi,
                    nx=0.0,
                    ny=0.0,
                    nz=0.0,
                    weight=constraint.weight,
                    source=f"box_{constraint.sign.value}_inv_sq",
                    is_surface=False,
                    is_free=constraint.sign == SignConvention.EMPTY,
                )
            )

    return samples
