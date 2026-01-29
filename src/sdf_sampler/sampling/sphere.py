# ABOUTME: Sphere constraint sampling functions
# ABOUTME: Generates training samples from spherical region constraints

import numpy as np

from sdf_sampler.models.constraints import SignConvention, SphereConstraint
from sdf_sampler.models.samples import TrainingSample


def sample_sphere(
    constraint: SphereConstraint,
    rng: np.random.Generator,
    near_band: float,
    n_samples: int,
) -> list[TrainingSample]:
    """Generate samples from a sphere constraint.

    Args:
        constraint: Sphere constraint to sample
        rng: Random number generator
        near_band: Near-band width for offset
        n_samples: Number of samples to generate

    Returns:
        List of TrainingSample objects
    """
    samples = []
    center = np.array(constraint.center)
    radius = constraint.radius

    for _ in range(n_samples):
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)

        point = center + radius * direction

        offset = near_band if constraint.sign == SignConvention.EMPTY else -near_band
        point = point + offset * direction
        phi = offset

        samples.append(
            TrainingSample(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
                phi=phi,
                nx=float(direction[0]),
                ny=float(direction[1]),
                nz=float(direction[2]),
                weight=constraint.weight,
                source=f"sphere_{constraint.sign.value}",
                is_surface=False,
                is_free=constraint.sign == SignConvention.EMPTY,
            )
        )

    return samples
