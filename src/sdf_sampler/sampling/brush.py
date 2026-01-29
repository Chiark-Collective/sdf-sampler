# ABOUTME: Brush stroke constraint sampling functions
# ABOUTME: Generates training samples from volumetric brush strokes

import numpy as np

from sdf_sampler.models.constraints import BrushStrokeConstraint, SignConvention
from sdf_sampler.models.samples import TrainingSample


def sample_brush_stroke(
    constraint: BrushStrokeConstraint,
    rng: np.random.Generator,
    near_band: float,
    n_samples_per_point: int,
) -> list[TrainingSample]:
    """Generate samples from brush stroke volume.

    Samples uniformly within the tube-like stroke region.

    Args:
        constraint: Brush stroke constraint to sample
        rng: Random number generator
        near_band: Near-band width for phi calculation
        n_samples_per_point: Number of samples per stroke point

    Returns:
        List of TrainingSample objects
    """
    samples = []
    stroke_points = np.array(constraint.stroke_points)
    radius = constraint.radius

    if constraint.sign == SignConvention.SURFACE:
        phi = 0.0
    elif constraint.sign == SignConvention.SOLID:
        phi = -near_band
    else:
        phi = near_band

    for center in stroke_points:
        for _ in range(n_samples_per_point):
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            distance = rng.uniform(0, radius)
            point = center + distance * direction

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
                    source=f"brush_{constraint.sign.value}",
                    is_surface=constraint.sign == SignConvention.SURFACE,
                    is_free=constraint.sign == SignConvention.EMPTY,
                )
            )

    return samples
