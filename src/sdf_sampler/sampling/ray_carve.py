# ABOUTME: Ray carve constraint sampling functions
# ABOUTME: Generates training samples from ray-scribble interactions

import numpy as np

from sdf_sampler.models.constraints import RayCarveConstraint
from sdf_sampler.models.samples import TrainingSample


def sample_ray_carve(
    constraint: RayCarveConstraint,
    rng: np.random.Generator,
    n_samples_per_ray: int,
) -> list[TrainingSample]:
    """Generate samples from ray-carve constraint.

    For each ray:
    1. Sample EMPTY points uniformly along ray from origin to (hit - empty_band)
    2. Sample SURFACE points in band around hit point

    Args:
        constraint: Ray carve constraint to sample
        rng: Random number generator
        n_samples_per_ray: Number of samples per ray

    Returns:
        List of TrainingSample objects
    """
    samples = []

    # Pre-compute ray data for outlier detection
    ray_data = []
    for ray in constraint.rays:
        origin = np.array(ray.origin)
        direction = np.array(ray.direction)
        direction = direction / np.linalg.norm(direction)
        hit_point = origin + direction * ray.hit_distance
        ray_data.append(
            {
                "origin": origin,
                "direction": direction,
                "hit_distance": ray.hit_distance,
                "hit_point": hit_point,
                "local_spacing": ray.local_spacing,
                "surface_normal": ray.surface_normal,
            }
        )

    # Detect outliers: rays that pass through gaps
    effective_hit_distances = []
    for i, ray in enumerate(ray_data):
        effective_dist = ray["hit_distance"]

        for j, other in enumerate(ray_data):
            if i == j:
                continue

            dir_dot = np.dot(ray["direction"], other["direction"])
            if dir_dot > 0.95:
                to_hit = ray["hit_point"] - other["origin"]
                proj_dist = np.dot(to_hit, other["direction"])

                if proj_dist > other["hit_distance"] * 1.1:
                    effective_dist = min(effective_dist, other["hit_distance"])

        effective_hit_distances.append(effective_dist)

    for idx, ray in enumerate(ray_data):
        origin = ray["origin"]
        direction = ray["direction"]
        hit_dist = effective_hit_distances[idx]

        if ray["local_spacing"] is not None:
            buffer_zone = ray["local_spacing"] * constraint.back_buffer_coefficient
        else:
            buffer_zone = constraint.back_buffer_width

        # EMPTY samples along ray (before hit, stopping at buffer zone)
        empty_end = hit_dist - buffer_zone
        n_empty = n_samples_per_ray // 2

        if empty_end > 0:
            for _ in range(n_empty):
                t = rng.uniform(0, empty_end)
                point = origin + t * direction

                samples.append(
                    TrainingSample(
                        x=float(point[0]),
                        y=float(point[1]),
                        z=float(point[2]),
                        phi=hit_dist - t,
                        nx=float(direction[0]),
                        ny=float(direction[1]),
                        nz=float(direction[2]),
                        weight=constraint.weight,
                        source="ray_carve_empty",
                        is_surface=False,
                        is_free=True,
                    )
                )

        # SURFACE samples near hit
        n_surface = n_samples_per_ray - n_empty
        for _ in range(n_surface):
            t = rng.uniform(
                hit_dist - constraint.surface_band_width,
                hit_dist,
            )
            point = origin + t * direction
            phi = 0.0

            if ray["surface_normal"]:
                nx, ny, nz = ray["surface_normal"]
            else:
                nx, ny, nz = -direction[0], -direction[1], -direction[2]

            samples.append(
                TrainingSample(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=float(point[2]),
                    phi=phi,
                    nx=float(nx),
                    ny=float(ny),
                    nz=float(nz),
                    weight=constraint.weight,
                    source="ray_carve_surface",
                    is_surface=True,
                    is_free=False,
                )
            )

    return samples
