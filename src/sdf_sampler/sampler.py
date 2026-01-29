# ABOUTME: SDFSampler class for generating training samples from constraints
# ABOUTME: Converts constraints to survi-compatible training data

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from sdf_sampler.config import SamplerConfig
from sdf_sampler.models.constraints import (
    BoxConstraint,
    BrushStrokeConstraint,
    HalfspaceConstraint,
    PocketConstraint,
    RayCarveConstraint,
    SamplePointConstraint,
    SeedPropagationConstraint,
    SignConvention,
    SliceSelectionConstraint,
    SphereConstraint,
)
from sdf_sampler.models.samples import SamplingStrategy, TrainingSample
from sdf_sampler.sampling.box import sample_box, sample_box_inverse_square
from sdf_sampler.sampling.brush import sample_brush_stroke
from sdf_sampler.sampling.ray_carve import sample_ray_carve
from sdf_sampler.sampling.sphere import sample_sphere


class SDFSampler:
    """Generate training samples from constraints for SDF learning.

    Converts spatial constraints (boxes, spheres, etc.) into training samples
    with position, signed distance, and optional normals.

    Example:
        >>> sampler = SDFSampler()
        >>> samples = sampler.generate(
        ...     xyz=points,
        ...     normals=normals,
        ...     constraints=result.constraints,
        ... )
        >>> sampler.export_parquet(samples, "training.parquet")

    Strategies:
        - CONSTANT: Fixed samples per constraint
        - DENSITY: Samples proportional to constraint volume
        - INVERSE_SQUARE: More samples near surface, fewer far away
    """

    def __init__(self, config: SamplerConfig | None = None):
        """Initialize the sampler.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or SamplerConfig()

    def generate(
        self,
        xyz: np.ndarray,
        constraints: list[dict[str, Any]],
        normals: np.ndarray | None = None,
        total_samples: int | None = None,
        strategy: str | SamplingStrategy = SamplingStrategy.INVERSE_SQUARE,
        seed: int | None = None,
    ) -> list[TrainingSample]:
        """Generate training samples from constraints.

        Args:
            xyz: Point cloud positions (N, 3) for distance computation
            constraints: List of constraint dicts (from analyzer.analyze().constraints)
            normals: Optional point normals (N, 3)
            total_samples: Total samples to generate (default from config)
            strategy: Sampling strategy (CONSTANT, DENSITY, or INVERSE_SQUARE)
            seed: Random seed for reproducibility

        Returns:
            List of TrainingSample objects

        Example:
            >>> samples = sampler.generate(
            ...     xyz=points,
            ...     constraints=result.constraints,
            ...     strategy="inverse_square",
            ...     total_samples=50000,
            ... )
        """
        xyz = np.asarray(xyz)
        if normals is not None:
            normals = np.asarray(normals)

        if total_samples is None:
            total_samples = self.config.total_samples

        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy)

        rng = np.random.default_rng(seed if seed is not None else self.config.seed)

        # Build KD-tree for inverse_square strategy
        surface_tree = None
        if strategy == SamplingStrategy.INVERSE_SQUARE:
            surface_tree = KDTree(xyz)

        samples: list[TrainingSample] = []

        for constraint_dict in constraints:
            # Convert dict to typed constraint
            constraint = self._parse_constraint(constraint_dict)
            if constraint is None:
                continue

            n_samples = self._compute_sample_count(constraint, strategy)

            if isinstance(constraint, BoxConstraint):
                if strategy == SamplingStrategy.INVERSE_SQUARE and surface_tree is not None:
                    samples.extend(
                        sample_box_inverse_square(
                            constraint,
                            rng,
                            self.config.near_band,
                            n_samples,
                            surface_tree,
                            self.config.inverse_square_falloff,
                        )
                    )
                else:
                    samples.extend(
                        sample_box(constraint, rng, self.config.near_band, n_samples)
                    )
            elif isinstance(constraint, SphereConstraint):
                samples.extend(
                    sample_sphere(constraint, rng, self.config.near_band, n_samples)
                )
            elif isinstance(constraint, HalfspaceConstraint):
                samples.extend(
                    self._sample_halfspace(constraint, xyz, rng, n_samples)
                )
            elif isinstance(constraint, BrushStrokeConstraint):
                samples.extend(
                    sample_brush_stroke(constraint, rng, self.config.near_band, n_samples)
                )
            elif isinstance(constraint, SeedPropagationConstraint):
                samples.extend(self._sample_propagated(constraint, xyz, normals))
            elif isinstance(constraint, RayCarveConstraint):
                samples.extend(sample_ray_carve(constraint, rng, n_samples))
            elif isinstance(constraint, PocketConstraint):
                samples.extend(self._sample_pocket(constraint, rng, n_samples))
            elif isinstance(constraint, SliceSelectionConstraint):
                samples.extend(self._sample_slice_selection(constraint, xyz, normals))
            elif isinstance(constraint, SamplePointConstraint):
                samples.extend(self._sample_sample_point(constraint))

        return samples

    def to_dataframe(self, samples: list[TrainingSample]) -> pd.DataFrame:
        """Convert samples to pandas DataFrame.

        Args:
            samples: List of TrainingSample objects

        Returns:
            DataFrame with columns: x, y, z, phi, nx, ny, nz, weight, source, is_surface, is_free
        """
        return pd.DataFrame([s.to_dict() for s in samples])

    def export_parquet(
        self,
        samples: list[TrainingSample],
        path: str | Path,
    ) -> Path:
        """Export samples to Parquet file.

        Args:
            samples: List of TrainingSample objects
            path: Output file path

        Returns:
            Path to created file
        """
        path = Path(path)
        df = self.to_dataframe(samples)
        df.to_parquet(path)
        return path

    def _parse_constraint(self, constraint_dict: dict[str, Any]) -> Any:
        """Parse a constraint dict into a typed constraint object."""
        c_type = constraint_dict.get("type")

        if c_type == "box":
            return BoxConstraint(**constraint_dict)
        elif c_type == "sphere":
            return SphereConstraint(**constraint_dict)
        elif c_type == "halfspace":
            return HalfspaceConstraint(**constraint_dict)
        elif c_type == "brush_stroke":
            return BrushStrokeConstraint(**constraint_dict)
        elif c_type == "seed_propagation":
            return SeedPropagationConstraint(**constraint_dict)
        elif c_type == "ray_carve":
            return RayCarveConstraint(**constraint_dict)
        elif c_type == "pocket":
            return PocketConstraint(**constraint_dict)
        elif c_type == "slice_selection":
            return SliceSelectionConstraint(**constraint_dict)
        elif c_type == "sample_point":
            return SamplePointConstraint(**constraint_dict)

        return None

    def _compute_sample_count(
        self,
        constraint: Any,
        strategy: SamplingStrategy,
    ) -> int:
        """Compute number of samples for a constraint based on strategy."""
        if strategy == SamplingStrategy.CONSTANT:
            return self.config.samples_per_primitive

        elif strategy == SamplingStrategy.DENSITY:
            volume = self._compute_constraint_volume(constraint)
            return max(10, int(volume * self.config.samples_per_cubic_meter))

        elif strategy == SamplingStrategy.INVERSE_SQUARE:
            return self.config.inverse_square_base_samples

        return self.config.samples_per_primitive

    def _compute_constraint_volume(self, constraint: Any) -> float:
        """Compute approximate volume of a constraint in cubic meters."""
        if isinstance(constraint, BoxConstraint):
            half = np.array(constraint.half_extents)
            return float(np.prod(half * 2))

        elif isinstance(constraint, SphereConstraint):
            return (4 / 3) * np.pi * (constraint.radius**3)

        elif isinstance(constraint, PocketConstraint):
            voxel_size = 0.01
            return constraint.voxel_count * (voxel_size**3)

        elif isinstance(constraint, BrushStrokeConstraint):
            n_points = len(constraint.stroke_points)
            sphere_vol = (4 / 3) * np.pi * (constraint.radius**3)
            return n_points * sphere_vol * 0.5

        return 0.001

    def _sample_halfspace(
        self,
        constraint: HalfspaceConstraint,
        xyz: np.ndarray,
        rng: np.random.Generator,
        n_samples: int,
    ) -> list[TrainingSample]:
        """Generate samples from a halfspace constraint."""
        samples = []
        point = np.array(constraint.point)
        normal = np.array(constraint.normal)
        normal /= np.linalg.norm(normal)

        bounds_low = xyz.min(axis=0)
        bounds_high = xyz.max(axis=0)

        for _ in range(n_samples):
            sample_point = rng.uniform(bounds_low, bounds_high)
            dist = np.dot(sample_point - point, normal)

            if constraint.sign == SignConvention.EMPTY:
                phi = abs(dist) + self.config.near_band
            else:
                phi = -(abs(dist) + self.config.near_band)

            samples.append(
                TrainingSample(
                    x=float(sample_point[0]),
                    y=float(sample_point[1]),
                    z=float(sample_point[2]),
                    phi=phi,
                    nx=float(normal[0]),
                    ny=float(normal[1]),
                    nz=float(normal[2]),
                    weight=constraint.weight,
                    source=f"halfspace_{constraint.sign.value}",
                    is_surface=False,
                    is_free=constraint.sign == SignConvention.EMPTY,
                )
            )

        return samples

    def _sample_propagated(
        self,
        constraint: SeedPropagationConstraint,
        xyz: np.ndarray,
        normals: np.ndarray | None,
    ) -> list[TrainingSample]:
        """Generate samples from propagated seed."""
        samples = []

        for i, idx in enumerate(constraint.propagated_indices):
            if idx >= len(xyz):
                continue

            point = xyz[idx]
            normal = normals[idx] if normals is not None else [0, 0, 1]
            confidence = constraint.confidences[i] if i < len(constraint.confidences) else 1.0

            phi = (
                0.0
                if constraint.sign == SignConvention.SURFACE
                else (-0.01 if constraint.sign == SignConvention.SOLID else 0.01)
            )

            samples.append(
                TrainingSample(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=float(point[2]),
                    phi=phi,
                    nx=float(normal[0]),
                    ny=float(normal[1]),
                    nz=float(normal[2]),
                    weight=constraint.weight * confidence,
                    source=f"propagated_{constraint.sign.value}",
                    is_surface=constraint.sign == SignConvention.SURFACE,
                    is_free=constraint.sign == SignConvention.EMPTY,
                )
            )

        return samples

    def _sample_pocket(
        self,
        constraint: PocketConstraint,
        rng: np.random.Generator,
        n_samples: int,
    ) -> list[TrainingSample]:
        """Generate samples from a pocket constraint.

        Note: Without access to voxel data, we sample uniformly within bounds.
        For full pocket sampling, use the original SDF Labeler backend.
        """
        samples = []

        if constraint.sign == SignConvention.SOLID:
            phi = -0.05
        else:
            phi = 0.05

        bounds_low = np.array(constraint.bounds_low)
        bounds_high = np.array(constraint.bounds_high)

        for _ in range(n_samples):
            point = rng.uniform(bounds_low, bounds_high)

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
                    source=f"pocket_{constraint.sign.value}",
                    is_surface=False,
                    is_free=constraint.sign == SignConvention.EMPTY,
                )
            )

        return samples

    def _sample_slice_selection(
        self,
        constraint: SliceSelectionConstraint,
        xyz: np.ndarray,
        normals: np.ndarray | None,
    ) -> list[TrainingSample]:
        """Generate samples from slice selection constraint."""
        samples = []

        for idx in constraint.point_indices:
            if idx >= len(xyz):
                continue

            point = xyz[idx]
            normal = normals[idx] if normals is not None else [0, 0, 1]

            if constraint.sign == SignConvention.SURFACE:
                phi = 0.0
            elif constraint.sign == SignConvention.SOLID:
                phi = -0.01
            else:
                phi = 0.01

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
                    source=f"slice_{constraint.sign.value}",
                    is_surface=constraint.sign == SignConvention.SURFACE,
                    is_free=constraint.sign == SignConvention.EMPTY,
                )
            )

        return samples

    def _sample_sample_point(
        self,
        constraint: SamplePointConstraint,
    ) -> list[TrainingSample]:
        """Convert a sample_point constraint directly to a training sample."""
        phi = constraint.distance

        return [
            TrainingSample(
                x=float(constraint.position[0]),
                y=float(constraint.position[1]),
                z=float(constraint.position[2]),
                phi=phi,
                nx=0.0,
                ny=0.0,
                nz=0.0,
                weight=constraint.weight,
                source=f"idw_{constraint.sign.value}",
                is_surface=constraint.sign == SignConvention.SURFACE,
                is_free=constraint.sign == SignConvention.EMPTY,
            )
        ]
