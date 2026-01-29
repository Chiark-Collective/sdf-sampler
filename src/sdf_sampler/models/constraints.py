# ABOUTME: Constraint models for SDF region marking
# ABOUTME: Defines geometric primitives and regions for inside/outside labeling

import uuid
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SignConvention(str, Enum):
    """Sign convention for SDF values.

    User-friendly terminology:
    - SOLID = negative SDF (inside material)
    - EMPTY = positive SDF (outside, free space)
    - SURFACE = zero SDF (on the boundary)
    """

    SOLID = "solid"  # Inside / negative SDF
    EMPTY = "empty"  # Outside / positive SDF
    SURFACE = "surface"  # On boundary / zero SDF


class BaseConstraint(BaseModel):
    """Base class for all constraint types."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = Field(default=None, description="Optional user-friendly name")
    sign: SignConvention = Field(..., description="Label: solid (inside) or empty (outside)")
    weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Sample weight")


class BoxConstraint(BaseConstraint):
    """Axis-aligned bounding box constraint."""

    type: Literal["box"] = "box"
    center: tuple[float, float, float] = Field(..., description="Box center (x, y, z)")
    half_extents: tuple[float, float, float] = Field(
        ..., description="Half-extents in each dimension"
    )


class SphereConstraint(BaseConstraint):
    """Spherical region constraint."""

    type: Literal["sphere"] = "sphere"
    center: tuple[float, float, float] = Field(..., description="Sphere center (x, y, z)")
    radius: float = Field(..., gt=0, description="Sphere radius")


class HalfspaceConstraint(BaseConstraint):
    """Half-space (infinite plane) constraint.

    Defines a plane and which side is labeled.
    The normal points toward the EMPTY (outside) side by convention.
    """

    type: Literal["halfspace"] = "halfspace"
    point: tuple[float, float, float] = Field(..., description="A point on the plane")
    normal: tuple[float, float, float] = Field(
        ..., description="Outward normal (points toward empty/outside)"
    )


class CylinderConstraint(BaseConstraint):
    """Cylindrical region constraint."""

    type: Literal["cylinder"] = "cylinder"
    center: tuple[float, float, float] = Field(..., description="Center of cylinder base")
    axis: tuple[float, float, float] = Field(default=(0, 0, 1), description="Cylinder axis")
    radius: float = Field(..., gt=0, description="Cylinder radius")
    height: float = Field(..., gt=0, description="Cylinder height")


class BrushStrokeConstraint(BaseConstraint):
    """User-painted volumetric stroke in 3D space."""

    type: Literal["brush_stroke"] = "brush_stroke"
    stroke_points: list[tuple[float, float, float]] = Field(
        ..., description="Path of brush center positions"
    )
    radius: float = Field(..., gt=0, description="Brush/stroke radius")


class SeedPropagationConstraint(BaseConstraint):
    """Seed point with spatial propagation."""

    type: Literal["seed_propagation"] = "seed_propagation"
    seed_point: tuple[float, float, float] = Field(..., description="Seed location")
    seed_index: int | None = Field(default=None, description="Nearest point index")
    propagation_radius: float = Field(..., gt=0, description="Max propagation distance")
    propagation_method: Literal["euclidean", "geodesic"] = Field(
        default="euclidean", description="Distance metric for propagation"
    )
    propagated_indices: list[int] = Field(
        default_factory=list, description="Points reached by propagation"
    )
    confidences: list[float] = Field(
        default_factory=list, description="Confidence per propagated point"
    )


class RayInfo(BaseModel):
    """Single ray from a ray-scribble interaction."""

    origin: tuple[float, float, float] = Field(..., description="Ray origin (camera position)")
    direction: tuple[float, float, float] = Field(..., description="Normalized ray direction")
    hit_distance: float = Field(..., gt=0, description="Distance to first point cloud hit")
    surface_normal: tuple[float, float, float] | None = Field(
        default=None, description="Surface normal at hit point if available"
    )
    hit_point_index: int | None = Field(
        default=None, description="Index of hit point in consolidated positions"
    )
    local_spacing: float | None = Field(
        default=None, description="Local point spacing at hit point (k-NN mean distance)"
    )


class RayCarveConstraint(BaseConstraint):
    """Constraint from ray-scribble interaction.

    Each ray defines:
    - EMPTY samples along ray from origin to (hit_distance - empty_band)
    - SURFACE samples near hit_distance (from -surface_band to +back_buffer)
    """

    type: Literal["ray_carve"] = "ray_carve"
    rays: list[RayInfo] = Field(..., description="Rays cast during scribble stroke")
    empty_band_width: float = Field(
        default=0.1, gt=0, description="Distance before hit to sample as EMPTY"
    )
    surface_band_width: float = Field(
        default=0.02, gt=0, description="Distance before hit for SURFACE samples"
    )
    back_buffer_width: float = Field(
        default=0.0, ge=0, description="Fixed distance past hit (fallback when no local_spacing)"
    )
    back_buffer_coefficient: float = Field(
        default=1.0, ge=0, description="Multiplier for per-ray local_spacing"
    )


class PocketConstraint(BaseConstraint):
    """Pocket (cavity) constraint from voxel analysis."""

    type: Literal["pocket"] = "pocket"
    pocket_id: int = Field(..., description="Pocket identifier from analysis")
    voxel_count: int = Field(..., description="Number of voxels in pocket")
    centroid: tuple[float, float, float] = Field(..., description="Pocket centroid")
    bounds_low: tuple[float, float, float] = Field(..., description="Pocket AABB min")
    bounds_high: tuple[float, float, float] = Field(..., description="Pocket AABB max")
    volume_estimate: float = Field(..., description="Estimated volume in world units")


class SliceSelectionConstraint(BaseConstraint):
    """Constraint from 2D slice selection."""

    type: Literal["slice_selection"] = "slice_selection"
    point_indices: list[int] = Field(..., description="Selected point indices")
    slice_plane: Literal["xy", "xz", "yz"] = Field(..., description="Which plane was used")
    slice_position: float = Field(..., description="Position along perpendicular axis")


class SamplePointConstraint(BaseConstraint):
    """Single point sample constraint from IDW normal sampling.

    Direct training samples at specific 3D positions with known signed distance.
    """

    type: Literal["sample_point"] = "sample_point"
    position: tuple[float, float, float] = Field(..., description="Sample position (x, y, z)")
    distance: float = Field(
        ..., description="Signed distance to surface (negative=solid, positive=empty)"
    )


# Union type for all constraints
Constraint = Annotated[
    BoxConstraint
    | SphereConstraint
    | HalfspaceConstraint
    | CylinderConstraint
    | BrushStrokeConstraint
    | SeedPropagationConstraint
    | RayCarveConstraint
    | PocketConstraint
    | SliceSelectionConstraint
    | SamplePointConstraint,
    Field(discriminator="type"),
]
