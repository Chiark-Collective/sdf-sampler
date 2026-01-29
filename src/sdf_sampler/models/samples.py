# ABOUTME: Training sample models for SDF data generation
# ABOUTME: Defines sample formats and sampling strategies

from enum import Enum

from pydantic import BaseModel, Field


class SamplingStrategy(str, Enum):
    """Sampling strategy for generating training samples from constraints."""

    CONSTANT = "constant"  # Fixed samples per constraint
    DENSITY = "density"  # Samples proportional to constraint volume
    INVERSE_SQUARE = "inverse_square"  # More samples near surface, fewer far away


class TrainingSample(BaseModel):
    """Single training sample with SDF value.

    This is the core output format for SDF training data.
    """

    x: float
    y: float
    z: float
    phi: float = Field(..., description="Signed distance value")
    nx: float | None = None
    ny: float | None = None
    nz: float | None = None
    weight: float = 1.0
    source: str = Field(..., description="Sample source (e.g., 'surface_anchor', 'near_band')")
    is_surface: bool = False
    is_free: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "phi": self.phi,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "weight": self.weight,
            "source": self.source,
            "is_surface": self.is_surface,
            "is_free": self.is_free,
        }
