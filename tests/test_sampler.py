# ABOUTME: Unit tests for SDFSampler
# ABOUTME: Tests sample generation from constraints

import numpy as np
import pandas as pd
import pytest

from sdf_sampler.config import SamplerConfig
from sdf_sampler.models.constraints import SignConvention
from sdf_sampler.models.samples import SamplingStrategy, TrainingSample
from sdf_sampler.sampler import SDFSampler


@pytest.fixture
def simple_xyz():
    """Simple point cloud for testing."""
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, (100, 3))


@pytest.fixture
def box_constraint():
    """Simple box constraint dict."""
    return {
        "type": "box",
        "sign": "solid",
        "center": (0.0, 0.0, 0.0),
        "half_extents": (0.5, 0.5, 0.5),
        "weight": 1.0,
    }


@pytest.fixture
def sphere_constraint():
    """Simple sphere constraint dict."""
    return {
        "type": "sphere",
        "sign": "empty",
        "center": (0.0, 0.0, 0.0),
        "radius": 0.5,
        "weight": 1.0,
    }


@pytest.fixture
def sample_point_constraint():
    """Sample point constraint dict."""
    return {
        "type": "sample_point",
        "sign": "empty",
        "position": (1.0, 2.0, 3.0),
        "distance": 0.5,
        "weight": 1.0,
    }


class TestSDFSampler:
    """Tests for SDFSampler class."""

    def test_init_default_config(self):
        sampler = SDFSampler()
        assert sampler.config is not None
        assert isinstance(sampler.config, SamplerConfig)

    def test_init_custom_config(self):
        config = SamplerConfig(total_samples=5000)
        sampler = SDFSampler(config=config)
        assert sampler.config.total_samples == 5000

    def test_generate_from_box(self, simple_xyz, box_constraint):
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[box_constraint],
            strategy=SamplingStrategy.CONSTANT,
        )

        assert len(samples) > 0
        assert all(isinstance(s, TrainingSample) for s in samples)

        # Check samples have correct source
        assert all("box_solid" in s.source for s in samples)

    def test_generate_from_sphere(self, simple_xyz, sphere_constraint):
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[sphere_constraint],
            strategy=SamplingStrategy.CONSTANT,
        )

        assert len(samples) > 0
        assert all("sphere_empty" in s.source for s in samples)

    def test_generate_from_sample_point(self, simple_xyz, sample_point_constraint):
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[sample_point_constraint],
        )

        assert len(samples) == 1
        assert samples[0].x == 1.0
        assert samples[0].y == 2.0
        assert samples[0].z == 3.0
        assert samples[0].phi == 0.5

    def test_generate_inverse_square_strategy(self, simple_xyz, box_constraint):
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[box_constraint],
            strategy=SamplingStrategy.INVERSE_SQUARE,
        )

        assert len(samples) > 0
        # Inverse square samples should have inv_sq in source
        assert all("inv_sq" in s.source for s in samples)

    def test_generate_density_strategy(self, simple_xyz, box_constraint):
        sampler = SDFSampler(config=SamplerConfig(
            samples_per_cubic_meter=1000,
        ))
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[box_constraint],
            strategy=SamplingStrategy.DENSITY,
        )

        # Density-based sampling - box volume is 1 cubic unit
        # At 1000 samples/m³, should get ~1000 samples
        assert len(samples) > 0

    def test_generate_with_seed(self, simple_xyz, box_constraint):
        sampler = SDFSampler()

        samples1 = sampler.generate(
            xyz=simple_xyz,
            constraints=[box_constraint],
            seed=42,
        )
        samples2 = sampler.generate(
            xyz=simple_xyz,
            constraints=[box_constraint],
            seed=42,
        )

        # Same seed should produce same samples
        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2):
            assert s1.x == s2.x
            assert s1.y == s2.y
            assert s1.z == s2.z

    def test_generate_multiple_constraints(self, simple_xyz, box_constraint, sphere_constraint):
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[box_constraint, sphere_constraint],
        )

        # Should have samples from both constraints
        box_samples = [s for s in samples if "box" in s.source]
        sphere_samples = [s for s in samples if "sphere" in s.source]

        assert len(box_samples) > 0
        assert len(sphere_samples) > 0

    def test_to_dataframe(self, simple_xyz, box_constraint):
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[box_constraint],
        )

        df = sampler.to_dataframe(samples)

        assert isinstance(df, pd.DataFrame)
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns
        assert "phi" in df.columns
        assert "source" in df.columns
        assert len(df) == len(samples)

    def test_export_parquet(self, simple_xyz, box_constraint, tmp_path):
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[box_constraint],
        )

        path = tmp_path / "samples.parquet"
        result_path = sampler.export_parquet(samples, path)

        assert result_path == path
        assert path.exists()

        # Verify we can read it back
        df = pd.read_parquet(path)
        assert len(df) == len(samples)


class TestSamplerSignConvention:
    """Tests for correct sign handling in sampler."""

    def test_solid_has_negative_phi(self, simple_xyz, box_constraint):
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[box_constraint],  # sign=solid
            strategy=SamplingStrategy.CONSTANT,
        )

        # SOLID should have negative phi
        for s in samples:
            assert s.phi < 0, f"SOLID sample should have negative phi, got {s.phi}"
            assert not s.is_free

    def test_empty_has_positive_phi(self, simple_xyz, sphere_constraint):
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=simple_xyz,
            constraints=[sphere_constraint],  # sign=empty
            strategy=SamplingStrategy.CONSTANT,
        )

        # EMPTY should have positive phi
        for s in samples:
            assert s.phi > 0, f"EMPTY sample should have positive phi, got {s.phi}"
            assert s.is_free


class TestBoxInverseSquarePhiValues:
    """Tests that box inverse_square samples use actual distance to surface for phi."""

    @pytest.fixture
    def plane_surface(self):
        """Flat plane at z=0 for easy distance calculation."""
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        xx, yy = np.meshgrid(x, y)
        xyz = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(2500)])
        return xyz

    @pytest.fixture
    def box_above_plane(self):
        """Box constraint above z=0 plane (empty region above surface)."""
        return {
            "type": "box",
            "sign": "empty",  # Above surface = empty/positive phi
            "center": (0.0, 0.0, 0.5),  # Center at z=0.5
            "half_extents": (1.0, 1.0, 0.5),  # Extends from z=0 to z=1
            "weight": 1.0,
        }

    @pytest.fixture
    def box_below_plane(self):
        """Box constraint below z=0 plane (solid region below surface)."""
        return {
            "type": "box",
            "sign": "solid",  # Below surface = solid/negative phi
            "center": (0.0, 0.0, -0.5),  # Center at z=-0.5
            "half_extents": (1.0, 1.0, 0.5),  # Extends from z=-1 to z=0
            "weight": 1.0,
        }

    def test_inverse_square_phi_is_actual_distance(self, plane_surface, box_above_plane):
        """Verify phi is based on distance to nearest surface point, not constant near_band."""
        sampler = SDFSampler(config=SamplerConfig(total_samples=500))
        samples = sampler.generate(
            xyz=plane_surface,
            constraints=[box_above_plane],
            strategy=SamplingStrategy.INVERSE_SQUARE,
            seed=42,
        )

        assert len(samples) > 0, "Should generate samples"

        for s in samples:
            # For a flat plane at z=0, distance to surface is approximately |z|
            # (exact value depends on nearest point in the discrete point cloud)
            z_distance = abs(s.z)

            # phi should be approximately |z| (within grid spacing tolerance)
            # Grid spacing is ~0.08 units, so allow some tolerance
            assert abs(s.phi - z_distance) < 0.1, (
                f"phi should be approximately equal to |z| distance. "
                f"Got phi={s.phi}, z={s.z}, expected ~{z_distance}"
            )

            # phi should definitely be positive for empty constraint
            assert s.phi > 0, f"EMPTY sample should have positive phi, got {s.phi}"

    def test_inverse_square_phi_varies_with_distance(self, plane_surface, box_above_plane):
        """Verify phi values vary based on sample distance from surface."""
        sampler = SDFSampler(config=SamplerConfig(total_samples=500))
        samples = sampler.generate(
            xyz=plane_surface,
            constraints=[box_above_plane],
            strategy=SamplingStrategy.INVERSE_SQUARE,
            seed=42,
        )

        phi_values = [s.phi for s in samples]

        # Phi should vary (not be constant ±near_band)
        phi_std = np.std(phi_values)
        assert phi_std > 0.01, (
            f"phi values should vary with distance, got std={phi_std}. "
            "This suggests phi is constant (bug: using near_band instead of distance)"
        )

        # Should have a range of values, not just near_band=0.02
        phi_min, phi_max = min(phi_values), max(phi_values)
        phi_range = phi_max - phi_min
        assert phi_range > 0.1, (
            f"phi range should be > 0.1, got {phi_range}. "
            "Values: min={phi_min}, max={phi_max}"
        )

    def test_inverse_square_solid_has_negative_phi(self, plane_surface, box_below_plane):
        """Verify solid box samples have negative phi with magnitude proportional to distance."""
        sampler = SDFSampler(config=SamplerConfig(total_samples=500))
        samples = sampler.generate(
            xyz=plane_surface,
            constraints=[box_below_plane],
            strategy=SamplingStrategy.INVERSE_SQUARE,
            seed=42,
        )

        for s in samples:
            # SOLID constraint should always have negative phi
            assert s.phi < 0, f"SOLID sample should have negative phi, got {s.phi}"

            # For flat plane at z=0, solid samples are at z<0
            # Distance to nearest surface point is approximately |z|
            z_distance = abs(s.z)

            # phi magnitude should be approximately |z| (within grid spacing tolerance)
            assert abs(abs(s.phi) - z_distance) < 0.1, (
                f"phi magnitude should be approximately |z|. "
                f"Got phi={s.phi}, z={s.z}, expected ~{-z_distance}"
            )

    def test_inverse_square_phi_correlates_with_z_coordinate(self, plane_surface, box_above_plane):
        """For plane at z=0, phi should be correlated with |z| coordinate."""
        sampler = SDFSampler(config=SamplerConfig(total_samples=200))
        samples = sampler.generate(
            xyz=plane_surface,
            constraints=[box_above_plane],
            strategy=SamplingStrategy.INVERSE_SQUARE,
            seed=123,
        )

        # Collect z values and phi values
        z_values = np.array([abs(s.z) for s in samples])
        phi_values = np.array([s.phi for s in samples])

        # phi should be positively correlated with |z|
        # (samples further from z=0 should have larger phi)
        correlation = np.corrcoef(z_values, phi_values)[0, 1]
        assert correlation > 0.9, (
            f"phi should be strongly correlated with |z|. "
            f"Got correlation={correlation:.3f}"
        )

    def test_inverse_square_not_constant_near_band(self, plane_surface, box_above_plane):
        """Explicitly verify phi is NOT the constant near_band value."""
        near_band = 0.02  # Default near_band value
        sampler = SDFSampler(config=SamplerConfig(total_samples=200, near_band=near_band))
        samples = sampler.generate(
            xyz=plane_surface,
            constraints=[box_above_plane],
            strategy=SamplingStrategy.INVERSE_SQUARE,
            seed=42,
        )

        # Count how many samples have phi approximately equal to near_band
        near_band_count = sum(1 for s in samples if abs(abs(s.phi) - near_band) < 0.001)
        total = len(samples)

        # With actual distance-based phi, very few samples should be exactly at near_band
        # (only those that happen to be exactly 0.02 away from surface)
        ratio = near_band_count / total
        assert ratio < 0.1, (
            f"{near_band_count}/{total} ({ratio:.0%}) samples have phi≈±near_band. "
            "This suggests phi is still using constant near_band instead of actual distance."
        )
