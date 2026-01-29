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
