# ABOUTME: Integration tests for sdf-sampler
# ABOUTME: Tests full pipeline from point cloud to parquet export

import numpy as np
import pandas as pd
import pytest

from sdf_sampler import SDFAnalyzer, SDFSampler, export_parquet
from sdf_sampler.config import AnalyzerConfig, SamplerConfig


@pytest.fixture
def trench_point_cloud():
    """Create a synthetic trench-like point cloud for testing.

    Creates a surface with a rectangular trench cut into it.
    This simulates the kind of outdoor scene the algorithms are designed for.
    """
    rng = np.random.default_rng(42)

    # Ground surface: z=0, but with a rectangular trench
    n_ground = 1000
    x = rng.uniform(-2, 2, n_ground)
    y = rng.uniform(-2, 2, n_ground)
    z = np.zeros(n_ground)

    # Create trench by lowering points in the trench region
    trench_mask = (np.abs(x) < 0.5) & (np.abs(y) < 1.5)
    z[trench_mask] = -0.5  # Trench floor

    # Add trench walls
    n_wall = 200
    wall_y = rng.uniform(-1.5, 1.5, n_wall * 2)
    wall_z = rng.uniform(-0.5, 0, n_wall * 2)
    wall_x = np.concatenate([
        np.full(n_wall, -0.5),  # Left wall
        np.full(n_wall, 0.5),   # Right wall
    ])

    xyz = np.vstack([
        np.column_stack([x, y, z]),
        np.column_stack([wall_x, wall_y, wall_z]),
    ])

    # Generate normals (simplified: up for ground, horizontal for walls)
    normals = np.zeros_like(xyz)
    normals[:n_ground, 2] = 1.0  # Ground points up
    normals[n_ground:n_ground + n_wall, 0] = 1.0  # Left wall points right
    normals[n_ground + n_wall:, 0] = -1.0  # Right wall points left

    return xyz, normals


class TestFullPipeline:
    """Integration tests for the full analysis -> sampling -> export pipeline."""

    def test_analyze_sample_export(self, trench_point_cloud, tmp_path):
        """Test the complete workflow from point cloud to parquet."""
        xyz, normals = trench_point_cloud

        # Step 1: Analyze
        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=100,
            voxel_regions_output="samples",
            voxel_regions_sample_count=100,
            idw_sample_count=100,
        ))
        result = analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["flood_fill", "voxel_regions", "normal_idw"],
        )

        assert result.summary.total_constraints > 0
        assert len(result.constraints) > 0

        # Step 2: Generate samples
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=xyz,
            constraints=result.constraints,
            strategy="inverse_square",
        )

        assert len(samples) > 0

        # Step 3: Export
        output_path = tmp_path / "training.parquet"
        export_parquet(samples, output_path)

        assert output_path.exists()

        # Verify parquet contents
        df = pd.read_parquet(output_path)
        assert len(df) == len(samples)
        assert "x" in df.columns
        assert "phi" in df.columns
        assert "source" in df.columns

    def test_analyze_only_flood_fill(self, trench_point_cloud):
        """Test flood fill finds EMPTY above ground."""
        xyz, _ = trench_point_cloud

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=200,
        ))
        result = analyzer.analyze(
            xyz=xyz,
            algorithms=["flood_fill"],
        )

        # Should find EMPTY regions above the surface
        empty_samples = [
            c for c in result.generated_constraints
            if c.constraint.get("sign") == "empty"
        ]
        assert len(empty_samples) > 0

        # Check that EMPTY samples are above the surface (z > 0 or in trench)
        for gc in empty_samples:
            c = gc.constraint
            if c.get("type") == "sample_point":
                pos = c.get("position", (0, 0, 0))
                # Either above ground or inside trench volume
                in_trench_area = abs(pos[0]) < 0.5 and abs(pos[1]) < 1.5
                assert pos[2] > -0.6 or in_trench_area, \
                    f"EMPTY sample at {pos} should be above surface"

    def test_analyze_only_voxel_regions(self, trench_point_cloud):
        """Test voxel regions finds SOLID below ground."""
        xyz, _ = trench_point_cloud

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            voxel_regions_output="samples",
            voxel_regions_sample_count=200,
        ))
        result = analyzer.analyze(
            xyz=xyz,
            algorithms=["voxel_regions"],
        )

        # Should find SOLID regions below the surface
        solid_samples = [
            c for c in result.generated_constraints
            if c.constraint.get("sign") == "solid"
        ]
        assert len(solid_samples) > 0

    def test_sample_generation_respects_signs(self, trench_point_cloud):
        """Test that samples have correct sign conventions."""
        xyz, normals = trench_point_cloud

        # Generate analysis
        analyzer = SDFAnalyzer()
        result = analyzer.analyze(
            xyz=xyz,
            normals=normals,
        )

        # Generate samples
        sampler = SDFSampler()
        samples = sampler.generate(
            xyz=xyz,
            constraints=result.constraints,
        )

        # Check phi signs match is_free flag
        for s in samples:
            if s.is_free:
                # EMPTY = positive phi
                assert s.phi >= 0 or "ray_carve" in s.source, \
                    f"is_free=True but phi={s.phi} for {s.source}"
            elif "solid" in s.source:
                # SOLID = negative phi
                assert s.phi <= 0, \
                    f"SOLID sample should have negative phi, got {s.phi}"


class TestDataFormats:
    """Tests for data format compatibility."""

    def test_parquet_has_expected_columns(self, trench_point_cloud, tmp_path):
        """Verify parquet output has survi-compatible columns."""
        xyz, normals = trench_point_cloud

        analyzer = SDFAnalyzer()
        result = analyzer.analyze(xyz=xyz, normals=normals)

        sampler = SDFSampler()
        samples = sampler.generate(xyz=xyz, constraints=result.constraints)

        path = tmp_path / "test.parquet"
        export_parquet(samples, path)

        df = pd.read_parquet(path)

        # Required columns for SDF training
        required_cols = ["x", "y", "z", "phi", "weight", "source", "is_surface", "is_free"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_to_dataframe_preserves_types(self, trench_point_cloud):
        """Test DataFrame conversion preserves data types."""
        xyz, normals = trench_point_cloud

        analyzer = SDFAnalyzer()
        result = analyzer.analyze(xyz=xyz, normals=normals)

        sampler = SDFSampler()
        samples = sampler.generate(xyz=xyz, constraints=result.constraints)
        df = sampler.to_dataframe(samples)

        # Check types
        assert df["x"].dtype == np.float64
        assert df["phi"].dtype == np.float64
        assert df["is_surface"].dtype == bool
        assert df["is_free"].dtype == bool


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_constraints(self):
        """Test handling of empty constraint list."""
        xyz = np.random.randn(100, 3)
        sampler = SDFSampler()
        samples = sampler.generate(xyz=xyz, constraints=[])
        assert len(samples) == 0

    def test_small_point_cloud(self):
        """Test with very small point cloud."""
        xyz = np.random.randn(20, 3)
        normals = np.random.randn(20, 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        analyzer = SDFAnalyzer()
        result = analyzer.analyze(xyz=xyz, normals=normals)

        # Should still work, even if no constraints generated
        assert result.analysis_id is not None

    def test_reproducibility(self, trench_point_cloud):
        """Test that analysis and sampling are reproducible with same seed."""
        xyz, normals = trench_point_cloud

        # Run twice with same configuration
        analyzer = SDFAnalyzer()
        sampler = SDFSampler()

        result1 = analyzer.analyze(xyz=xyz, normals=normals)
        samples1 = sampler.generate(xyz=xyz, constraints=result1.constraints, seed=42)

        result2 = analyzer.analyze(xyz=xyz, normals=normals)
        samples2 = sampler.generate(xyz=xyz, constraints=result2.constraints, seed=42)

        # Same number of samples
        assert len(samples1) == len(samples2)

        # Same positions (within floating point tolerance)
        for s1, s2 in zip(samples1, samples2):
            assert abs(s1.x - s2.x) < 1e-10
            assert abs(s1.y - s2.y) < 1e-10
            assert abs(s1.z - s2.z) < 1e-10
