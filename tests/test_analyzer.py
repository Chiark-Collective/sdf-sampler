# ABOUTME: Unit tests for SDFAnalyzer
# ABOUTME: Tests auto-analysis algorithms and constraint generation

import numpy as np
import pytest

from sdf_sampler.analyzer import SDFAnalyzer
from sdf_sampler.config import AnalyzerConfig
from sdf_sampler.models.analysis import AlgorithmType
from sdf_sampler.models.constraints import SignConvention


@pytest.fixture
def simple_plane_xyz():
    """Create a simple planar point cloud for testing."""
    rng = np.random.default_rng(42)
    n_points = 500

    # Generate points on XY plane at z=0
    x = rng.uniform(-1, 1, n_points)
    y = rng.uniform(-1, 1, n_points)
    z = np.zeros(n_points)

    xyz = np.column_stack([x, y, z])
    return xyz


@pytest.fixture
def simple_plane_normals(simple_plane_xyz):
    """Create normals pointing up for the planar point cloud."""
    n_points = len(simple_plane_xyz)
    normals = np.zeros((n_points, 3))
    normals[:, 2] = 1.0  # All pointing up
    return normals


@pytest.fixture
def hemisphere_xyz():
    """Create a hemisphere point cloud for testing."""
    rng = np.random.default_rng(42)
    n_points = 1000

    # Fibonacci sphere sampling for upper hemisphere
    indices = np.arange(n_points)
    phi = np.arccos(1 - indices / n_points)  # Only upper hemisphere
    theta = np.pi * (1 + np.sqrt(5)) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    xyz = np.column_stack([x, y, z])
    return xyz


@pytest.fixture
def hemisphere_normals(hemisphere_xyz):
    """Normals for hemisphere (pointing outward = same as position for unit sphere)."""
    return hemisphere_xyz.copy()


class TestSDFAnalyzer:
    """Tests for SDFAnalyzer class."""

    def test_init_default_config(self):
        analyzer = SDFAnalyzer()
        assert analyzer.config is not None
        assert isinstance(analyzer.config, AnalyzerConfig)

    def test_init_custom_config(self):
        config = AnalyzerConfig(max_grid_dim=100)
        analyzer = SDFAnalyzer(config=config)
        assert analyzer.config.max_grid_dim == 100

    def test_analyze_validates_xyz_shape(self):
        analyzer = SDFAnalyzer()
        with pytest.raises(ValueError, match="xyz must be"):
            analyzer.analyze(xyz=np.array([1, 2, 3]))  # Wrong shape

    def test_analyze_validates_normals_shape(self, simple_plane_xyz):
        analyzer = SDFAnalyzer()
        with pytest.raises(ValueError, match="normals shape"):
            analyzer.analyze(
                xyz=simple_plane_xyz,
                normals=np.ones((10, 3)),  # Wrong size
            )

    def test_analyze_basic(self, simple_plane_xyz, simple_plane_normals):
        analyzer = SDFAnalyzer()
        result = analyzer.analyze(
            xyz=simple_plane_xyz,
            normals=simple_plane_normals,
            algorithms=["normal_idw"],
        )

        assert result.analysis_id is not None
        assert "normal_idw" in result.algorithms_run
        assert result.summary.total_constraints > 0

    def test_analyze_flood_fill(self, simple_plane_xyz):
        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=50,
        ))
        result = analyzer.analyze(
            xyz=simple_plane_xyz,
            algorithms=["flood_fill"],
        )

        assert "flood_fill" in result.algorithms_run
        # Should generate EMPTY samples above the plane
        empty_count = sum(
            1 for c in result.generated_constraints
            if c.constraint.get("sign") == SignConvention.EMPTY.value
        )
        assert empty_count > 0

    def test_analyze_voxel_regions(self, simple_plane_xyz):
        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            voxel_regions_output="samples",
            voxel_regions_sample_count=50,
        ))
        result = analyzer.analyze(
            xyz=simple_plane_xyz,
            algorithms=["voxel_regions"],
        )

        assert "voxel_regions" in result.algorithms_run
        # Should generate SOLID samples below the plane
        solid_count = sum(
            1 for c in result.generated_constraints
            if c.constraint.get("sign") == SignConvention.SOLID.value
        )
        assert solid_count > 0

    def test_analyze_normal_offset(self, simple_plane_xyz, simple_plane_normals):
        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            normal_offset_pairs=10,
        ))
        result = analyzer.analyze(
            xyz=simple_plane_xyz,
            normals=simple_plane_normals,
            algorithms=["normal_offset"],
        )

        assert "normal_offset" in result.algorithms_run
        # Should generate both EMPTY and SOLID boxes
        assert result.summary.empty_constraints > 0
        assert result.summary.solid_constraints > 0

    def test_analyze_normal_idw(self, hemisphere_xyz, hemisphere_normals):
        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            idw_sample_count=100,
        ))
        result = analyzer.analyze(
            xyz=hemisphere_xyz,
            normals=hemisphere_normals,
            algorithms=["normal_idw"],
        )

        assert "normal_idw" in result.algorithms_run
        # Should generate sample_point constraints
        sample_points = [
            c for c in result.generated_constraints
            if c.constraint.get("type") == "sample_point"
        ]
        assert len(sample_points) > 0

    def test_analyze_without_normals(self, simple_plane_xyz):
        analyzer = SDFAnalyzer()
        # Algorithms requiring normals should be skipped
        result = analyzer.analyze(
            xyz=simple_plane_xyz,
            normals=None,
            algorithms=["normal_offset", "normal_idw", "flood_fill"],
        )

        # Only flood_fill should run (doesn't require normals)
        assert "flood_fill" in result.algorithms_run
        assert "normal_offset" not in result.algorithms_run
        assert "normal_idw" not in result.algorithms_run

    def test_analyze_returns_constraints_property(self, simple_plane_xyz, simple_plane_normals):
        analyzer = SDFAnalyzer()
        result = analyzer.analyze(
            xyz=simple_plane_xyz,
            normals=simple_plane_normals,
            algorithms=["normal_idw"],
        )

        # The .constraints property should return just the constraint dicts
        constraints = result.constraints
        assert isinstance(constraints, list)
        if len(constraints) > 0:
            assert isinstance(constraints[0], dict)
            assert "type" in constraints[0]

    def test_analyze_summary_stats(self, simple_plane_xyz, simple_plane_normals):
        analyzer = SDFAnalyzer()
        result = analyzer.analyze(
            xyz=simple_plane_xyz,
            normals=simple_plane_normals,
        )

        summary = result.summary
        assert summary.total_constraints == summary.solid_constraints + summary.empty_constraints
        assert summary.algorithms_contributing > 0


class TestAnalyzerHullFiltering:
    """Tests for hull filtering functionality."""

    def test_hull_filter_removes_outside_constraints(self):
        # Create L-shaped point cloud
        rng = np.random.default_rng(42)
        n_per_arm = 200

        # Horizontal arm: x in [0, 2], y in [0, 1]
        x1 = rng.uniform(0, 2, n_per_arm)
        y1 = rng.uniform(0, 1, n_per_arm)

        # Vertical arm: x in [0, 1], y in [0, 2]
        x2 = rng.uniform(0, 1, n_per_arm)
        y2 = rng.uniform(0, 2, n_per_arm)

        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        z = np.zeros_like(x)

        xyz = np.column_stack([x, y, z])

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            hull_filter_enabled=True,
            hull_alpha=1.0,
        ))
        result = analyzer.analyze(xyz=xyz, algorithms=["flood_fill"])

        # Check that no constraints have centers in the "cut out" corner
        # (x > 1 AND y > 1 should be filtered out)
        for gc in result.generated_constraints:
            c = gc.constraint
            if c.get("type") == "box":
                center = c.get("center", (0, 0, 0))
                # Both x > 1 and y > 1 shouldn't happen
                assert not (center[0] > 1.5 and center[1] > 1.5), \
                    f"Constraint at {center} should have been filtered"
