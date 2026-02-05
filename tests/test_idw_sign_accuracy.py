# ABOUTME: Tests for IDW sign accuracy using voxel-informed normal orientation
# ABOUTME: Verifies that IDW samples get correct signs for trench-like geometries

import numpy as np

from sdf_sampler.algorithms.normal_idw import (
    _orient_normals_outward,
    generate_idw_normal_samples,
)
from sdf_sampler.config import AnalyzerConfig, AutoAnalysisOptions


def _make_idw_options(**overrides) -> AutoAnalysisOptions:
    """Create AutoAnalysisOptions for IDW testing."""
    defaults = dict(
        idw_sample_count=200,
        idw_max_distance=0.3,
        idw_power=2.0,
        flood_fill_output="samples",
        flood_fill_sample_count=50,
        voxel_regions_output="samples",
        voxel_regions_sample_count=50,
        hull_filter_enabled=False,
    )
    defaults.update(overrides)
    config = AnalyzerConfig(**defaults)
    return AutoAnalysisOptions.from_analyzer_config(config)


class TestOrientNormalsOutward:
    """Tests for normal orientation heuristic."""

    def test_flat_plane_normals_point_up(self):
        """For a flat horizontal plane, normals should point upward."""
        rng = np.random.default_rng(42)
        n = 100
        xyz = np.column_stack([
            rng.uniform(-1, 1, n),
            rng.uniform(-1, 1, n),
            np.zeros(n),
        ])
        # Random normal orientations (some up, some down)
        normals = np.zeros((n, 3))
        normals[:, 2] = rng.choice([-1.0, 1.0], n)

        oriented = _orient_normals_outward(xyz, normals)
        # All normals should now point up (toward the viewpoint above)
        assert np.all(oriented[:, 2] > 0)


class TestIDWSignAccuracyFlat:
    """Test IDW sign accuracy on a flat ground plane."""

    def test_flat_plane_signs_correct(self):
        """IDW samples above a flat plane should be EMPTY, below should be SOLID."""
        rng = np.random.default_rng(42)
        n = 500
        xyz = np.column_stack([
            rng.uniform(-1, 1, n),
            rng.uniform(-1, 1, n),
            np.zeros(n),
        ])
        normals = np.zeros((n, 3))
        normals[:, 2] = 1.0  # All pointing up

        options = _make_idw_options(idw_sample_count=200, idw_max_distance=0.5)
        constraints = generate_idw_normal_samples(xyz, normals, options)

        # Check sign accuracy: above plane (z>0) should be empty, below (z<0) should be solid
        n_correct = 0
        n_total = 0
        for c in constraints:
            pos = np.array(c.constraint["position"])
            sign = c.constraint["sign"]
            if abs(pos[2]) < 0.01:
                continue  # Skip points very close to surface
            n_total += 1
            expected = "empty" if pos[2] > 0 else "solid"
            if sign == expected:
                n_correct += 1

        accuracy = n_correct / n_total if n_total > 0 else 0
        assert accuracy > 0.90, f"Flat plane IDW accuracy {accuracy:.1%} should be >90%"


def _make_trench_points(n_ground=300, n_walls=100, n_floor=100, seed=42):
    """Create a simple trench: ground at z=0, trench from x=-0.3 to x=0.3, depth=0.5.

    Returns (xyz, normals) for a simple rectangular trench along Y axis.
    """
    rng = np.random.default_rng(seed)
    points = []
    norms = []

    # Ground surface (z=0, excluding trench opening)
    gx = rng.uniform(-2, 2, n_ground)
    gy = rng.uniform(-2, 2, n_ground)
    gz = np.zeros(n_ground)
    # Remove points inside trench opening
    mask = (np.abs(gx) > 0.3) | (np.abs(gy) > 1.0)
    points.append(np.column_stack([gx[mask], gy[mask], gz[mask]]))
    norms.append(np.column_stack([np.zeros(mask.sum()), np.zeros(mask.sum()),
                                   np.ones(mask.sum())]))

    # Left trench wall (x=-0.3, z from 0 to -0.5)
    wy = rng.uniform(-1, 1, n_walls)
    wz = rng.uniform(-0.5, 0, n_walls)
    wx = np.full(n_walls, -0.3)
    points.append(np.column_stack([wx, wy, wz]))
    norms.append(np.column_stack([-np.ones(n_walls), np.zeros(n_walls),
                                   np.zeros(n_walls)]))

    # Right trench wall (x=0.3, z from 0 to -0.5)
    wy = rng.uniform(-1, 1, n_walls)
    wz = rng.uniform(-0.5, 0, n_walls)
    wx = np.full(n_walls, 0.3)
    points.append(np.column_stack([wx, wy, wz]))
    norms.append(np.column_stack([np.ones(n_walls), np.zeros(n_walls),
                                   np.zeros(n_walls)]))

    # Trench floor (z=-0.5)
    fx = rng.uniform(-0.3, 0.3, n_floor)
    fy = rng.uniform(-1, 1, n_floor)
    fz = np.full(n_floor, -0.5)
    points.append(np.column_stack([fx, fy, fz]))
    norms.append(np.column_stack([np.zeros(n_floor), np.zeros(n_floor),
                                   np.ones(n_floor)]))

    xyz = np.vstack(points)
    normals = np.vstack(norms)
    return xyz, normals


class TestIDWSignAccuracyTrench:
    """Test IDW sign accuracy on a trench-like geometry (the hard case)."""

    def test_trench_void_samples_are_empty(self):
        """Samples inside the trench void should be EMPTY, not SOLID."""
        xyz, normals = _make_trench_points()

        options = _make_idw_options(idw_sample_count=500, idw_max_distance=0.2)
        constraints = generate_idw_normal_samples(xyz, normals, options)

        # Check sign accuracy for samples inside the trench void
        # Trench void: -0.3 < x < 0.3, -1 < y < 1, -0.5 < z < 0
        n_trench_correct = 0
        n_trench_total = 0
        for c in constraints:
            pos = np.array(c.constraint["position"])
            sign = c.constraint["sign"]

            # Is this point in the trench void?
            in_trench = (
                -0.25 < pos[0] < 0.25
                and -0.9 < pos[1] < 0.9
                and -0.45 < pos[2] < -0.05
            )
            if not in_trench:
                continue

            n_trench_total += 1
            if sign == "empty":
                n_trench_correct += 1

        if n_trench_total > 0:
            accuracy = n_trench_correct / n_trench_total
            assert accuracy > 0.70, (
                f"Trench void IDW accuracy {accuracy:.1%} ({n_trench_correct}/{n_trench_total}) "
                f"should be >70%"
            )

    def test_solid_ground_samples_are_solid(self):
        """Samples inside solid ground should be SOLID, not EMPTY."""
        xyz, normals = _make_trench_points()

        options = _make_idw_options(idw_sample_count=500, idw_max_distance=0.2)
        constraints = generate_idw_normal_samples(xyz, normals, options)

        # Check sign accuracy for samples inside solid ground
        # Solid ground: outside trench, z < 0
        n_ground_correct = 0
        n_ground_total = 0
        for c in constraints:
            pos = np.array(c.constraint["position"])
            sign = c.constraint["sign"]

            in_solid_ground = (
                (pos[0] < -0.4 or pos[0] > 0.4)
                and pos[2] < -0.05
            )
            if not in_solid_ground:
                continue

            n_ground_total += 1
            if sign == "solid":
                n_ground_correct += 1

        if n_ground_total > 0:
            accuracy = n_ground_correct / n_ground_total
            assert accuracy > 0.70, (
                f"Solid ground IDW accuracy {accuracy:.1%} ({n_ground_correct}/{n_ground_total}) "
                f"should be >70%"
            )


class TestKNNSignClassification:
    """Test that KNN-based sign classification is accurate."""

    def test_knn_sign_accuracy_on_trench(self):
        """KNN sign classification should correctly handle trench geometry."""
        xyz, normals = _make_trench_points(
            n_ground=500, n_walls=200, n_floor=200
        )

        options = _make_idw_options(
            idw_sample_count=500,
            idw_max_distance=0.2,
        )

        constraints = generate_idw_normal_samples(xyz, normals, options)

        def _score_accuracy(constraints, region_fn, expected_sign):
            correct = 0
            total = 0
            for c in constraints:
                pos = np.array(c.constraint["position"])
                sign = c.constraint["sign"]
                if not region_fn(pos):
                    continue
                total += 1
                if sign == expected_sign:
                    correct += 1
            return correct, total

        # Check trench void (should be EMPTY)
        def in_trench(pos):
            return (
                -0.25 < pos[0] < 0.25
                and -0.9 < pos[1] < 0.9
                and -0.45 < pos[2] < -0.05
            )

        trench_correct, trench_total = _score_accuracy(constraints, in_trench, "empty")
        if trench_total > 0:
            trench_acc = trench_correct / trench_total
            assert trench_acc > 0.65, (
                f"Trench void accuracy {trench_acc:.1%} ({trench_correct}/{trench_total}) "
                f"should be >65%"
            )

        # Check solid ground (should be SOLID)
        def in_solid(pos):
            return (pos[0] < -0.4 or pos[0] > 0.4) and pos[2] < -0.05

        solid_correct, solid_total = _score_accuracy(constraints, in_solid, "solid")
        if solid_total > 0:
            solid_acc = solid_correct / solid_total
            assert solid_acc > 0.65, (
                f"Solid ground accuracy {solid_acc:.1%} ({solid_correct}/{solid_total}) "
                f"should be >65%"
            )

        # Check above ground (should be EMPTY)
        def above_ground(pos):
            return pos[2] > 0.05

        above_correct, above_total = _score_accuracy(constraints, above_ground, "empty")
        if above_total > 0:
            above_acc = above_correct / above_total
            assert above_acc > 0.85, (
                f"Above ground accuracy {above_acc:.1%} ({above_correct}/{above_total}) "
                f"should be >85%"
            )
