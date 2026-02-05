# ABOUTME: Tests for flood-fill based sign validation in the analyzer
# ABOUTME: Verifies that constraint signs are corrected using voxel classification

import numpy as np
import pytest

from sdf_sampler.analyzer import SDFAnalyzer
from sdf_sampler.config import AnalyzerConfig
from sdf_sampler.models.analysis import AlgorithmType, GeneratedConstraint
from sdf_sampler.models.constraints import SignConvention


def _make_sample_point_constraint(
    position: tuple[float, float, float],
    sign: str,
    distance: float,
    algorithm: AlgorithmType = AlgorithmType.NORMAL_IDW,
) -> GeneratedConstraint:
    """Helper to create a sample_point GeneratedConstraint."""
    return GeneratedConstraint(
        constraint={
            "type": "sample_point",
            "sign": sign,
            "position": position,
            "distance": distance,
        },
        algorithm=algorithm,
        confidence=0.8,
        description=f"Test constraint at {position}",
    )


def _make_box_constraint(
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    sign: str,
    algorithm: AlgorithmType = AlgorithmType.NORMAL_OFFSET,
) -> GeneratedConstraint:
    """Helper to create a box GeneratedConstraint."""
    return GeneratedConstraint(
        constraint={
            "type": "box",
            "sign": sign,
            "center": center,
            "half_extents": half_extents,
        },
        algorithm=algorithm,
        confidence=0.85,
        description=f"Test box at {center}",
    )


class TestSignValidationUnit:
    """Unit tests for _validate_constraint_signs() using synthetic voxel state."""

    def test_wrong_sign_in_empty_voxel_gets_flipped(self):
        """A SOLID constraint in a flood-fill EMPTY voxel should be flipped to EMPTY."""
        # Build a simple plane point cloud (z=0 surface)
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.zeros(n)
        xyz = np.column_stack([x, y, z])

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=10,
            validate_signs=True,
        ))

        # Run flood_fill to populate voxel state
        from sdf_sampler.config import AutoAnalysisOptions
        options = AutoAnalysisOptions.from_analyzer_config(analyzer.config)
        analyzer._run_algorithm("flood_fill", xyz, None, options)

        # The analyzer should now have flood fill state cached
        assert analyzer._flood_fill_state is not None

        # Create a constraint above the plane (in empty space) but mislabeled SOLID
        constraint = _make_sample_point_constraint(
            position=(0.0, 0.0, 0.5),
            sign="solid",
            distance=-0.5,
        )

        result, stats = analyzer._validate_constraint_signs([constraint])
        assert len(result) == 1

        validated = result[0]
        # Should be flipped to EMPTY
        assert validated.constraint["sign"] == "empty"
        assert validated.constraint["distance"] > 0
        assert stats["n_flipped"] >= 1

    def test_correct_sign_in_empty_voxel_unchanged(self):
        """An EMPTY constraint in a flood-fill EMPTY voxel should remain unchanged."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.zeros(n)
        xyz = np.column_stack([x, y, z])

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=10,
            validate_signs=True,
        ))

        from sdf_sampler.config import AutoAnalysisOptions
        options = AutoAnalysisOptions.from_analyzer_config(analyzer.config)
        analyzer._run_algorithm("flood_fill", xyz, None, options)

        constraint = _make_sample_point_constraint(
            position=(0.0, 0.0, 0.5),
            sign="empty",
            distance=0.5,
        )

        result, stats = analyzer._validate_constraint_signs([constraint])
        assert len(result) == 1
        assert result[0].constraint["sign"] == "empty"
        assert result[0].constraint["distance"] == 0.5
        assert stats["n_flipped"] == 0

    def test_wrong_sign_below_ground_gets_flipped_to_solid(self):
        """A constraint in non-empty space below ground should be SOLID."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.zeros(n)
        xyz = np.column_stack([x, y, z])

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=10,
            validate_signs=True,
        ))

        from sdf_sampler.config import AutoAnalysisOptions
        options = AutoAnalysisOptions.from_analyzer_config(analyzer.config)
        analyzer._run_algorithm("flood_fill", xyz, None, options)

        # A point well below the surface, mislabeled as EMPTY
        constraint = _make_sample_point_constraint(
            position=(0.0, 0.0, -0.5),
            sign="empty",
            distance=0.5,
        )

        result, stats = analyzer._validate_constraint_signs([constraint])
        assert len(result) == 1
        validated = result[0]
        assert validated.constraint["sign"] == "solid"
        assert validated.constraint["distance"] < 0
        assert stats["n_flipped"] >= 1

    def test_box_constraints_validated_by_center(self):
        """Box constraints should be validated using their center position."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.zeros(n)
        xyz = np.column_stack([x, y, z])

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=10,
            validate_signs=True,
        ))

        from sdf_sampler.config import AutoAnalysisOptions
        options = AutoAnalysisOptions.from_analyzer_config(analyzer.config)
        analyzer._run_algorithm("flood_fill", xyz, None, options)

        # Box centered above the plane (in empty space) but mislabeled SOLID
        constraint = _make_box_constraint(
            center=(0.0, 0.0, 0.5),
            half_extents=(0.1, 0.1, 0.1),
            sign="solid",
        )

        result, stats = analyzer._validate_constraint_signs([constraint])
        assert len(result) == 1
        assert result[0].constraint["sign"] == "empty"

    def test_no_state_skips_validation(self):
        """If no flood_fill state is cached, validation should pass through unchanged."""
        analyzer = SDFAnalyzer(config=AnalyzerConfig(validate_signs=True))

        constraint = _make_sample_point_constraint(
            position=(0.0, 0.0, 0.5),
            sign="solid",
            distance=-0.5,
        )

        result, stats = analyzer._validate_constraint_signs([constraint])
        assert len(result) == 1
        assert result[0].constraint["sign"] == "solid"
        assert stats["n_flipped"] == 0

    def test_validation_preserves_other_fields(self):
        """Validation should only modify sign and distance, preserving all other fields."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.zeros(n)
        xyz = np.column_stack([x, y, z])

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=10,
            validate_signs=True,
        ))

        from sdf_sampler.config import AutoAnalysisOptions
        options = AutoAnalysisOptions.from_analyzer_config(analyzer.config)
        analyzer._run_algorithm("flood_fill", xyz, None, options)

        constraint = _make_sample_point_constraint(
            position=(0.0, 0.0, 0.5),
            sign="solid",
            distance=-0.5,
            algorithm=AlgorithmType.NORMAL_IDW,
        )

        result, stats = analyzer._validate_constraint_signs([constraint])
        validated = result[0]
        # Algorithm and confidence should be preserved
        assert validated.algorithm == AlgorithmType.NORMAL_IDW
        assert validated.confidence == 0.8
        # Position should be unchanged
        assert validated.constraint["position"] == (0.0, 0.0, 0.5)
        assert validated.constraint["type"] == "sample_point"

    def test_empty_constraints_returns_empty(self):
        """Passing an empty list should return empty list."""
        analyzer = SDFAnalyzer(config=AnalyzerConfig(validate_signs=True))
        result, stats = analyzer._validate_constraint_signs([])
        assert result == []
        assert stats["n_flipped"] == 0
        assert stats["n_checked"] == 0


class TestSignValidationIntegration:
    """Integration tests for sign validation through the full analyze() pipeline."""

    def test_analyze_with_validation_enabled(self):
        """Full analyze() run with validate_signs=True should produce valid results."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.zeros(n)
        xyz = np.column_stack([x, y, z])
        normals = np.zeros((n, 3))
        normals[:, 2] = 1.0

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=50,
            voxel_regions_output="samples",
            voxel_regions_sample_count=50,
            idw_sample_count=100,
            validate_signs=True,
        ))

        result = analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["flood_fill", "voxel_regions", "normal_idw"],
        )

        assert result.summary.total_constraints > 0
        # All empty constraints should be above or at z=0 (the surface)
        for gc in result.generated_constraints:
            c = gc.constraint
            if c.get("type") == "sample_point" and c.get("sign") == "empty":
                pos = c.get("position")
                if pos is not None:
                    # EMPTY samples should generally be above the surface
                    # (allow some tolerance for voxel discretization)
                    assert pos[2] >= -0.2, (
                        f"EMPTY sample at z={pos[2]:.3f} is too far below "
                        f"the surface (algorithm: {gc.algorithm})"
                    )

    def test_analyze_with_validation_disabled(self):
        """Full analyze() run with validate_signs=False should skip validation."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.zeros(n)
        xyz = np.column_stack([x, y, z])
        normals = np.zeros((n, 3))
        normals[:, 2] = 1.0

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=50,
            idw_sample_count=100,
            validate_signs=False,
        ))

        result = analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["flood_fill", "normal_idw"],
        )

        # Should still produce results, just without sign correction
        assert result.summary.total_constraints > 0

    def test_validation_requires_flood_fill_in_algorithm_list(self):
        """Sign validation only works when flood_fill is in the algorithm list."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.zeros(n)
        xyz = np.column_stack([x, y, z])
        normals = np.zeros((n, 3))
        normals[:, 2] = 1.0

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            idw_sample_count=100,
            validate_signs=True,
        ))

        # Run without flood_fill - validation should gracefully skip
        result = analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["normal_idw"],
        )

        assert result.summary.total_constraints > 0
        # No error should occur - validation just can't run without flood_fill state


class TestSignValidationStats:
    """Tests for validation statistics reporting."""

    def test_stats_track_flips_and_checks(self):
        """Validation stats should accurately count checks and flips."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        z = np.zeros(n)
        xyz = np.column_stack([x, y, z])

        analyzer = SDFAnalyzer(config=AnalyzerConfig(
            flood_fill_output="samples",
            flood_fill_sample_count=10,
            validate_signs=True,
        ))

        from sdf_sampler.config import AutoAnalysisOptions
        options = AutoAnalysisOptions.from_analyzer_config(analyzer.config)
        analyzer._run_algorithm("flood_fill", xyz, None, options)

        constraints = [
            # This one should get flipped (SOLID but in empty space above surface)
            _make_sample_point_constraint((0.0, 0.0, 0.5), "solid", -0.5),
            # This one is correct (EMPTY above surface)
            _make_sample_point_constraint((0.0, 0.0, 0.3), "empty", 0.3),
        ]

        _, stats = analyzer._validate_constraint_signs(constraints)
        assert stats["n_checked"] == 2
        assert stats["n_flipped"] >= 1
