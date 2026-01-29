# ABOUTME: Unit tests for sdf-sampler models
# ABOUTME: Tests constraint types, analysis results, and sample models

import pytest

from sdf_sampler.models.analysis import (
    AlgorithmStats,
    AlgorithmType,
    AnalysisResult,
    AnalysisSummary,
    GeneratedConstraint,
)
from sdf_sampler.models.constraints import (
    BoxConstraint,
    BrushStrokeConstraint,
    HalfspaceConstraint,
    PocketConstraint,
    RayCarveConstraint,
    RayInfo,
    SamplePointConstraint,
    SignConvention,
    SphereConstraint,
)
from sdf_sampler.models.samples import SamplingStrategy, TrainingSample


class TestSignConvention:
    """Tests for SignConvention enum."""

    def test_values(self):
        assert SignConvention.SOLID.value == "solid"
        assert SignConvention.EMPTY.value == "empty"
        assert SignConvention.SURFACE.value == "surface"

    def test_string_comparison(self):
        assert SignConvention.SOLID == "solid"
        assert SignConvention.EMPTY == "empty"


class TestBoxConstraint:
    """Tests for BoxConstraint model."""

    def test_create_box(self):
        box = BoxConstraint(
            sign=SignConvention.SOLID,
            center=(1.0, 2.0, 3.0),
            half_extents=(0.5, 0.5, 0.5),
        )
        assert box.type == "box"
        assert box.sign == SignConvention.SOLID
        assert box.center == (1.0, 2.0, 3.0)
        assert box.half_extents == (0.5, 0.5, 0.5)
        assert box.weight == 1.0  # Default

    def test_box_with_weight(self):
        box = BoxConstraint(
            sign=SignConvention.EMPTY,
            center=(0, 0, 0),
            half_extents=(1, 1, 1),
            weight=2.5,
        )
        assert box.weight == 2.5

    def test_box_has_id(self):
        box = BoxConstraint(
            sign=SignConvention.SOLID,
            center=(0, 0, 0),
            half_extents=(1, 1, 1),
        )
        assert box.id is not None
        assert len(box.id) > 0


class TestSphereConstraint:
    """Tests for SphereConstraint model."""

    def test_create_sphere(self):
        sphere = SphereConstraint(
            sign=SignConvention.EMPTY,
            center=(0, 0, 0),
            radius=1.0,
        )
        assert sphere.type == "sphere"
        assert sphere.radius == 1.0

    def test_sphere_radius_validation(self):
        with pytest.raises(ValueError):
            SphereConstraint(
                sign=SignConvention.EMPTY,
                center=(0, 0, 0),
                radius=-1.0,  # Invalid
            )


class TestSamplePointConstraint:
    """Tests for SamplePointConstraint model."""

    def test_create_sample_point(self):
        sample = SamplePointConstraint(
            sign=SignConvention.EMPTY,
            position=(1.0, 2.0, 3.0),
            distance=0.5,
        )
        assert sample.type == "sample_point"
        assert sample.position == (1.0, 2.0, 3.0)
        assert sample.distance == 0.5


class TestTrainingSample:
    """Tests for TrainingSample model."""

    def test_create_sample(self):
        sample = TrainingSample(
            x=1.0,
            y=2.0,
            z=3.0,
            phi=0.5,
            source="test",
        )
        assert sample.x == 1.0
        assert sample.phi == 0.5
        assert sample.source == "test"
        assert sample.weight == 1.0  # Default

    def test_sample_to_dict(self):
        sample = TrainingSample(
            x=1.0,
            y=2.0,
            z=3.0,
            phi=0.5,
            nx=0.0,
            ny=0.0,
            nz=1.0,
            source="test",
            is_surface=True,
        )
        d = sample.to_dict()
        assert d["x"] == 1.0
        assert d["phi"] == 0.5
        assert d["is_surface"] is True


class TestAnalysisResult:
    """Tests for AnalysisResult model."""

    def test_create_result(self):
        result = AnalysisResult(
            analysis_id="test-123",
            algorithms_run=["flood_fill"],
            summary=AnalysisSummary(
                total_constraints=10,
                solid_constraints=5,
                empty_constraints=5,
                algorithms_contributing=1,
            ),
        )
        assert result.analysis_id == "test-123"
        assert len(result.algorithms_run) == 1

    def test_result_constraints_property(self):
        result = AnalysisResult(
            analysis_id="test",
            algorithms_run=["flood_fill"],
            summary=AnalysisSummary(
                total_constraints=1,
                solid_constraints=0,
                empty_constraints=1,
                algorithms_contributing=1,
            ),
            generated_constraints=[
                GeneratedConstraint(
                    constraint={"type": "box", "sign": "empty"},
                    algorithm=AlgorithmType.FLOOD_FILL,
                    confidence=0.9,
                    description="Test",
                )
            ],
        )
        assert len(result.constraints) == 1
        assert result.constraints[0]["type"] == "box"


class TestSamplingStrategy:
    """Tests for SamplingStrategy enum."""

    def test_values(self):
        assert SamplingStrategy.CONSTANT.value == "constant"
        assert SamplingStrategy.DENSITY.value == "density"
        assert SamplingStrategy.INVERSE_SQUARE.value == "inverse_square"
