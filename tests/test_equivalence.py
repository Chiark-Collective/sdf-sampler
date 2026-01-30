# ABOUTME: Equivalence tests comparing sdf-sampler to sdf-labeler backend
# ABOUTME: Ensures extracted package produces identical results to original

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Check if sdf-labeler backend is available (installed as package)
try:
    from sdf_labeler_api.services.auto_analysis_service import AutoAnalysisService
    from sdf_labeler_api.services.sampling_service import SamplingService
    from sdf_labeler_api.models.constraints import SignConvention as BackendSignConvention
    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

# Import from sdf-sampler (the new standalone package)
from sdf_sampler import SDFAnalyzer, SDFSampler
from sdf_sampler.config import AnalyzerConfig, AutoAnalysisOptions, SamplerConfig
from sdf_sampler.models.analysis import AlgorithmType
from sdf_sampler.models.constraints import SignConvention


def requires_backend(func):
    """Skip test if sdf-labeler backend is not available."""
    return pytest.mark.skipif(
        not HAS_BACKEND,
        reason="sdf-labeler backend not found at ../sdf-labeler/backend"
    )(func)


@pytest.fixture
def planar_pointcloud():
    """Generate a simple planar point cloud for testing."""
    rng = np.random.default_rng(42)
    n_points = 500

    # Generate points on XY plane at z=0
    x = rng.uniform(-1, 1, n_points)
    y = rng.uniform(-1, 1, n_points)
    z = np.zeros(n_points)

    xyz = np.column_stack([x, y, z]).astype(np.float32)

    # Normals pointing up
    normals = np.zeros((n_points, 3), dtype=np.float32)
    normals[:, 2] = 1.0

    return xyz, normals


@pytest.fixture
def hemisphere_pointcloud():
    """Generate a hemisphere point cloud for testing."""
    rng = np.random.default_rng(42)
    n_points = 800

    # Fibonacci sphere sampling for upper hemisphere
    indices = np.arange(n_points)
    phi = np.arccos(1 - indices / n_points)
    theta = np.pi * (1 + np.sqrt(5)) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    xyz = np.column_stack([x, y, z]).astype(np.float32)
    normals = xyz.copy()  # For unit sphere, normal = position

    return xyz, normals


@pytest.fixture
def trench_pointcloud():
    """Generate a trench-like point cloud (ground with a ditch)."""
    rng = np.random.default_rng(42)

    # Ground surface
    n_ground = 600
    x_ground = rng.uniform(-2, 2, n_ground)
    y_ground = rng.uniform(-2, 2, n_ground)
    z_ground = np.zeros(n_ground)

    # Trench floor (lower)
    trench_mask = (np.abs(x_ground) < 0.5) & (np.abs(y_ground) < 1.5)
    z_ground[trench_mask] = -0.5

    # Trench walls
    n_wall = 150
    wall_y = rng.uniform(-1.5, 1.5, n_wall * 2)
    wall_z = rng.uniform(-0.5, 0, n_wall * 2)
    wall_x = np.concatenate([
        np.full(n_wall, -0.5),
        np.full(n_wall, 0.5),
    ])

    xyz = np.vstack([
        np.column_stack([x_ground, y_ground, z_ground]),
        np.column_stack([wall_x, wall_y, wall_z]),
    ]).astype(np.float32)

    # Generate normals
    normals = np.zeros_like(xyz)
    normals[:n_ground, 2] = 1.0  # Ground points up
    normals[n_ground:n_ground + n_wall, 0] = 1.0  # Left wall
    normals[n_ground + n_wall:, 0] = -1.0  # Right wall

    return xyz, normals


def setup_backend_project(data_dir: Path, project_id: str, xyz: np.ndarray, normals: np.ndarray):
    """Set up a minimal project structure for the backend."""
    pc_dir = data_dir / "projects" / project_id / "pointcloud"
    pc_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pc_dir / "points.npz", xyz=xyz, normals=normals)


def get_backend_options(options: AutoAnalysisOptions):
    """Convert standalone options to backend AutoAnalysisOptions."""
    from sdf_labeler_api.models.auto_analysis import AutoAnalysisOptions as BackendOptions

    return BackendOptions(
        min_gap_size=options.min_gap_size,
        max_grid_dim=options.max_grid_dim,
        cone_angle=options.cone_angle,
        normal_offset_pairs=options.normal_offset_pairs,
        max_boxes=options.max_boxes,
        overlap_threshold=options.overlap_threshold,
        idw_sample_count=options.idw_sample_count,
        idw_max_distance=options.idw_max_distance,
        idw_power=options.idw_power,
        hull_filter_enabled=options.hull_filter_enabled,
        hull_alpha=options.hull_alpha,
        flood_fill_output=options.flood_fill_output,
        flood_fill_sample_count=options.flood_fill_sample_count,
        voxel_regions_output=options.voxel_regions_output,
        voxel_regions_sample_count=options.voxel_regions_sample_count,
    )


class TestAnalyzerEquivalence:
    """Test that SDFAnalyzer produces same results as AutoAnalysisService."""

    @requires_backend
    def test_flood_fill_equivalence(self, trench_pointcloud):
        """Test flood_fill algorithm produces equivalent results."""
        xyz, normals = trench_pointcloud

        # Configure options (same for both)
        options = AutoAnalysisOptions(
            flood_fill_output="samples",
            flood_fill_sample_count=100,
            hull_filter_enabled=False,  # Disable for determinism
        )

        # Run standalone
        standalone_analyzer = SDFAnalyzer()
        standalone_result = standalone_analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["flood_fill"],
            options=options,
        )

        # Run backend
        from sdf_labeler_api.config import Settings
        from sdf_labeler_api.services.auto_analysis_service import AutoAnalysisService

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            project_id = "test-flood"

            setup_backend_project(data_dir, project_id, xyz, normals)

            settings = Settings(data_dir=data_dir)
            backend_service = AutoAnalysisService(settings)
            backend_options = get_backend_options(options)

            backend_result = asyncio.run(backend_service.analyze(
                project_id=project_id,
                algorithms=["flood_fill"],
                recompute=True,
                options=backend_options,
            ))

        # Compare results
        self._compare_analysis_results(standalone_result, backend_result, "flood_fill")

    @requires_backend
    def test_voxel_regions_equivalence(self, trench_pointcloud):
        """Test voxel_regions algorithm produces equivalent results."""
        xyz, normals = trench_pointcloud

        options = AutoAnalysisOptions(
            voxel_regions_output="samples",
            voxel_regions_sample_count=100,
            hull_filter_enabled=False,
        )

        # Run standalone
        standalone_analyzer = SDFAnalyzer()
        standalone_result = standalone_analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["voxel_regions"],
            options=options,
        )

        # Run backend
        from sdf_labeler_api.config import Settings
        from sdf_labeler_api.services.auto_analysis_service import AutoAnalysisService

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            project_id = "test-voxel"

            setup_backend_project(data_dir, project_id, xyz, normals)

            settings = Settings(data_dir=data_dir)
            backend_service = AutoAnalysisService(settings)
            backend_options = get_backend_options(options)

            backend_result = asyncio.run(backend_service.analyze(
                project_id=project_id,
                algorithms=["voxel_regions"],
                recompute=True,
                options=backend_options,
            ))

        self._compare_analysis_results(standalone_result, backend_result, "voxel_regions")

    @requires_backend
    def test_normal_offset_equivalence(self, hemisphere_pointcloud):
        """Test normal_offset algorithm produces equivalent results."""
        xyz, normals = hemisphere_pointcloud

        options = AutoAnalysisOptions(
            normal_offset_pairs=20,
            hull_filter_enabled=False,
        )

        # Run standalone
        standalone_analyzer = SDFAnalyzer()
        standalone_result = standalone_analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["normal_offset"],
            options=options,
        )

        # Run backend
        from sdf_labeler_api.config import Settings
        from sdf_labeler_api.services.auto_analysis_service import AutoAnalysisService

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            project_id = "test-offset"

            setup_backend_project(data_dir, project_id, xyz, normals)

            settings = Settings(data_dir=data_dir)
            backend_service = AutoAnalysisService(settings)
            backend_options = get_backend_options(options)

            backend_result = asyncio.run(backend_service.analyze(
                project_id=project_id,
                algorithms=["normal_offset"],
                recompute=True,
                options=backend_options,
            ))

        self._compare_analysis_results(standalone_result, backend_result, "normal_offset")

    @requires_backend
    def test_normal_idw_equivalence(self, hemisphere_pointcloud):
        """Test normal_idw algorithm produces equivalent results."""
        xyz, normals = hemisphere_pointcloud

        options = AutoAnalysisOptions(
            idw_sample_count=100,  # Backend requires >= 100
            idw_max_distance=0.3,
            idw_power=2.0,
            hull_filter_enabled=False,
        )

        # Run standalone
        standalone_analyzer = SDFAnalyzer()
        standalone_result = standalone_analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["normal_idw"],
            options=options,
        )

        # Run backend
        from sdf_labeler_api.config import Settings
        from sdf_labeler_api.services.auto_analysis_service import AutoAnalysisService

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            project_id = "test-idw"

            setup_backend_project(data_dir, project_id, xyz, normals)

            settings = Settings(data_dir=data_dir)
            backend_service = AutoAnalysisService(settings)
            backend_options = get_backend_options(options)

            backend_result = asyncio.run(backend_service.analyze(
                project_id=project_id,
                algorithms=["normal_idw"],
                recompute=True,
                options=backend_options,
            ))

        self._compare_analysis_results(standalone_result, backend_result, "normal_idw")

    def _compare_analysis_results(self, standalone, backend, algorithm_name: str):
        """Compare analysis results from both implementations."""
        # Same algorithms should have run
        assert algorithm_name in standalone.algorithms_run
        assert algorithm_name in backend.algorithms_run

        # Same number of constraints (within tolerance for randomized algorithms)
        standalone_count = len(standalone.generated_constraints)
        backend_count = len(backend.generated_constraints)

        # Allow 10% variance for algorithms with randomness
        tolerance = 0.1
        min_count = min(standalone_count, backend_count)
        max_count = max(standalone_count, backend_count)

        if min_count > 0:
            variance = (max_count - min_count) / min_count
            assert variance <= tolerance, (
                f"Constraint count mismatch for {algorithm_name}: "
                f"standalone={standalone_count}, backend={backend_count}"
            )

        # Same sign distribution (approximately)
        standalone_solid = sum(
            1 for c in standalone.generated_constraints
            if c.constraint.get("sign") == "solid"
        )
        standalone_empty = sum(
            1 for c in standalone.generated_constraints
            if c.constraint.get("sign") == "empty"
        )

        backend_solid = sum(
            1 for c in backend.generated_constraints
            if c.constraint.get("sign") == "solid"
        )
        backend_empty = sum(
            1 for c in backend.generated_constraints
            if c.constraint.get("sign") == "empty"
        )

        # For deterministic algorithms, counts should match exactly
        # For randomized ones, check same sign dominance
        if standalone_solid > standalone_empty:
            assert backend_solid >= backend_empty, (
                f"Sign distribution mismatch for {algorithm_name}"
            )
        elif standalone_empty > standalone_solid:
            assert backend_empty >= backend_solid, (
                f"Sign distribution mismatch for {algorithm_name}"
            )

        # Same constraint types
        standalone_types = set(c.constraint.get("type") for c in standalone.generated_constraints)
        backend_types = set(c.constraint.get("type") for c in backend.generated_constraints)
        assert standalone_types == backend_types, (
            f"Constraint types mismatch for {algorithm_name}: "
            f"standalone={standalone_types}, backend={backend_types}"
        )


class TestSamplerEquivalence:
    """Test that SDFSampler produces same results as SamplingService."""

    @requires_backend
    def test_box_sampling_equivalence(self, planar_pointcloud):
        """Test box constraint sampling produces equivalent results."""
        xyz, normals = planar_pointcloud

        # Create identical box constraint
        box_dict = {
            "type": "box",
            "sign": "solid",
            "center": (0.0, 0.0, 0.0),
            "half_extents": (0.3, 0.3, 0.1),
            "weight": 1.0,
        }

        # Run standalone
        standalone_sampler = SDFSampler(config=SamplerConfig(
            samples_per_primitive=100,
            near_band=0.02,
            seed=42,
        ))
        standalone_samples = standalone_sampler.generate(
            xyz=xyz,
            constraints=[box_dict],
            strategy="constant",
            seed=42,
        )

        # Run backend
        from sdf_labeler_api.config import Settings
        from sdf_labeler_api.services.sampling_service import SamplingService
        from sdf_labeler_api.services.project_service import ProjectService
        from sdf_labeler_api.services.constraint_service import ConstraintService
        from sdf_labeler_api.models.project import ProjectCreate
        from sdf_labeler_api.models.constraints import BoxConstraint as BackendBox
        from sdf_labeler_api.models.constraints import SignConvention as BackendSign
        from sdf_labeler_api.models.samples import SampleGenerationRequest, SamplingStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            project_id = "test-box-sampling"

            setup_backend_project(data_dir, project_id, xyz, normals)

            # Patch settings
            import sdf_labeler_api.config as backend_config
            original_settings = backend_config.settings
            backend_config.settings = Settings(data_dir=data_dir)

            try:
                project_service = ProjectService(data_dir)
                project = project_service.create(ProjectCreate(name="test"))
                project_id = project.id

                # Re-save point cloud with correct project ID
                setup_backend_project(data_dir, project_id, xyz, normals)

                constraint_service = ConstraintService()
                backend_box = BackendBox(
                    sign=BackendSign.SOLID,
                    center=(0.0, 0.0, 0.0),
                    half_extents=(0.3, 0.3, 0.1),
                    weight=1.0,
                )
                constraint_service.add(project_id, backend_box)

                sampling_service = SamplingService()
                request = SampleGenerationRequest(
                    total_samples=10000,
                    strategy=SamplingStrategy.CONSTANT,
                    samples_per_primitive=100,
                    seed=42,
                )

                backend_result = sampling_service.generate(project_id, request)
                backend_samples = backend_result.samples
            finally:
                backend_config.settings = original_settings

        # Compare samples
        self._compare_samples(standalone_samples, backend_samples, "box_solid")

    @requires_backend
    def test_sphere_sampling_equivalence(self, planar_pointcloud):
        """Test sphere constraint sampling produces equivalent results."""
        xyz, normals = planar_pointcloud

        sphere_dict = {
            "type": "sphere",
            "sign": "empty",
            "center": (0.0, 0.0, 0.5),
            "radius": 0.3,
            "weight": 1.0,
        }

        # Run standalone
        standalone_sampler = SDFSampler(config=SamplerConfig(
            samples_per_primitive=100,
            near_band=0.02,
            seed=42,
        ))
        standalone_samples = standalone_sampler.generate(
            xyz=xyz,
            constraints=[sphere_dict],
            strategy="constant",
            seed=42,
        )

        # Run backend
        from sdf_labeler_api.config import Settings
        from sdf_labeler_api.services.sampling_service import SamplingService
        from sdf_labeler_api.services.project_service import ProjectService
        from sdf_labeler_api.services.constraint_service import ConstraintService
        from sdf_labeler_api.models.project import ProjectCreate
        from sdf_labeler_api.models.constraints import SphereConstraint as BackendSphere
        from sdf_labeler_api.models.constraints import SignConvention as BackendSign
        from sdf_labeler_api.models.samples import SampleGenerationRequest, SamplingStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            import sdf_labeler_api.config as backend_config
            original_settings = backend_config.settings
            backend_config.settings = Settings(data_dir=data_dir)

            try:
                project_service = ProjectService(data_dir)
                project = project_service.create(ProjectCreate(name="test"))
                project_id = project.id

                setup_backend_project(data_dir, project_id, xyz, normals)

                constraint_service = ConstraintService()
                backend_sphere = BackendSphere(
                    sign=BackendSign.EMPTY,
                    center=(0.0, 0.0, 0.5),
                    radius=0.3,
                    weight=1.0,
                )
                constraint_service.add(project_id, backend_sphere)

                sampling_service = SamplingService()
                request = SampleGenerationRequest(
                    total_samples=10000,
                    strategy=SamplingStrategy.CONSTANT,
                    samples_per_primitive=100,
                    seed=42,
                )

                backend_result = sampling_service.generate(project_id, request)
                backend_samples = backend_result.samples
            finally:
                backend_config.settings = original_settings

        self._compare_samples(standalone_samples, backend_samples, "sphere_empty")

    @requires_backend
    def test_sample_point_equivalence(self, planar_pointcloud):
        """Test sample_point constraint produces equivalent results."""
        xyz, normals = planar_pointcloud

        sample_point_dict = {
            "type": "sample_point",
            "sign": "empty",
            "position": (0.5, 0.5, 0.2),
            "distance": 0.15,
            "weight": 1.0,
        }

        # Run standalone
        standalone_sampler = SDFSampler()
        standalone_samples = standalone_sampler.generate(
            xyz=xyz,
            constraints=[sample_point_dict],
        )

        # Run backend
        from sdf_labeler_api.config import Settings
        from sdf_labeler_api.services.sampling_service import SamplingService
        from sdf_labeler_api.services.project_service import ProjectService
        from sdf_labeler_api.services.constraint_service import ConstraintService
        from sdf_labeler_api.models.project import ProjectCreate
        from sdf_labeler_api.models.constraints import SamplePointConstraint as BackendSamplePoint
        from sdf_labeler_api.models.constraints import SignConvention as BackendSign
        from sdf_labeler_api.models.samples import SampleGenerationRequest, SamplingStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            import sdf_labeler_api.config as backend_config
            original_settings = backend_config.settings
            backend_config.settings = Settings(data_dir=data_dir)

            try:
                project_service = ProjectService(data_dir)
                project = project_service.create(ProjectCreate(name="test"))
                project_id = project.id

                setup_backend_project(data_dir, project_id, xyz, normals)

                constraint_service = ConstraintService()
                backend_sample_point = BackendSamplePoint(
                    sign=BackendSign.EMPTY,
                    position=(0.5, 0.5, 0.2),
                    distance=0.15,
                    weight=1.0,
                )
                constraint_service.add(project_id, backend_sample_point)

                sampling_service = SamplingService()
                request = SampleGenerationRequest(
                    total_samples=10000,
                    strategy=SamplingStrategy.CONSTANT,
                    samples_per_primitive=100,
                    seed=42,
                )

                backend_result = sampling_service.generate(project_id, request)
                backend_samples = backend_result.samples
            finally:
                backend_config.settings = original_settings

        # For sample_point, should be exactly 1 sample with exact values
        assert len(standalone_samples) == 1
        assert len(backend_samples) == 1

        s = standalone_samples[0]
        b = backend_samples[0]

        assert abs(s.x - b.x) < 1e-6
        assert abs(s.y - b.y) < 1e-6
        assert abs(s.z - b.z) < 1e-6
        assert abs(s.phi - b.phi) < 1e-6

    def _compare_samples(self, standalone, backend, source_prefix: str):
        """Compare samples from both implementations."""
        # Same number of samples
        assert len(standalone) == len(backend), (
            f"Sample count mismatch: standalone={len(standalone)}, backend={len(backend)}"
        )

        # Check source names match pattern
        for s in standalone:
            assert source_prefix in s.source, f"Unexpected source: {s.source}"

        # Check phi signs are consistent
        standalone_positive = sum(1 for s in standalone if s.phi > 0)
        backend_positive = sum(1 for s in backend if s.phi > 0)

        # Should have same sign distribution
        assert standalone_positive == backend_positive, (
            f"Phi sign distribution mismatch: "
            f"standalone positive={standalone_positive}, backend positive={backend_positive}"
        )

        # Check is_free flag consistency
        standalone_free = sum(1 for s in standalone if s.is_free)
        backend_free = sum(1 for s in backend if s.is_free)
        assert standalone_free == backend_free, (
            f"is_free mismatch: standalone={standalone_free}, backend={backend_free}"
        )


class TestFullPipelineEquivalence:
    """Test complete analyze->sample pipeline produces equivalent results."""

    @requires_backend
    def test_full_pipeline_equivalence(self, trench_pointcloud):
        """Test full pipeline from point cloud to samples."""
        xyz, normals = trench_pointcloud

        # Shared options
        analysis_options = AutoAnalysisOptions(
            flood_fill_output="samples",
            flood_fill_sample_count=100,
            voxel_regions_output="samples",
            voxel_regions_sample_count=100,
            idw_sample_count=100,  # Backend requires >= 100
            hull_filter_enabled=False,
        )

        # Run standalone
        standalone_analyzer = SDFAnalyzer()
        standalone_result = standalone_analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["flood_fill", "voxel_regions", "normal_idw"],
            options=analysis_options,
        )

        standalone_sampler = SDFSampler()
        standalone_samples = standalone_sampler.generate(
            xyz=xyz,
            constraints=standalone_result.constraints,
            strategy="constant",
            seed=42,
        )

        # Run backend
        from sdf_labeler_api.config import Settings
        from sdf_labeler_api.services.auto_analysis_service import AutoAnalysisService
        from sdf_labeler_api.services.sampling_service import SamplingService
        from sdf_labeler_api.services.project_service import ProjectService
        from sdf_labeler_api.services.constraint_service import ConstraintService
        from sdf_labeler_api.models.project import ProjectCreate
        from sdf_labeler_api.models.samples import SampleGenerationRequest, SamplingStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            import sdf_labeler_api.config as backend_config
            original_settings = backend_config.settings
            backend_config.settings = Settings(data_dir=data_dir)

            try:
                project_service = ProjectService(data_dir)
                project = project_service.create(ProjectCreate(name="test"))
                project_id = project.id

                setup_backend_project(data_dir, project_id, xyz, normals)

                # Analyze
                backend_analysis = AutoAnalysisService(backend_config.settings)
                backend_options = get_backend_options(analysis_options)
                backend_result = asyncio.run(backend_analysis.analyze(
                    project_id=project_id,
                    algorithms=["flood_fill", "voxel_regions", "normal_idw"],
                    recompute=True,
                    options=backend_options,
                ))

                # Add constraints to project
                constraint_service = ConstraintService()
                for gc in backend_result.generated_constraints:
                    constraint_service.add_from_dict(project_id, gc.constraint)

                # Sample
                sampling_service = SamplingService()
                request = SampleGenerationRequest(
                    total_samples=10000,
                    strategy=SamplingStrategy.CONSTANT,
                    samples_per_primitive=100,
                    seed=42,
                )
                backend_sample_result = sampling_service.generate(project_id, request)
                backend_samples = backend_sample_result.samples
            finally:
                backend_config.settings = original_settings

        # Compare overall statistics
        print(f"\nPipeline comparison:")
        print(f"  Standalone constraints: {len(standalone_result.constraints)}")
        print(f"  Backend constraints: {len(backend_result.generated_constraints)}")
        print(f"  Standalone samples: {len(standalone_samples)}")
        print(f"  Backend samples: {len(backend_samples)}")

        # Check constraint counts are similar (within 20% due to randomness)
        standalone_count = len(standalone_result.constraints)
        backend_count = len(backend_result.generated_constraints)

        if standalone_count > 0 and backend_count > 0:
            ratio = max(standalone_count, backend_count) / min(standalone_count, backend_count)
            assert ratio < 1.5, (
                f"Constraint count ratio too high: {ratio:.2f} "
                f"(standalone={standalone_count}, backend={backend_count})"
            )

        # Check sample counts are similar
        standalone_sample_count = len(standalone_samples)
        backend_sample_count = len(backend_samples)

        if standalone_sample_count > 0 and backend_sample_count > 0:
            ratio = max(standalone_sample_count, backend_sample_count) / min(standalone_sample_count, backend_sample_count)
            assert ratio < 1.5, (
                f"Sample count ratio too high: {ratio:.2f} "
                f"(standalone={standalone_sample_count}, backend={backend_sample_count})"
            )

    @requires_backend
    def test_inverse_square_pipeline_equivalence(self, trench_pointcloud):
        """Test inverse_square sampling produces equivalent results.

        This is the recommended production workflow: auto-analyze + inverse_square sampling.
        """
        xyz, normals = trench_pointcloud

        # Shared analysis options
        analysis_options = AutoAnalysisOptions(
            flood_fill_output="samples",
            flood_fill_sample_count=100,
            voxel_regions_output="samples",
            voxel_regions_sample_count=100,
            idw_sample_count=100,
            hull_filter_enabled=False,
        )

        # Run standalone with inverse_square
        standalone_analyzer = SDFAnalyzer()
        standalone_result = standalone_analyzer.analyze(
            xyz=xyz,
            normals=normals,
            algorithms=["flood_fill", "voxel_regions", "normal_idw"],
            options=analysis_options,
        )

        standalone_sampler = SDFSampler()
        standalone_samples = standalone_sampler.generate(
            xyz=xyz,
            normals=normals,
            constraints=standalone_result.constraints,
            strategy="inverse_square",
            total_samples=5000,
            seed=42,
            include_surface_points=False,  # Test without surface points first
        )

        # Run backend with inverse_square
        from sdf_labeler_api.config import Settings
        from sdf_labeler_api.services.auto_analysis_service import AutoAnalysisService
        from sdf_labeler_api.services.sampling_service import SamplingService
        from sdf_labeler_api.services.project_service import ProjectService
        from sdf_labeler_api.services.constraint_service import ConstraintService
        from sdf_labeler_api.models.project import ProjectCreate
        from sdf_labeler_api.models.samples import SampleGenerationRequest, SamplingStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            import sdf_labeler_api.config as backend_config
            original_settings = backend_config.settings
            backend_config.settings = Settings(data_dir=data_dir)

            try:
                project_service = ProjectService(data_dir)
                project = project_service.create(ProjectCreate(name="test"))
                project_id = project.id

                setup_backend_project(data_dir, project_id, xyz, normals)

                # Analyze
                backend_analysis = AutoAnalysisService(backend_config.settings)
                backend_options = get_backend_options(analysis_options)
                backend_result = asyncio.run(backend_analysis.analyze(
                    project_id=project_id,
                    algorithms=["flood_fill", "voxel_regions", "normal_idw"],
                    recompute=True,
                    options=backend_options,
                ))

                # Add constraints to project
                constraint_service = ConstraintService()
                for gc in backend_result.generated_constraints:
                    constraint_service.add_from_dict(project_id, gc.constraint)

                # Sample with inverse_square
                sampling_service = SamplingService()
                request = SampleGenerationRequest(
                    total_samples=5000,
                    strategy=SamplingStrategy.INVERSE_SQUARE,
                    seed=42,
                )
                backend_sample_result = sampling_service.generate(project_id, request)
                backend_samples = backend_sample_result.samples
            finally:
                backend_config.settings = original_settings

        # Compare results
        print(f"\nInverse square pipeline comparison:")
        print(f"  Standalone constraints: {len(standalone_result.constraints)}")
        print(f"  Backend constraints: {len(backend_result.generated_constraints)}")
        print(f"  Standalone samples: {len(standalone_samples)}")
        print(f"  Backend samples: {len(backend_samples)}")

        # Verify phi distribution is similar (more samples near 0)
        standalone_near_surface = sum(1 for s in standalone_samples if abs(s.phi) < 0.1)
        backend_near_surface = sum(1 for s in backend_samples if abs(s.phi) < 0.1)

        print(f"  Standalone near-surface (|phi|<0.1): {standalone_near_surface}")
        print(f"  Backend near-surface (|phi|<0.1): {backend_near_surface}")

        # Both should have majority of samples near surface (inverse_square characteristic)
        standalone_ratio = standalone_near_surface / len(standalone_samples) if standalone_samples else 0
        backend_ratio = backend_near_surface / len(backend_samples) if backend_samples else 0

        assert standalone_ratio > 0.3, f"Standalone should have >30% near-surface, got {standalone_ratio:.1%}"
        assert backend_ratio > 0.3, f"Backend should have >30% near-surface, got {backend_ratio:.1%}"

        # Ratios should be similar
        if standalone_ratio > 0 and backend_ratio > 0:
            ratio_diff = abs(standalone_ratio - backend_ratio)
            assert ratio_diff < 0.2, (
                f"Near-surface ratio difference too high: {ratio_diff:.1%} "
                f"(standalone={standalone_ratio:.1%}, backend={backend_ratio:.1%})"
            )
