# ABOUTME: sdf-sampler public API exports
# ABOUTME: Standalone SDF training data generation from point clouds

"""
sdf-sampler: Auto-analysis and sampling of point clouds for SDF training.

Example usage:
    from sdf_sampler import SDFAnalyzer, SDFSampler, load_point_cloud

    # Load point cloud
    xyz, normals = load_point_cloud("scan.ply")

    # Auto-analyze to detect EMPTY/SOLID regions
    analyzer = SDFAnalyzer()
    result = analyzer.analyze(xyz=xyz, normals=normals)

    # Generate training samples
    sampler = SDFSampler()
    samples = sampler.generate(
        xyz=xyz,
        normals=normals,
        constraints=result.constraints,
    )

    # Export to parquet
    sampler.export_parquet(samples, "training_data.parquet")
"""

from sdf_sampler.analyzer import SDFAnalyzer
from sdf_sampler.config import AnalyzerConfig, SamplerConfig
from sdf_sampler.io import Mesh, export_parquet, load_mesh, load_point_cloud
from sdf_sampler.models import (
    AlgorithmType,
    AnalysisResult,
    BoxConstraint,
    BrushStrokeConstraint,
    Constraint,
    HalfspaceConstraint,
    PocketConstraint,
    RayCarveConstraint,
    SamplePointConstraint,
    SamplingStrategy,
    SeedPropagationConstraint,
    SignConvention,
    SphereConstraint,
    TrainingSample,
)
from sdf_sampler.sampler import SDFSampler

__version__ = "0.5.0"

__all__ = [
    # Main classes
    "SDFAnalyzer",
    "SDFSampler",
    # Config
    "AnalyzerConfig",
    "SamplerConfig",
    # I/O
    "load_point_cloud",
    "load_mesh",
    "Mesh",
    "export_parquet",
    # Models
    "SignConvention",
    "AlgorithmType",
    "SamplingStrategy",
    "AnalysisResult",
    "TrainingSample",
    # Constraints
    "Constraint",
    "BoxConstraint",
    "SphereConstraint",
    "HalfspaceConstraint",
    "BrushStrokeConstraint",
    "SeedPropagationConstraint",
    "RayCarveConstraint",
    "PocketConstraint",
    "SamplePointConstraint",
]
