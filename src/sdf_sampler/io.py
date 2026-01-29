# ABOUTME: I/O utilities for point cloud loading and sample export
# ABOUTME: Supports PLY, LAS/LAZ, CSV, NPZ, and Parquet formats

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sdf_sampler.models.samples import TrainingSample


def load_point_cloud(
    path: str | Path,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load point cloud from file.

    Supported formats:
    - PLY (requires trimesh in [io] extras)
    - LAS/LAZ (requires laspy in [io] extras)
    - CSV (columns: x, y, z, [nx, ny, nz])
    - NPZ (arrays: xyz, [normals])
    - Parquet (columns: x, y, z, [nx, ny, nz])

    Args:
        path: Path to point cloud file
        **kwargs: Additional arguments for specific loaders

    Returns:
        Tuple of (xyz, normals) where xyz is (N, 3) and normals is (N, 3) or None

    Example:
        >>> xyz, normals = load_point_cloud("scan.ply")
        >>> xyz, normals = load_point_cloud("points.csv")
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".ply":
        return _load_ply(path, **kwargs)
    elif suffix in (".las", ".laz"):
        return _load_las(path, **kwargs)
    elif suffix == ".csv":
        return _load_csv(path, **kwargs)
    elif suffix == ".npz":
        return _load_npz(path, **kwargs)
    elif suffix == ".parquet":
        return _load_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def export_parquet(
    samples: list[TrainingSample],
    path: str | Path,
) -> Path:
    """Export training samples to Parquet file.

    Creates a survi-compatible Parquet file with columns:
    x, y, z, phi, nx, ny, nz, weight, source, is_surface, is_free

    Args:
        samples: List of TrainingSample objects
        path: Output file path

    Returns:
        Path to created file

    Example:
        >>> export_parquet(samples, "training_data.parquet")
    """
    path = Path(path)
    df = pd.DataFrame([s.to_dict() for s in samples])
    df.to_parquet(path)
    return path


def _load_ply(path: Path, **kwargs: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Load PLY file using trimesh."""
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "trimesh is required for PLY support. "
            "Install with: pip install sdf-sampler[io]"
        ) from e

    mesh = trimesh.load(path, **kwargs)

    # Handle both PointCloud and Trimesh objects
    if hasattr(mesh, "vertices"):
        xyz = np.asarray(mesh.vertices)
    else:
        xyz = np.asarray(mesh.points if hasattr(mesh, "points") else mesh)

    normals = None
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        normals = np.asarray(mesh.vertex_normals)
        if normals.shape != xyz.shape:
            normals = None

    return xyz, normals


def _load_las(path: Path, **kwargs: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Load LAS/LAZ file using laspy."""
    try:
        import laspy
    except ImportError as e:
        raise ImportError(
            "laspy is required for LAS/LAZ support. "
            "Install with: pip install sdf-sampler[io]"
        ) from e

    las = laspy.read(path, **kwargs)

    xyz = np.column_stack([las.x, las.y, las.z])

    # LAS files typically don't have normals
    normals = None

    return xyz, normals


def _load_csv(path: Path, **kwargs: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Load CSV file with pandas."""
    df = pd.read_csv(path, **kwargs)

    # Required columns
    if not all(c in df.columns for c in ["x", "y", "z"]):
        raise ValueError("CSV must have x, y, z columns")

    xyz = df[["x", "y", "z"]].values

    # Optional normal columns
    normals = None
    if all(c in df.columns for c in ["nx", "ny", "nz"]):
        normals = df[["nx", "ny", "nz"]].values

    return xyz, normals


def _load_npz(path: Path, **kwargs: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Load NPZ file."""
    data = np.load(path, **kwargs)

    # Support different naming conventions
    if "xyz" in data:
        xyz = data["xyz"]
    elif "points" in data:
        xyz = data["points"]
    else:
        raise ValueError("NPZ must have 'xyz' or 'points' array")

    normals = None
    if "normals" in data and data["normals"].size > 0:
        normals = data["normals"]

    return xyz, normals


def _load_parquet(path: Path, **kwargs: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Load Parquet file with pandas."""
    df = pd.read_parquet(path, **kwargs)

    # Required columns
    if not all(c in df.columns for c in ["x", "y", "z"]):
        raise ValueError("Parquet must have x, y, z columns")

    xyz = df[["x", "y", "z"]].values

    # Optional normal columns
    normals = None
    if all(c in df.columns for c in ["nx", "ny", "nz"]):
        normals = df[["nx", "ny", "nz"]].values

    return xyz, normals
