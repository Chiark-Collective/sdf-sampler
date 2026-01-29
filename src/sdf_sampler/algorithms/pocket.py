# ABOUTME: Pocket detection algorithm for cavity identification
# ABOUTME: Uses voxel flood fill to find disconnected interior cavities

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy import ndimage

from sdf_sampler.config import AnalyzerConfig
from sdf_sampler.models.analysis import AlgorithmType, GeneratedConstraint
from sdf_sampler.models.constraints import SignConvention


class VoxelState(IntEnum):
    """State of a voxel in the occupancy grid."""

    EMPTY = 0
    OCCUPIED = 1
    OUTSIDE = 2


@dataclass
class PocketInfo:
    """Information about a detected pocket."""

    pocket_id: int
    voxel_count: int
    centroid: tuple[float, float, float]
    bounds_low: tuple[float, float, float]
    bounds_high: tuple[float, float, float]
    volume_estimate: float
    voxel_coords: np.ndarray  # World coordinates of voxel centers


def detect_pockets(
    xyz: np.ndarray,
    config: AnalyzerConfig,
) -> list[GeneratedConstraint]:
    """Detect pocket cavities in point cloud using voxel analysis.

    Pockets are disconnected interior voids that should be marked as SOLID.

    Args:
        xyz: Point cloud positions (N, 3)
        config: Analyzer configuration

    Returns:
        List of PocketConstraint GeneratedConstraints
    """
    constraints: list[GeneratedConstraint] = []

    if len(xyz) < 10:
        return constraints

    # Compute bounds
    bounds_low = xyz.min(axis=0)
    bounds_high = xyz.max(axis=0)

    # Compute voxel size
    extent = bounds_high - bounds_low
    longest_axis = np.max(extent)
    voxel_size = longest_axis / config.pocket_voxel_target
    voxel_size = max(voxel_size, config.pocket_min_voxel_size)

    # Compute grid resolution - ensure at least 1 in each dimension
    resolution = np.ceil(extent / voxel_size).astype(int)
    resolution = np.maximum(resolution, 1)  # Avoid zero-size dimensions
    resolution = np.minimum(resolution, config.pocket_max_voxels_per_axis)

    # Build occupancy grid
    grid = np.full(tuple(resolution), VoxelState.EMPTY, dtype=np.uint8)

    voxel_indices = ((xyz - bounds_low) / voxel_size).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, resolution - 1)

    # Mark occupied with dilation
    if config.pocket_occupancy_dilation == 0:
        grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = VoxelState.OCCUPIED
    else:
        occupied_mask = np.zeros_like(grid, dtype=bool)
        occupied_mask[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
        struct = ndimage.generate_binary_structure(3, 1)
        dilated = ndimage.binary_dilation(
            occupied_mask, structure=struct, iterations=config.pocket_occupancy_dilation
        )
        grid[dilated] = VoxelState.OCCUPIED

    # Flood fill from boundary to mark outside
    boundary_mask = np.zeros_like(grid, dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True

    seed = boundary_mask & (grid == VoxelState.EMPTY)
    traversable = grid == VoxelState.EMPTY

    struct = ndimage.generate_binary_structure(3, 1)
    outside = ndimage.binary_dilation(seed, mask=traversable, iterations=-1, structure=struct)
    grid[outside] = VoxelState.OUTSIDE

    # Label remaining empty voxels as pockets
    pocket_mask = grid == VoxelState.EMPTY
    labeled, num_pockets = ndimage.label(pocket_mask, structure=struct)

    # Extract pocket info
    for pocket_id in range(1, num_pockets + 1):
        voxel_count = int(np.sum(labeled == pocket_id))
        if voxel_count < config.pocket_min_volume_voxels:
            continue

        mask = labeled == pocket_id
        voxel_coords = np.argwhere(mask)
        world_coords = voxel_coords * voxel_size + bounds_low + voxel_size / 2

        centroid = tuple(world_coords.mean(axis=0).tolist())
        pocket_bounds_low = tuple((voxel_coords.min(axis=0) * voxel_size + bounds_low).tolist())
        pocket_bounds_high = tuple(
            ((voxel_coords.max(axis=0) + 1) * voxel_size + bounds_low).tolist()
        )
        volume = voxel_count * (voxel_size**3)

        pocket_constraint = {
            "type": "pocket",
            "sign": SignConvention.SOLID.value,
            "pocket_id": pocket_id,
            "voxel_count": voxel_count,
            "centroid": centroid,
            "bounds_low": pocket_bounds_low,
            "bounds_high": pocket_bounds_high,
            "volume_estimate": volume,
        }

        constraints.append(
            GeneratedConstraint(
                constraint=pocket_constraint,
                algorithm=AlgorithmType.POCKET,
                confidence=0.95,
                description=f"Interior cavity at ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}), {voxel_count} voxels",
            )
        )

    return constraints
