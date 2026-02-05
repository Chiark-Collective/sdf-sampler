# ABOUTME: Voxel grid construction utilities
# ABOUTME: Shared voxelization and hull computation for analysis algorithms

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull, Delaunay, KDTree


def estimate_mean_spacing(xyz: np.ndarray, tree: KDTree | None = None, k: int = 8) -> float:
    """Estimate mean point spacing using k-NN.

    Args:
        xyz: Point positions (N, 3)
        tree: Optional pre-built KDTree
        k: Number of neighbors to use

    Returns:
        Mean distance to k nearest neighbors
    """
    if tree is None:
        tree = KDTree(xyz)

    n_sample = min(1000, len(xyz))
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(xyz), n_sample, replace=False)

    distances = []
    for idx in sample_indices:
        dists, _ = tree.query(xyz[idx], k=k + 1)  # +1 for self
        distances.extend(dists[1:])  # Exclude self (distance 0)

    return float(np.mean(distances))


def build_voxel_grid(
    xyz: np.ndarray,
    min_gap_size: float = 0.10,
    max_dim: int = 200,
    voxel_size: float | None = None,
    z_extension: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int, int]] | None:
    """Build a voxel grid from point cloud data.

    Args:
        xyz: Point cloud coordinates (N, 3)
        min_gap_size: Minimum gap size flood fill should traverse
        max_dim: Maximum voxel grid dimension
        voxel_size: Optional voxel size (auto-computed if None)
        z_extension: How much to extend grid above point cloud in +Z

    Returns:
        Tuple of (occupied_grid, bbox_min, voxel_size, grid_shape) or None if invalid.
    """
    if len(xyz) < 10:
        return None

    # Determine voxel size based on point cloud density
    tree = KDTree(xyz)
    mean_spacing = estimate_mean_spacing(xyz, tree)

    if voxel_size is None:
        # Voxel size based on point density, constrained by min_gap_size
        density_based = mean_spacing * 2.0
        gap_based = min_gap_size / 3.0
        voxel_size = min(density_based, gap_based)

    voxel_size_float: float = float(voxel_size)
    if voxel_size_float <= 0 or not np.isfinite(voxel_size_float):
        return None

    voxel_size = voxel_size_float

    # Compute bounding box with padding
    bbox_min = xyz.min(axis=0) - voxel_size
    bbox_max = xyz.max(axis=0) + voxel_size

    # Extend in +Z direction for sky space (outdoor scenes)
    if z_extension is None:
        z_range = xyz[:, 2].max() - xyz[:, 2].min()
        z_extension = max(z_range * 0.5, voxel_size * 5)
    bbox_max[2] += z_extension
    bbox_min[2] -= voxel_size * 5  # Small extension for underground
    bbox_size = bbox_max - bbox_min

    if np.any(bbox_size <= 0) or not np.all(np.isfinite(bbox_size)):
        return None

    grid_shape = np.ceil(bbox_size / voxel_size).astype(int)

    if np.any(grid_shape <= 0):
        return None

    # Cap grid size for performance
    vs: float = voxel_size
    if grid_shape.max() > max_dim:
        scale = float(max_dim / grid_shape.max())
        vs = vs / scale
        grid_shape = np.ceil(bbox_size / vs).astype(int)
        grid_shape = np.minimum(grid_shape, max_dim)

    # Mark occupied voxels
    point_voxel_indices = ((xyz - bbox_min) / vs).astype(int)
    point_voxel_indices = np.clip(point_voxel_indices, 0, grid_shape - 1)

    occupied = np.zeros(tuple(grid_shape), dtype=bool)
    for idx in point_voxel_indices:
        occupied[tuple(idx)] = True

    # Dilate to ensure surface blocks flood fill
    structure = np.ones((3, 3, 3), dtype=bool)
    occupied = binary_dilation(occupied, structure, iterations=1)

    shape_tuple: tuple[int, int, int] = (
        int(grid_shape[0]),
        int(grid_shape[1]),
        int(grid_shape[2]),
    )
    return occupied, bbox_min, vs, shape_tuple


def compute_hull_mask(
    xyz: np.ndarray,
    bbox_min: np.ndarray,
    voxel_size: float,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    """Compute a 2D mask of which XY voxel positions are inside the convex hull.

    Args:
        xyz: Point cloud coordinates
        bbox_min: Bounding box minimum
        voxel_size: Size of each voxel
        grid_shape: Shape of the voxel grid

    Returns:
        2D boolean array (nx, ny) where True = inside the XY convex hull.
    """
    nx, ny, _nz = grid_shape

    xy_points = xyz[:, :2]

    try:
        hull = ConvexHull(xy_points)
        hull_delaunay = Delaunay(xy_points[hull.vertices])
    except Exception:
        return np.ones((nx, ny), dtype=bool)

    inside_hull = np.zeros((nx, ny), dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            world_x = bbox_min[0] + (ix + 0.5) * voxel_size
            world_y = bbox_min[1] + (iy + 0.5) * voxel_size
            inside_hull[ix, iy] = hull_delaunay.find_simplex([world_x, world_y]) >= 0

    return inside_hull


def ray_propagation_with_bounces(
    occupied: np.ndarray,
    grid_shape: tuple[int, int, int],
    inside_hull: np.ndarray,
    cone_angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate EMPTY/SOLID using ray model with cone angles and flood fill.

    Physical model:
    1. EMPTY rays shine from +Z (sky) in a cone, then flood-fill from seeds
    2. SOLID rays shine from -Z (underground) in a cone, then flood-fill
    3. EMPTY has priority - SOLID flood-fill never overwrites EMPTY

    Args:
        occupied: Boolean grid of occupied voxels
        grid_shape: Grid dimensions
        inside_hull: 2D mask of XY positions inside hull
        cone_angle_degrees: Half-angle of the cone

    Returns:
        Tuple of (empty_mask, solid_mask) boolean arrays.
    """
    from scipy.ndimage import label

    nx, ny, nz = grid_shape
    empty = np.zeros(grid_shape, dtype=bool)
    solid = np.zeros(grid_shape, dtype=bool)

    # Phase 1: Rays from multiple angles within cone
    tan_angle = np.tan(np.radians(cone_angle_degrees))
    diag = tan_angle * 0.707

    ray_tilts = [
        (0.0, 0.0),
        (tan_angle, 0.0),
        (-tan_angle, 0.0),
        (0.0, tan_angle),
        (0.0, -tan_angle),
        (diag, diag),
        (diag, -diag),
        (-diag, diag),
        (-diag, -diag),
    ]

    # EMPTY rays from sky (top-down with cone)
    for dx_rate, dy_rate in ray_tilts:
        for start_ix in range(nx):
            for start_iy in range(ny):
                fx, fy = float(start_ix), float(start_iy)
                for iz in range(nz - 1, -1, -1):
                    ix, iy = int(round(fx)), int(round(fy))
                    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
                        break
                    if occupied[ix, iy, iz]:
                        break
                    empty[ix, iy, iz] = True
                    fx += dx_rate
                    fy += dy_rate

    # SOLID rays from underground (bottom-up with cone), only inside hull
    for dx_rate, dy_rate in ray_tilts:
        for start_ix in range(nx):
            for start_iy in range(ny):
                if not inside_hull[start_ix, start_iy]:
                    continue
                fx, fy = float(start_ix), float(start_iy)
                for iz in range(nz):
                    ix, iy = int(round(fx)), int(round(fy))
                    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
                        break
                    if occupied[ix, iy, iz]:
                        break
                    if not empty[ix, iy, iz]:
                        solid[ix, iy, iz] = True
                    fx += dx_rate
                    fy += dy_rate

    # Phase 2: Full flood fill from seeds
    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    # Compute per-column floor
    column_floor = np.full((nx, ny), -1, dtype=int)
    for ix in range(nx):
        for iy in range(ny):
            occupied_z = np.where(occupied[ix, iy, :])[0]
            if len(occupied_z) > 0:
                column_floor[ix, iy] = occupied_z.min()

    # Flood fill EMPTY
    empty_stack = [tuple(coord) for coord in np.argwhere(empty)]
    while empty_stack:
        ix, iy, iz = empty_stack.pop()
        for dx, dy, dz in directions:
            nx_, ny_, nz_ = ix + dx, iy + dy, iz + dz
            if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
                floor_z = column_floor[nx_, ny_]
                if floor_z >= 0 and nz_ < floor_z:
                    continue
                if not occupied[nx_, ny_, nz_] and not empty[nx_, ny_, nz_]:
                    empty[nx_, ny_, nz_] = True
                    empty_stack.append((nx_, ny_, nz_))

    # Filter EMPTY by sky connectivity
    labeled_empty, num_components = label(empty)
    if num_components > 0:
        top_slice = labeled_empty[:, :, -1]
        sky_labels = set(top_slice[top_slice > 0])
        if sky_labels:
            sky_connected = np.isin(labeled_empty, list(sky_labels))
            empty = empty & sky_connected

    # Remove small isolated EMPTY regions
    labeled_empty, num_components = label(empty)
    if num_components > 0:
        component_sizes = np.bincount(labeled_empty.ravel())
        min_component_voxels = max(10, (nx * ny) // 100)
        large_enough = component_sizes >= min_component_voxels
        large_enough[0] = False
        keep_mask = large_enough[labeled_empty]
        empty = empty & keep_mask

    # Flood fill SOLID
    solid_stack = [tuple(coord) for coord in np.argwhere(solid)]
    while solid_stack:
        ix, iy, iz = solid_stack.pop()
        for dx, dy, dz in directions:
            nx_, ny_, nz_ = ix + dx, iy + dy, iz + dz
            if 0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz:
                if (
                    not occupied[nx_, ny_, nz_]
                    and not empty[nx_, ny_, nz_]
                    and not solid[nx_, ny_, nz_]
                ):
                    solid[nx_, ny_, nz_] = True
                    solid_stack.append((nx_, ny_, nz_))

    return empty, solid


def greedy_2d_mesh(mask_2d: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Decompose a 2D boolean mask into axis-aligned rectangles.

    Uses a greedy algorithm: find first True voxel, expand as far as
    possible in X, then Y, record rectangle, clear voxels, repeat.

    Args:
        mask_2d: 2D boolean array to decompose

    Returns:
        List of (x_min, x_max, y_min, y_max) rectangles (exclusive max).
    """
    mask = mask_2d.copy()
    boxes: list[tuple[int, int, int, int]] = []

    while mask.any():
        coords = np.argwhere(mask)
        if len(coords) == 0:
            break
        x, y = coords[0]

        x_max = x
        while x_max + 1 < mask.shape[0] and mask[x_max + 1, y]:
            x_max += 1

        y_max = y
        while y_max + 1 < mask.shape[1]:
            if mask[x : x_max + 1, y_max + 1].all():
                y_max += 1
            else:
                break

        boxes.append((x, x_max + 1, y, y_max + 1))
        mask[x : x_max + 1, y : y_max + 1] = False

    return boxes
