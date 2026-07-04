import numpy as np
from numba import njit, prange


@njit
def signum(x):
    return (x > 0) - (x < 0)


@njit
def raycast_range_filtered_first_hit(start, direction, occupancy_grid, voxel_size, min_dist=0.0):
    """
    Parameters:
        start: (x, y, z) float32
        direction: (dx, dy, dz) normalized float32
        occupancy_grid: 3D binary mask (e.g., numpy array)
        voxel_size: scalar size of each voxel
        min_dist: minimum distance to the first occupied voxel
    Returns:
        (i, j, k) voxel indices of the first occupied voxel intersected by the ray
        distance to the first occupied voxel's surface
    """

    # Compute voxel coordinates of the starting point
    x, y, z = start
    dx, dy, dz = direction

    grid_shape = occupancy_grid.shape

    ix, iy, iz = int(x // voxel_size), int(y // voxel_size), int(z // voxel_size)

    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1

    t_max_x = ((ix + (step_x > 0)) * voxel_size - x) / dx if dx != 0 else float('inf')
    t_max_y = ((iy + (step_y > 0)) * voxel_size - y) / dy if dy != 0 else float('inf')
    t_max_z = ((iz + (step_z > 0)) * voxel_size - z) / dz if dz != 0 else float('inf')

    t_delta_x = voxel_size / abs(dx) if dx != 0 else float('inf')
    t_delta_y = voxel_size / abs(dy) if dy != 0 else float('inf')
    t_delta_z = voxel_size / abs(dz) if dz != 0 else float('inf')
    
    t = 0.0

    while 0 <= ix < grid_shape[0] and 0 <= iy < grid_shape[1] and 0 <= iz < grid_shape[2]:
        if occupancy_grid[ix, iy, iz] and np.linalg.norm(start - np.array([ix, iy, iz]) * voxel_size) > min_dist:
            return np.array([ix, iy, iz], dtype=np.int32), t

        # Move to next voxel
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                ix += step_x
                t = t_max_x
                t_max_x += t_delta_x
            else:
                iz += step_z
                t = t_max_z
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                iy += step_y
                t = t_max_y
                t_max_y += t_delta_y
            else:
                iz += step_z
                t = t_max_z
                t_max_z += t_delta_z

    return np.array([-1, -1, -1], dtype=np.int32), -1.0  # No intersection found


@njit(parallel=True)
def batch_range_filtered_raycast_first_hit(starts, directions, occupancy_grid, voxel_size, min_dist=0.0):
    """
    Parameters:
        starts: (N, 3) float32 array of starting points
        directions: (N, 3) float32 array of normalized directions
        occupancy_grid: 3D binary mask (e.g., numpy array)
        voxel_size: scalar size of each voxel
        min_dist: minimum distance to the first occupied voxel
    Returns:
        (N, 3) array of voxel indices of the first occupied voxel intersected by the rays
        (N,) array of distances to the first occupied voxel's surface
    """
    num_rays = starts.shape[0]
    first_hits = np.empty((num_rays, 3), dtype=np.int32)
    distances = np.empty(num_rays, dtype=np.float32)

    for i in prange(num_rays):
        first_hit, dist = raycast_range_filtered_first_hit(starts[i], directions[i], occupancy_grid, voxel_size, min_dist=min_dist)
        first_hits[i] = first_hit
        distances[i] = dist
        
    return first_hits, distances
