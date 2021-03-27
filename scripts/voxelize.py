import math
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union

def voxelize(coords: Union[List[float], np.ndarray],
             voxel_res: int,
             max_global: float,
             min_global: float) -> np.ndarray:
    """
    Creates a cubic voxel array

    Args:
        coords:
        voxel_res: Number of voxels per dimension
        max_global: Maximum dimension
        min_global: Minimum dimension

    Returns:
        voxel_array
    """
    voxel_length = (max_global - min_global)/(voxel_res -1)
    voxel_array = np.zeros(voxel_res, voxel_res, voxel_res)
    for i in range(len(coords)): # loop over all points
        point_x = (coords[i][1] - min_global)/(voxel_length)
        point_y = (coords[i][2] - min_global)/(voxel_length)
        point_z = (coords[i][3] - min_global)/(voxel_length)
        x_index = math.ceil(point_x) + 1
        y_index = math.ceil(point_y) + 1
        z_index = math.ceil(point_z) + 1
        # voxel_array[x_index][y_index][z_index] = voxel_array[x_index][y_index][z_index] + 1
        voxel_array[x_index][y_index][z_index]= 1 # logical array