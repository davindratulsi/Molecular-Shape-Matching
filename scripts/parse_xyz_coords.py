import math
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union

def parse_xyz_coords(traj_file: str,
                     traj_type: str,
                     type_list: List[int],
                     frames: int,
                     box_dim: List[float]) -> Tuple[int, np.ndarray]:
    """
    Parses xyz coordinates and number of frames
    from a .xyz or .cus file

    Args:
        traj_file: Location of trajectory file of interest
        traj_type: Trajectory type - currently support ``cus`` or ``xyz``
        type_list: List of atom types to consider for parsing
        frames: Number of frames in trajectory
        box_dim: x, y, z dimensions of periodic box defining system

    Returns:
        (num_types, raw_coords): number of atoms,
                                 and numpy array of shape (num_atoms, 3)
                                 Last dimension is of form [x, y, z]
    """
    # # parse file and retrieve all data excluding headers
    raw_coords = []
    with open(traj_file) as file_in:
        if traj_type == 'xyz':
            for line in file_in:
                if len(line.split()) > 3 and int(line.split()[0]) in type_list:
                    row = line.split()
                    coord = [float(row[1]), float(row[2]), float(row[3])]
                    raw_coords.append(coord)
        elif traj_type == 'cus':
            for line in file_in:
                if not line.startswith('ITEM: ATOMS') and len(line.split()) > 6 \
                   and int(line.split()[1]) in type_list:
                    row = line.split()
                    x, y, z = float(row[2]), float(row[3]), float(row[4])
                    # get true coords based on image flags
                    ix, iy, iz = float(row[-1]), float(row[-2]), float(row[-3])
                    x = x + ix*box_dim[0]
                    y = y + iy*box_dim[1]
                    z = z + iz*box_dim[2]
                    coord = [x, y, z]
                    raw_coords.append(coord)

    # # Return coordinates and number of total atoms parsed
    raw_coords = np.array(raw_coords)
    num_atoms = int(len(raw_coords)/frames)
    return num_atoms, raw_coords

