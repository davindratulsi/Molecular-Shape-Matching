import math
import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union


def parse_xyz_coords(traj_file: str,
                     traj_type: str,
                     num_atoms: int,
                     box_dim: List[float]) -> Tuple[int, np.ndarray]:
    """
    Parses xyz coordinates and number of frames
    from a .xyz or .cus file

    Args:
        traj_file: Location of trajectory file of interest
        traj_type: Trajectory type - currently support ``cus`` or ``xyz``
        num_atoms: Number of atoms/beads in system
        box_dim: x,y,z dimensions of periodic box defining system

    Returns:
        (frames, xyzcoords): int, number of frames in trajectory
                             and 3D array with shape (frames, num_atoms, 4)
                             Last dimension is of form [atom_type, x, y, z]
    """
    # parse file and retrieve all data excluding headers
    raw_coords = []
    with open(traj_file) as file_in:
        if traj_type == 'xyz':
            for line in file_in:
                if not line.startswith("Atoms") and not line.startswith(str(num_atoms)):
                    row = line.split()
                    coord = np.array([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
                    raw_coords.append(coord)
        elif traj_type == 'cus':
            for line in file_in:
                if not line.startswith('ITEM: ATOMS') and len(line.split()) > 6:
                    row = line.split()
                    type = float(row[1])
                    x, y, z = float(row[2]), float(row[3]), float(row[4])
                    # get true coords based on image flags
                    ix, iy, iz = float(row[-1]), float(row[-2]), float(row[-3])
                    x = x + ix*box_dim[0]
                    y = y + iy*box_dim[1]
                    z = z + iz*box_dim[2]
                    coord = np.array([type, x, y, z])
                    raw_coords.append(coord)

    # Process raw coordinates
    frames = int(len(raw_coords)/num_atoms)
    # 3D array with shape (frames, num_atoms, 4)
    xyzcoords = np.zeros((frames, num_atoms, 4))
    for frame in range(frames):
        frame_coords = raw_coords[frame: frame + num_atoms]
        xyzcoords[frame] = frame_coords
    return frames, xyzcoords