import math
import numpy as np
from scripts import Molecule, parse_xyz_coords, voxelize

# # Test
traj_file = "./examples/torus_2000mer.cus"
num_atoms = 2000
box_dim = [198, 198, 198]
traj_type = "cus"
frame_to_read = 0
frames, xyzcoords = parse_xyz_coords(traj_type=traj_type, traj_file=traj_file,
                                     num_atoms=num_atoms, box_dim = box_dim)
mol = Molecule(num_atoms=num_atoms, frames=frames, xyzcoords=xyzcoords)

coords = mol.get_xyz_coords(frame=frame_to_read)
print(coords)