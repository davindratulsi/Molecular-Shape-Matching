import math
import numpy as np
from scripts import Molecule, parse_xyz_coords, voxelize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# # Test
def get_molecule_voxel_array(traj_file):
    box_dim = [198, 198, 198]
    traj_type = "cus"
    frames = 201
    frame_to_read = 0
    num_atoms, xyzcoords = parse_xyz_coords(traj_type=traj_type, traj_file=traj_file,
                                                    type_list=[1, 2], frames=frames,
                                                    box_dim=box_dim)
    mol = Molecule(num_atoms=num_atoms, frames=frames, xyzcoords=xyzcoords)
    # # voxelize
    coords = mol.get_xyz_coords(frame=frame_to_read)
    xdata = [coord[0] for coord in coords]
    ydata = [coord[1] for coord in coords]
    zdata = [coord[2] for coord in coords]
    max_global = max([max(xdata), max(ydata), max(zdata)]) + 1
    min_global = min([min(xdata), min(ydata), min(zdata)]) - 1
    voxels = voxelize(coords=coords, voxel_res=100,
                    min_global=min_global, max_global=max_global)
    return voxels

# # Voxelize
traj_file = './examples/torus_2000mer.cus'
voxels = get_molecule_voxel_array(traj_file)

# # Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.voxels(voxels, edgecolor="k")
plt.show()

