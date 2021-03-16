import math
import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from typing import List, Set, Dict, Tuple, Optional, Union

def parse_xyz_coords(traj_file: str,
                     traj_type: str,
                     num_atoms: int,
                     box_len: int) -> Tuple(int, np.ndarray):
    """
    Parses xyz coordinates and number of frames
    from a .xyz or .cus file

    Args:
        traj_file: Location of trajectory file of interest
        traj_type: Trajectory type - currently support ``cus`` or ``xyz``
        num_atoms: Number of atoms/beads in system
        box_len: Box length of periodic system - cubic systems only

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
                    x = x + ix*box_len
                    y = y + iy*box_len
                    z = z + iz*box_len
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


class Molecule(object):
    """
    Parent class for a molecule simulated through MD
    """
    def __init__(self,
                 num_atoms: int,
                 xyzcoords: np.ndarray,
                 frames: int) -> None:
        """
        Args:
            num_atoms: Number of atoms/beads in trajectory
            xyzcoords: Array specifying the coordinates for all
                       atoms/beads over all timesteps/frames
                       Must be a three-dimensional numpy array
                       of shape (frames, num_atoms, 4) where
                       the last dimension is of the form,
                       [atom_type, x, y, z]
            frames: Number of frames in trajectory

        Raises:
            ValueError: If xyzcoords is not of shape (frames, num_atoms, 4)

        """
        # molecule attributes
        self.num_atoms = num_atoms
        self.frames = frames
        if xyzcoords.shape == (frames, num_atoms, 4):
            self.xyzcoords = xyzcoords
        else:
            raise ValueError("Shape of xyzcoords must be (frames, num_atoms, 4)")

    @property
    def get_number_frames(self) -> int:
        """Returns number of trajectory frames"""
        return self.frames

    def get_xyz_coords(self, frame: int) -> np.ndarray:
        """
        Retrieve xyz coordinates for a given frame

        Args:
            frame: frame in trajectory

        Returns:
            A numpy array of shape (num_atoms, 4) representing
            the trajectory for a given frame.
        """
        return self.xyzcoords[frame]

    def get_center_of_mass(self, frame: int) -> List[float]:
        """
        Method to calculate center of mass of a given trajectory frame

        Args:
            frame: frame in trajectory

        Returns:
            com: List corresponding to xyz coordinates of center of mass
        """
        xmom = 0
        ymom = 0
        zmom = 0
        for i in range(self.num_atoms):
            xmom += self.get_xyz_coords(frame)[i][1]
            ymom += self.get_xyz_coords(frame)[i][2]
            zmom += self.get_xyz_coords(frame)[i][3]
        com = [xmom/self.num_atoms, ymom/self.num_atoms, zmom/self.num_atoms]
        return com

    def get_rg2_tensor(self, frame: int) -> np.ndarray:
        """
        Calculate the squared radius of gyration tensor of a given trajectory frame
        This method assumes that all atoms/beads have the same mass

        Args:
            frame: frame in trajectory

        Returns:
            rg_2_tensor: Numpy array representing the squared radius of gyration
                         tensor for a given trajectory frame
        """
        coords = self.get_xyz_coords(frame)
        center_of_mass = self.get_center_of_mass(frame)
        r_xx = 0
        r_yy = 0
        r_zz = 0
        r_xy = 0
        r_xz = 0
        r_yz = 0
        for i in range(self.num_atoms):
            # (Rij)^2 = (1/N)*summation(Rij - Rcom)^2
            r_x = coords[i][1] - center_of_mass[0]
            r_y = coords[i][2] - center_of_mass[1]
            r_z = coords[i][3] - center_of_mass[2]
            r_xx += r_x**2
            r_yy += r_y**2
            r_zz += r_z**2
            r_xy += r_x*r_y
            r_xz += r_x*r_z
            r_yz += r_y*r_z
        rg_2_tensor = np.array([[r_xx, r_xy, r_xz], [r_xy, r_yy, r_yz], [r_xz, r_yz, r_zz]])
        rg_2_tensor = rg_2_tensor/self.num_atoms # normalize by number of atoms
        return rg_2_tensor

    def get_eigs(self, frame: int) -> Tuple(np.ndarray, np.ndarray):
        """
        Calculate the eigenvalues and eigenvectors of the squared
        radius of gyration tensor

        Args:
            frame: frame in trajectory

        Returns
            eigs, eigvectors: Tuple consisting of the three eigenvalues,
                              and corresponding eigenvectors from the
                              squared radius of gyration tensor of a
                              given frame
        """
        tensor = self.get_rg2_tensor(frame)
        eigs, eigvectors = LA.eig(tensor)
        return sorted(eigs), eigvectors

    def scale_coordinates(self,
                          frame: int,
                          rg2_target: float) -> np.ndarray:
        """
        Scale coordinates of a given frame based on a target Rg^2
        Scaling is done based on the following:
        (rg2_target)^2 = lambda*(rg_2_molecule)^2
        Thus for each coordinate of the molecule,
        lambda*(xi-com)^2 = (xi_scaled-com)^2

        Args:
            frame: frame in trajectory
            rg2_target: target squared radius of gyration

        Returns:
            coords: Numpy array of shape (num_atoms, 4) representing
                    the scaled coordinates for the given frame.

        """
        coords = self.get_xyz_coords(frame)
        center_of_mass = self.get_center_of_mass(frame)
        eigvalues, _ = self.get_eigs(frame)
        rg_molecule = (eigvalues.sum)**0.5
        rg_target = (rg2_target)**0.5
        scale = rg_target/rg_molecule
        for i in range(self.num_atoms):
            coords[i][1] = scale*coords[i][1] + (1 - scale)*center_of_mass[0]
            coords[i][2] = scale*coords[i][2] + (1 - scale)*center_of_mass[1]
            coords[i][3] = scale*coords[i][3] + (1 - scale)*center_of_mass[2]
        return coords

    def translate_coordinates(self,
                              frame: int,
                              com_target: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Translate coordinates of a given frame based on a target center of mass

        Args:
            frame: frame in trajectory
            com_target: List representing xyz coordinate of a target center of mass

        Returns:
            coords: Numpy array of shape (num_atoms, 4) representing
                    the translated coordinates for the given frame.
        """
        coords = self.get_xyz_coords(frame)
        com_molecule = self.get_center_of_mass(frame)
        shift = [tar - mol for tar, mol in zip(com_target, com_molecule)]
        for i in range(self.num_atoms):
            coords[i][1] = coords[i][1] + shift[0]
            coords[i][2] = coords[i][2] + shift[1]
            coords[i][3] = coords[i][3] + shift[2]
        return coords

    def orient_molecule(self,
                        frame: int,
                        basis: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Orient coordinates of a given frame to a specified basis
        Orientation is done based on the following:
        If M is a matrix whose columns are the vectors of the new basis,
        the new coordinates for a column vector, v, are given by the matrix product (inv(M))*v.

        Args:
            frame: frame in trajectory
            basis: List or array representing the new coordinate basis

        Returns:
            coords: Numpy array of shape (num_atoms, 4) representing
                    the oriented coordinates for the given frame.
        """
        matrix_rotation = LA.inv(basis)
        coords = np.array(self.get_xyz_coords(frame))
        coords_oriented = matrix_rotation*coords.transpose()
        return coords_oriented.transpose()

    def get_number_clusters(self,
                            frame: int,
                            eps: float,
                            min_samples: int) -> int:
        """
            Uses DBSCAN clustering algorithm to determine number of clusters
            for a given frame

            For more information, see:
            https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

            Args:
                frame: Frame in trajectory
                eps: Maximum distance between points to define a cluster
                min_samples: Minimum number of points to define a cluster

            Returns:
                number_clusters: Number of clusters for specified frame
        """
        coords = np.array(self.get_xyz_coords(frame))
        model = DBSCAN(eps=eps, min_samples= min_samples).fit(coords)
        number_clusters = len(set(model.labels_))
        return number_clusters
