# Molecular-Shape-Matching
Implementation of a Molecule class from xyz or cus trajectories

An instance of the Molecule class can access various coordinate transformation methods (scaling, translation and rotation) and detect the number of clusters. 
Example applications include:
- Molecular Shape Matching via Voxel matching
- Generation of ordered configurations for single/multiple collections of molecules

Example of creating an ordered configuration of vesicles:

![multi_configuration](https://user-images.githubusercontent.com/50631178/134701268-29a78bdd-9ca0-4103-8fe9-ae01f4085f95.png)

Voxelization of a torus:

![torus_voxel](https://user-images.githubusercontent.com/50631178/134750911-b2d1e8bc-d715-4023-8b0c-5179d264c9bb.png)


# Usage
`python -m examples.test_voxelize`

