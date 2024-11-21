import meshio

import meshio

# Define paths
input_mesh_path = "/Users/pranavrajmane/Desktop/cube 2.msh"
output_mesh_path = "/Users/pranavrajmane/Desktop/model.xdmf"

# Read the Gmsh file and convert to XDMF
mesh = meshio.read(input_mesh_path)
meshio.write(output_mesh_path, mesh)


