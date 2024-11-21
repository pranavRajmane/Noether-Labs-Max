from mpi4py import MPI 
from dolfinx import mesh
from dolfinx import fem
import numpy
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem

import pyvista
from dolfinx import plot





domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

V = fem.functionspace(domain, ("Lagrange", 1))


uD = fem.Function(V)
uD.interpolate(lambda x: 1 +x[0]**2+2*x[1]**2)


tdim = domain.topology.dim
fdim = tdim-1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, default_scalar_type(-6))


a = ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx
L = f*v*ufl.dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type":"preonly", "pc_type":"lu"})
uh = problem.solve()


print(pyvista.global_theme.jupyter_backend)


domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")


u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()



