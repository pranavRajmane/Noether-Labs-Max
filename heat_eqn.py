import meshio
import dolfinx
from mpi4py import MPI
import numpy as np
from groq import Groq

from dolfinx import fem, mesh, io, plot, default_scalar_type
import ufl
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

from petsc4py import PETSc

import pyvista


from copy import deepcopy

from groq1 import parse_command_to_params
import matplotlib.pyplot as plt
import matplotlib as mpl  # This sets mpl as an alias for the main matplotlib package



client = Groq(
    api_key= "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",
)



# Example user command
command = input('Hi You, I am MAX from NOETHER LABS. I will help you simulate physics at the speed of thought \nand bring your dreams to reality faster than the othe Max Verstappen in the Sao Paulo GP 2024.\n so, what would you like to simulate on this fine day??')

# Parse the command
params = parse_command_to_params(command, client)

# Use parsed parameters
print("Parsed Parameters:", params)

# Example: Extracting specific values
simulation_time = params.get("simulation_time", 4.0)
time_steps = params.get("time_steps", 50)
dt = simulation_time/time_steps
mesh_type = params["mesh"]["type"]
print(f"Simulation Time: {simulation_time}, Time Steps: {time_steps}, Mesh Type: {mesh_type}")


def create_mesh_from_params(mesh_params):
    """
    Create a mesh based on the parsed parameters.

    Args:
        mesh_params (dict): Parameters defining the mesh.
    Returns:
        domain: The created mesh domain.
    """
    mesh_type = mesh_params.get("type", "rectangle")
    mesh_params.setdefault("bounds", [0, 1, 0, 1])  # Default bounds
    mesh_params.setdefault("resolution", [50, 50])  # Default resolution

    
    if mesh_type == "rectangle":
        # Apply defaults if bounds or resolution are None
        bounds = mesh_params.get("bounds", None) or [0, 1, 0, 1]  # Default bounds
        resolution = mesh_params.get("resolution", None) or [50, 50]  # Default resolution
        
        # Validate bounds and resolution
        if len(bounds) != 4 or len(resolution) != 2:
            raise ValueError("Rectangle requires 'bounds' (4 values) and 'resolution' (2 values).")
        
        # Convert bounds to two corner points: bottom-left and top-right
        p0 = [bounds[0], bounds[2]]  # Bottom-left corner
        p1 = [bounds[1], bounds[3]]  # Top-right corner
        if p0[0] >= p1[0] or p0[1] >= p1[1]:
            raise ValueError(f"Invalid rectangle bounds: {bounds}. Ensure bounds[0] < bounds[1] and bounds[2] < bounds[3].")
        
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD,
            [np.array(p0), np.array(p1)],
            resolution,
            mesh.CellType.triangle
        )
    elif mesh_type == "circle":
        # Apply defaults if center or radius are None
        center = mesh_params.get("center", None) or [0.0, 0.0]  # Default center
        radius = mesh_params.get("radius", None) or 1.0  # Default radius
        resolution = mesh_params.get("resolution", None) or 50  # Default resolution
        
        # Validate center
        if len(center) != 2:
            raise ValueError("Circle requires 'center' (2 values).")
        
        domain = mesh.create_circle(
            MPI.COMM_WORLD,
            np.array(center),
            radius,
            resolution,
            mesh.CellType.triangle
        )
    else:
        raise ValueError(f"Unsupported mesh type: {mesh_type}")

    print(f"Mesh created: {mesh_type} with bounds/resolution.")
    return domain




import numpy as np
from dolfinx import fem
import ufl

import numpy as np
from dolfinx import fem, mesh
import ufl
from mpi4py import MPI
from petsc4py import PETSc

def create_initial_condition(V, params=None):
    """
    Create an initial condition for the heat equation.
    Args:
        V (FunctionSpace): The function space
        params (dict): Parameters for initial condition
    Returns:
        fem.Function: The initialized function
    """
    if params is None:
        params = {"type": "gaussian"}

    # Create function for initial condition
    u0 = fem.Function(V)
    
    # Get mesh coordinates
    mesh = V.mesh
    x = ufl.SpatialCoordinate(mesh)
    
    # Set expression based on type
    ic_type = params.get("type", "gaussian").lower()
    
    if ic_type == "gaussian":
        # Parameters for Gaussian
        a = params.get("a", 1.0)
        p = params.get("p", 2)
        n = params.get("n", 2)
        
        # Create UFL expression
        expr = ufl.exp(-a * (x[0]**p + x[1]**n))
        
    elif ic_type == "sinusoidal":
        # Parameters for sinusoidal
        freq = params.get("frequency", 1.0)
        expr = ufl.sin(freq * x[0]) * ufl.sin(freq * x[1])
        
    elif ic_type == "dirac":
        # Parameters for dirac
        center = params.get("center", [0.0, 0.0])
        eps = params.get("epsilon", 1e-2)
        
        # Create UFL expression
        r = ufl.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2)
        expr = ufl.conditional(ufl.lt(r, eps), 1.0, 0.0)
    
    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")
    
    # Create Expression object
    expr = fem.Expression(expr, V.element.interpolation_points())
    
    # Interpolate the expression
    u0.interpolate(expr)
    
    return u0



def create_boundary_condition_from_params(boundary_params, V):
    """
    Create boundary conditions based on the parsed parameters.

    Args:
        boundary_params (dict): Parameters defining the boundary condition.
        V (FunctionSpace): The function space for the boundary condition.
    Returns:
        bc: The created boundary condition.
    """
    # Default to Dirichlet boundary condition with value 0
    bc_type = boundary_params.get("type", "Dirichlet").lower()
    value = boundary_params.get("value", 0.0)

    if bc_type == "dirichlet":
        fdim = V.mesh.topology.dim - 1  # Facet dimension
        # Apply Dirichlet boundary condition to all facets
        boundary_facets = mesh.locate_entities_boundary(
            V.mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = fem.dirichletbc(PETSc.ScalarType(value), dofs, V)
    elif bc_type == "neumann":
        # Neumann BC is applied as part of the weak form and not explicitly here
        bc = None
        print("Neumann boundary condition detected: Ensure it's implemented in the weak form.")
    else:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")

    print(f"Boundary condition created: {bc_type} with value {value}")
    return bc

    try:
        bc = create_boundary_condition_from_params(boundary_params, V)
    except ValueError as e:
        print(f"Error creating boundary condition: {e}")



# Main execution
# Write mesh and initial function to file

mesh_params = params["mesh"]
domain = create_mesh_from_params(mesh_params)

V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))



xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)




uh = fem.Function(V)
uh.name = "uh"
expr = create_initial_condition(V)

u_n = fem.Function(V)  # Previous solution
u_n.x.array[:] = expr.x.array  # Initialize with the initial condition

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)


xdmf.write_function(expr, simulation_time)


u, v =ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx 
L = (u_n + dt * f) * v * ufl.dx 


bilinear_form = fem.form(a)
linear_form = fem.form(L)
#solver
# Assembly of matrix A (only once)

A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
print("Non-zero entries in A:", A.getValuesCSR())



# Assemble initial RHS vector
b = create_vector(linear_form)
assemble_vector(b, linear_form)

# Setup solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)  # Using CG as an alternative to PREONLY
solver.getPC().setType(PETSc.PC.Type.JACOBI)  # Jacobi preconditioning

# Optional: Add a monitor to check residuals at each iteration
#solver.setMonitor(lambda _, its, rnorm: print(f"Iter {its}, residual norm {rnorm}"))




# Set up plotter and GIF
plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif", fps=1)




# Initialize time and loop
t = 0.0
for i in range(time_steps):
    t += dt

    # Update the right-hand side vector `b`
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve the linear system
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Debugging: Check `uh` values for each iteration
    print(f"Time step {i+1}/{time_steps}, t = {t:.3f}, max uh: {np.max(uh.x.array)}")

    # Update the solution for the next step
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)

    # Update grid's scalar field and reapply warp
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))  # Re-create grid for each frame
    grid.point_data["uh"] = uh.x.array  # Update scalar field

    # Warp by scalar (ensure warp factor is set properly)
    warped = deepcopy(grid).warp_by_scalar("uh", factor=1)  # Use deepcopy to avoid issues

    # Update mesh in plotter with the latest `warped` object
    plotter.add_mesh(warped, show_edges=True, lighting=False, cmap="viridis",
                     scalar_bar_args={"title": "uh", "title_font_size": 12, "label_font_size": 10},
                     clim=[0, np.max(uh.x.array)])  # Manually set color range

    plotter.set_background("black")

    # Write the current frame
    plotter.write_frame()
    plotter.clear()  # Clear previous frame to avoid overlay issues

# Close plotter and file
plotter.close()
xdmf.close()
