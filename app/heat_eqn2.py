import meshio
import dolfinx
from mpi4py import MPI
import numpy as np
from groq import Groq
import json
from dolfinx import fem, mesh, io, plot, default_scalar_type
import ufl
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

from petsc4py import PETSc

import pyvista as pv
from PIL import Image


from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib as mpl  # This sets mpl as an alias for the main matplotlib package

client = Groq(
    api_key= "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",
)


pv.OFF_SCREEN = True
def parse_command_to_params(user_command, client, model="llama3-70b-8192"):
    """
    Parse a natural language command into simulation parameters using Llama.

    Args:
        user_command (str): The user command in natural language.
        client: The Llama client object to communicate with the model.
        model (str): The Llama model to use for parsing.

    Returns:
        dict: Parsed parameters as a dictionary.
    """
    try:
        # Ensure the command is a string
        import json  # Make sure json is imported
        if isinstance(user_command, dict):
            user_command = json.dumps(user_command)  # Serialize dictionary to a string

        # Send command to Llama for parsing
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Parse the given command into JSON with the following keys: "
                        "mesh (type, bounds, resolution), initial_condition, boundary_condition, simulation_time, time_steps. "
                        "Only output valid JSON and nothing else. Strictly follow this format."
                    ),
                },
                {"role": "user", "content": user_command},
            ],
            model=model,
            temperature=0.5,
        )

        # Extract content from Llama response
        response_content = response.choices[0].message.content.strip()

        # Ensure Llama returned valid JSON
        if not response_content:
            raise ValueError("Llama returned an empty response.")

        # Parse the response content as JSON
        params = json.loads(response_content)
        return params

    except json.JSONDecodeError as e:
        print("Error: Llama did not return valid JSON.")
        print("Response Content:", response_content)
        raise e
    except Exception as e:
        print(f"Error while processing the command: {e}")
        raise e


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


def create_boundary_condition(V, params):
    """Creates boundary conditions based on input parameters."""
    bc_type = params.get("type", "Dirichlet").lower()
    if bc_type == "dirichlet":
        value = PETSc.ScalarType(params.get("value", 0.0))
        facets = mesh.locate_entities_boundary(V.mesh, V.mesh.topology.dim - 1, lambda x: True)
        dofs = fem.locate_dofs_topological(V, V.mesh.topology.dim - 1, facets)
        return fem.dirichletbc(value, dofs, V)
    else:
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")





def main_simulation(command):
    import numpy as np
    from dolfinx import fem, mesh, io
    from petsc4py import PETSc
    import ufl
    from copy import deepcopy
    import pyvista

    # Parse the command
    params = parse_command_to_params(command, client)

    # Extract and process parameters
    simulation_time = params.get("simulation_time", 4.0)
    time_steps = params.get("time_steps", 50)
    dt = simulation_time / time_steps
    mesh_params = params["mesh"]
    domain = create_mesh_from_params(mesh_params)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Output file setup
    xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
    xdmf.write_mesh(domain)

    # Set up initial conditions
    uh = fem.Function(V)
    uh.name = "uh"
    expr = create_initial_condition(V)
    u_n = fem.Function(V)
    u_n.x.array[:] = expr.x.array

    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    bc = fem.dirichletbc(
        PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V
    )

    # Write initial condition
    xdmf.write_function(expr, 0.0)

    # Setup forms
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(0))
    a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f) * v * ufl.dx

    # Assemble matrix
    A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    b = fem.petsc.create_vector(fem.form(L))

    # Solver setup
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.JACOBI)

    # Setup PyVista plotter
    print("Starting plotter...")
    plotter = pyvista.Plotter(off_screen=True)
    plotter.open_gif("u_time.gif", fps=10)

    # Time-stepping loop
    t = 0.0
    for i in range(time_steps):
        t += dt

        # Assemble and solve
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, fem.form(L))
        fem.petsc.apply_lifting(b, [fem.form(a)], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        # Write solution to file
        xdmf.write_function(uh, t)

        # Update plotter
        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
        grid.point_data["uh"] = uh.x.array
        warped = deepcopy(grid).warp_by_scalar("uh", factor=1)
        plotter.add_mesh(
            warped,
            show_edges=True,
            lighting=False,
            cmap="viridis",
            scalar_bar_args={"title": "uh", "title_font_size": 12, "label_font_size": 10},
            clim=[0, np.max(uh.x.array)],
        )
        plotter.set_background("black")
        plotter.write_frame()
        print(f"Frame {i + 1}/{time_steps} written successfully.")
        plotter.clear()

    # Close resources
    plotter.close()
    xdmf.close()
    print("Simulation completed successfully.")

    return "Simulation completed successfully."


# Run the simulation
command = "Run a heat transfer simulation for a rectangle mesh with bounds [1, 3, 1, 3], with resolution [50,50] with initial dirac condition and Dirichlet boundary condition. Run the simulation for 4 seconds at 50 time steps"

if __name__ == "__main__":
    result = main_simulation(command)
    print(result)

    # Verify the number of frames in the generated GIF
    gif_path = "u_time.gif"
    with Image.open(gif_path) as img:
        print(f"Number of frames in GIF: {img.n_frames}")




