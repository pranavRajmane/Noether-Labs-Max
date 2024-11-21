from flask import Flask, request, jsonify
from mpi4py import MPI
from dolfinx import fem, mesh, io
import numpy as np
from groq import Groq
import json
import ufl
from petsc4py import PETSc

# Initialize Flask app
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

# Initialize Groq client
client = Groq(api_key="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")


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
            temperature=0.05,
        )
        response_content = response.choices[0].message.content.strip()

        # Ensure Llama returned valid JSON
        if not response_content:
            raise ValueError("Llama returned an empty response.")

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
    """Create a mesh based on parsed parameters."""
    mesh_type = mesh_params.get("type", "rectangle")
    mesh_params.setdefault("bounds", [0, 1, 0, 1])
    mesh_params.setdefault("resolution", [50, 50])

    if mesh_type == "rectangle":
        p0 = [mesh_params["bounds"][0], mesh_params["bounds"][2]]
        p1 = [mesh_params["bounds"][1], mesh_params["bounds"][3]]
        resolution = mesh_params["resolution"]
        domain = mesh.create_rectangle(
            MPI.COMM_WORLD, [np.array(p0), np.array(p1)], resolution, mesh.CellType.triangle
        )
    elif mesh_type == "circle":
        center = mesh_params.get("center", [0.0, 0.0])
        radius = mesh_params.get("radius", 1.0)
        resolution = mesh_params.get("resolution", 50)
        domain = mesh.create_circle(
            MPI.COMM_WORLD, np.array(center), radius, resolution, mesh.CellType.triangle
        )
    else:
        raise ValueError(f"Unsupported mesh type: {mesh_type}")
    return domain


@app.route("/simulate", methods=["POST"])
def simulate():
    """Handle simulation requests."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415

        data = request.get_json()
        command = data.get("command")
        if not command:
            return jsonify({"error": "Missing 'command' in request"}), 400

        # Parse the command
        params = parse_command_to_params(command, client)
        simulation_time = params.get("simulation_time", 4.0)
        time_steps = params.get("time_steps", 50)
        dt = simulation_time / time_steps

        # Create the mesh
        domain = create_mesh_from_params(params["mesh"])
        V = fem.FunctionSpace(domain, ("Lagrange", 1))

        # Initialize solution and write to file
        u_n = fem.Function(V)
        xdmf = io.XDMFFile(domain.comm, "simulation_output.xdmf", "w")
        xdmf.write_mesh(domain)

        # Simulation loop (dummy logic for now)
        for t in np.linspace(0, simulation_time, time_steps):
            u_n.x.array[:] = np.sin(t)  # Example update for u_n

        xdmf.write_function(u_n, simulation_time)
        xdmf.close()

        return jsonify({"message": "Simulation completed. Output saved to simulation_output.xdmf"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
