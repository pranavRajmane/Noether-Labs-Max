from flask import Flask, request, jsonify, render_template
import json

from groq import Groq
import os

from heat_eqn2 import (
    main_simulation,
    parse_command_to_params,
    create_mesh_from_params,
    create_initial_condition,
    create_boundary_condition,
)

from flask import send_file

from multiprocessing import Process

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


from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib as mpl 

# Initialize Flask app
app = Flask(__name__)
pv.OFF_SCREEN = True
# Home route
@app.route("/")
def index():
    return render_template("index.html")  # Ensure templates/index.html exists

# Initialize Groq client with error handling
try:
    client = Groq(api_key="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")
except Exception as e:
    client = None
    print(f"Groq client initialization failed: {e}")

# Route to parse natural language command
@app.route("/parse_command", methods=["POST"])
def parse_command():
    try:
        user_command = request.json.get("command")
        if not user_command:
            return jsonify({"status": "error", "message": "Command not provided."}), 400

        # Serialize command if it's a dictionary
        if isinstance(user_command, dict):
            user_command = json.dumps(user_command)

        # Pass to parsing function
        parsed_params = parse_command_to_params(user_command, client)
        return jsonify({"status": "success", "parameters": parsed_params})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def render_simulation(params):
    output_directory = "./static"
    gif_filename = "u_time.gif"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Run the simulation
    main_simulation(params)

    # Move the GIF to the static directory
    if os.path.exists(gif_filename):
        os.rename(gif_filename, os.path.join(output_directory, gif_filename))



# Route to run simulation
@app.route("/simulate", methods=["POST"])
def simulate():
    """API endpoint to run heat equation simulations."""

        
    params = request.json
    process = Process(target=render_simulation, args=(params,))
    process.start()
    process.join()
    return jsonify({"status": "success", "message": "Simulation complete."})

@app.route("/view_gif")
def view_gif():
    try:
        gif_path = os.path.join(app.static_folder, "u_time.gif")
        if os.path.exists(gif_path):
            return send_file(gif_path, mimetype="image/gif")
        else:
            return jsonify({"status": "error", "message": "GIF not found."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500




if __name__ == "__main__":
    # Bind to all interfaces for network access
    app.run(host="0.0.0.0", port=5000, debug=True)
