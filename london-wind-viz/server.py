"""Flask server – serves the CesiumJS frontend, streamline JSON data,
and accepts heightmap uploads from the browser for tile-aligned voxelisation."""
import os
import json
import threading

import numpy as np
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

import config

_BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=os.path.join(_BASE, "frontend"))
CORS(app)

_heightmap_event = threading.Event()


# ── Static / frontend ───────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/frontend/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ── Config ──────────────────────────────────────────────────────────
@app.route("/api/config")
def get_config():
    return jsonify({
        "cesium_token": config.CESIUM_ION_TOKEN,
        "center_lat": config.DOMAIN_CENTER_LAT,
        "center_lon": config.DOMAIN_CENTER_LON,
        "domain_half_x": config.DOMAIN_HALF_X,
        "domain_half_y": config.DOMAIN_HALF_Y,
        "domain_height": config.DOMAIN_HEIGHT,
        "wind_speed": config.WIND_SPEED,
        "available_angles": config.WIND_DIRECTIONS,
        "heightmap_sample_res": config.HEIGHTMAP_SAMPLE_RES,
        "ground_ellipsoid_height": config.GROUND_ELLIPSOID_HEIGHT,
        "voxel_resolution": config.VOXEL_RESOLUTION,
    })


# ── Streamlines ─────────────────────────────────────────────────────
@app.route("/api/metadata")
def get_metadata():
    path = os.path.join(config.STREAMLINE_DIR, "metadata.json")
    if not os.path.exists(path):
        return jsonify({"error": "Run the pipeline first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/streamlines/<int:angle>")
def get_streamlines(angle):
    path = os.path.join(config.STREAMLINE_DIR, f"streamlines_{angle}.json")
    if not os.path.exists(path):
        return jsonify({"error": f"No data for {angle}°"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


# ── Heightmap upload (from CesiumJS tile scanning) ──────────────────
@app.route("/api/heightmap", methods=["POST"])
def receive_heightmap():
    """Accept a 2-D height grid sampled from Google 3D Tiles by the browser,
    voxelise it into an occupancy grid + SDF, and persist to disk."""
    data = request.get_json(force=True)
    try:
        heights = np.array(data["heights"], dtype=np.float64)
        nx = int(data["nx"])
        ny = int(data["ny"])
        sample_res = float(data["resolution"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Bad payload: {e}"}), 400

    if heights.size != nx * ny:
        return jsonify({"error": f"Expected {nx*ny} heights, got {heights.size}"}), 400

    height_grid = heights.reshape(ny, nx)

    import geometry
    occ, sdf = geometry.voxelize_from_heightmap(height_grid, sample_res)
    geometry.save_heightmap_geometry(occ, sdf)

    coverage = float(occ.mean() * 100)
    print(f"  Heightmap received and voxelised – {coverage:.1f}% coverage")

    _heightmap_event.set()

    return jsonify({"status": "ok", "coverage": round(coverage, 1)})


def wait_for_heightmap():
    """Block until the browser POSTs a heightmap (used by --from-heightmap mode)."""
    _heightmap_event.wait()


# ── Startup ─────────────────────────────────────────────────────────
def start():
    banner = (
        "\n" + "=" * 60 +
        f"\n  London Wind Visualisation"
        f"\n  http://localhost:{config.SERVER_PORT}"
        "\n" + "=" * 60 + "\n"
    )
    print(banner)
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False)


if __name__ == "__main__":
    start()
