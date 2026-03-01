"""Flask server -- serves the CesiumJS frontend, accepts coordinates and
heightmap uploads, and serves computed streamline data."""
import os
import json
import math
import threading

import numpy as np
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

import config

_BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=os.path.join(_BASE, "frontend"))
CORS(app)

_heightmap_event = threading.Event()
_coords_event = threading.Event()
_pipeline_status = {"stage": "idle", "detail": ""}


def reset_pipeline_events():
    """Clear both events so the pipeline loop can wait for the next request."""
    _coords_event.clear()
    _heightmap_event.clear()


# ── Static / frontend ───────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/frontend/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ── Config (reflects current dynamic domain) ────────────────────────

@app.route("/api/config")
def get_config():
    return jsonify({
        "cesium_token": config.CESIUM_ION_TOKEN,
        "center_lat": config.DOMAIN_CENTER_LAT,
        "center_lon": config.DOMAIN_CENTER_LON,
        "domain_half_x": config.DOMAIN_HALF_X,
        "domain_half_y": config.DOMAIN_HALF_Y,
        "domain_height": config.DOMAIN_HEIGHT,
        "heightmap_sample_res": config.HEIGHTMAP_SAMPLE_RES,
        "ground_ellipsoid_height": config.GROUND_ELLIPSOID_HEIGHT,
        "voxel_resolution": config.VOXEL_RESOLUTION,
    })


# ── Coordinates (origin + destination) ──────────────────────────────

@app.route("/api/coords", methods=["POST"])
def receive_coords():
    """Accept origin + destination, compute 2x-expanded domain rectangle,
    update config, and unblock the pipeline."""
    data = request.get_json(force=True)
    try:
        o_lat = float(data["origin"]["lat"])
        o_lon = float(data["origin"]["lon"])
        d_lat = float(data["dest"]["lat"])
        d_lon = float(data["dest"]["lon"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Bad coords: {e}"}), 400

    o_height = float(data.get("origin_height", 30.0))
    d_height = float(data.get("dest_height", 30.0))

    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians((o_lat + d_lat) / 2)))

    center_lat = (o_lat + d_lat) / 2
    center_lon = (o_lon + d_lon) / 2

    dx_m = abs(o_lon - d_lon) / lon_per_m
    dy_m = abs(o_lat - d_lat) / lat_per_m
    half_x = max(dx_m, 100.0)
    half_y = max(dy_m, 100.0)

    config.DOMAIN_CENTER_LAT = center_lat
    config.DOMAIN_CENTER_LON = center_lon
    config.DOMAIN_HALF_X = half_x
    config.DOMAIN_HALF_Y = half_y
    config.ORIGIN_LAT = o_lat
    config.ORIGIN_LON = o_lon
    config.ORIGIN_HEIGHT = o_height
    config.DEST_LAT = d_lat
    config.DEST_LON = d_lon
    config.DEST_HEIGHT = d_height

    print(f"  Domain set: center ({center_lat:.5f}, {center_lon:.5f}), "
          f"extent {half_x*2:.0f}×{half_y*2:.0f} m")

    _coords_event.set()

    return jsonify({
        "status": "ok",
        "center_lat": center_lat,
        "center_lon": center_lon,
        "half_x": half_x,
        "half_y": half_y,
        "cached": config.has_cached_geometry(),
    })


def wait_for_coords():
    _coords_event.wait()


# ── Pipeline status (polled by frontend) ────────────────────────────

@app.route("/api/pipeline-status")
def pipeline_status():
    return jsonify(_pipeline_status)


def set_pipeline_status(stage: str, detail: str = ""):
    _pipeline_status["stage"] = stage
    _pipeline_status["detail"] = detail


# ── Streamlines ─────────────────────────────────────────────────────

@app.route("/api/streamlines/combined")
def get_combined_streamlines():
    path = os.path.join(config.STREAMLINE_DIR, "streamlines_combined.json")
    if not os.path.exists(path):
        return jsonify({"error": "Not computed yet"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/streamlines/<int:angle>")
def get_streamlines(angle):
    path = os.path.join(config.STREAMLINE_DIR, f"streamlines_{angle}.json")
    if not os.path.exists(path):
        return jsonify({"error": f"No data for {angle}°"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/metadata")
def get_metadata():
    path = os.path.join(config.STREAMLINE_DIR, "metadata.json")
    if not os.path.exists(path):
        return jsonify({"error": "Run the pipeline first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/api/routes")
def get_routes():
    path = os.path.join(config.ROUTES_DIR, "routes.json")
    if not os.path.exists(path):
        return jsonify({"error": "Routes not computed yet"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


# ── Heightmap upload ────────────────────────────────────────────────

@app.route("/api/heightmap", methods=["POST"])
def receive_heightmap():
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
    _heightmap_event.wait()


# ── Startup ─────────────────────────────────────────────────────────

def start():
    banner = (
        "\n" + "=" * 60 +
        f"\n  Urban Wind Visualisation"
        f"\n  http://localhost:{config.SERVER_PORT}"
        "\n" + "=" * 60 + "\n"
    )
    print(banner)
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False)


if __name__ == "__main__":
    start()
