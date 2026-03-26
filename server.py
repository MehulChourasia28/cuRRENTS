"""Flask server — frontend, API, and pipeline coordination."""
import os
import json
import math
import threading

import numpy as np
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

import config

_BASE = os.path.dirname(os.path.abspath(__file__))
app   = Flask(__name__, static_folder=os.path.join(_BASE, "frontend"))
CORS(app)

# ── Pipeline synchronisation ──────────────────────────────────────────
# The pipeline loop waits on these events.
_coords_event    = threading.Event()
_heightmap_event = threading.Event()
_pipeline_status = {"stage": "idle", "detail": ""}
_status_lock     = threading.Lock()


def set_status(stage, detail=""):
    with _status_lock:
        _pipeline_status["stage"]  = stage
        _pipeline_status["detail"] = detail


def wait_for_coords():
    _coords_event.wait()


def wait_for_heightmap():
    _heightmap_event.wait()


def reset_events():
    _coords_event.clear()
    _heightmap_event.clear()


# ── Static ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/frontend/<path:fn>")
def static_files(fn):
    return send_from_directory(app.static_folder, fn)


# ── Config ────────────────────────────────────────────────────────────

@app.route("/api/config")
def get_config():
    return jsonify({
        "cesium_token":           config.CESIUM_ION_TOKEN,
        "center_lat":             config.DOMAIN_CENTER_LAT,
        "center_lon":             config.DOMAIN_CENTER_LON,
        "domain_half_x":          config.DOMAIN_HALF_X,
        "domain_half_y":          config.DOMAIN_HALF_Y,
        "domain_height":          config.DOMAIN_HEIGHT,
        "heightmap_sample_res":   config.HEIGHTMAP_SAMPLE_RES,
        "ground_ellipsoid_height":config.GROUND_ELLIPSOID_HEIGHT,
        "voxel_resolution":       config.VOXEL_RESOLUTION,
    })


# ── Coordinates ───────────────────────────────────────────────────────

@app.route("/api/coords", methods=["POST"])
def receive_coords():
    data = request.get_json(force=True)
    try:
        o_lat = float(data["origin"]["lat"])
        o_lon = float(data["origin"]["lon"])
        d_lat = float(data["dest"]["lat"])
        d_lon = float(data["dest"]["lon"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    o_h = float(data.get("origin_height", 30.0))
    d_h = float(data.get("dest_height",   30.0))

    mid_lat   = (o_lat + d_lat) / 2
    mid_lon   = (o_lon + d_lon) / 2
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(mid_lat)))

    # Domain is the bounding box of the route, with 1.5× padding
    dx_m  = abs(o_lon - d_lon) / lon_per_m
    dy_m  = abs(o_lat - d_lat) / lat_per_m
    half_x = max(dx_m * 1.5, 150.0)
    half_y = max(dy_m * 1.5, 150.0)

    config.set_domain(mid_lat, mid_lon, half_x, half_y,
                      o_lat, o_lon, o_h, d_lat, d_lon, d_h)

    cached = config.has_cached_geometry()
    _coords_event.set()

    return jsonify({
        "status":     "ok",
        "center_lat": mid_lat,
        "center_lon": mid_lon,
        "half_x":     half_x,
        "half_y":     half_y,
        "cached":     cached,
    })


# ── Pipeline status ───────────────────────────────────────────────────

@app.route("/api/pipeline-status")
def pipeline_status():
    with _status_lock:
        return jsonify(dict(_pipeline_status))


# ── Heightmap upload ──────────────────────────────────────────────────

@app.route("/api/heightmap", methods=["POST"])
def receive_heightmap():
    data = request.get_json(force=True)
    try:
        heights    = np.array(data["heights"], dtype=np.float64)
        nx         = int(data["nx"])
        ny         = int(data["ny"])
        sample_res = float(data["resolution"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    if heights.size != nx * ny:
        return jsonify({"error": f"Expected {nx*ny} samples, got {heights.size}"}), 400

    height_grid = heights.reshape(ny, nx)

    import geometry
    dom = config.snapshot()
    occ, ground_h = geometry.voxelize_from_heightmap(height_grid, sample_res, dom)
    config.GROUND_ELLIPSOID_HEIGHT = ground_h
    geometry.save_geometry(occ, dom, ground_h)

    coverage = float(occ.mean() * 100)
    print(f"  Voxelised: {coverage:.1f}% building coverage  ground={ground_h:.1f} m")
    _heightmap_event.set()
    return jsonify({"status": "ok", "coverage": round(coverage, 1)})


# ── Streamlines ───────────────────────────────────────────────────────

@app.route("/api/streamlines/combined")
def get_streamlines():
    path = os.path.join(config.STREAMLINE_DIR, "streamlines_combined.json")
    if not os.path.exists(path):
        return jsonify({"error": "Not computed yet"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/metadata")
def get_metadata():
    path = os.path.join(config.STREAMLINE_DIR, "metadata.json")
    if not os.path.exists(path):
        return jsonify({"error": "Not ready"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/api/routes")
def get_routes():
    path = os.path.join(config.ROUTES_DIR, "routes.json")
    if not os.path.exists(path):
        return jsonify({"error": "Routes not computed yet"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


# ── Start ─────────────────────────────────────────────────────────────

def start():
    print(f"\n{'='*55}")
    print(f"  Urban Wind Visualisation  —  http://localhost:{config.SERVER_PORT}")
    print(f"{'='*55}\n")
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False)


if __name__ == "__main__":
    start()
