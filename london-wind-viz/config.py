"""Configuration for the urban wind visualization system."""
import os
import hashlib

from dotenv import load_dotenv
load_dotenv()

# ===================== Domain (set dynamically by /api/coords) ======
DOMAIN_CENTER_LAT = 51.5135    # default; overwritten at runtime
DOMAIN_CENTER_LON = -0.0850
DOMAIN_HALF_X = 200.0          # metres
DOMAIN_HALF_Y = 200.0
DOMAIN_HEIGHT = 150.0

ORIGIN_LAT = 51.5130           # set dynamically from browser
ORIGIN_LON = -0.0890
ORIGIN_HEIGHT = 30.0           # metres AGL
DEST_LAT = 51.5150
DEST_LON = -0.0820
DEST_HEIGHT = 30.0

VOXEL_RESOLUTION = 2.0

# ===================== Building Configuration =====================
DEFAULT_BUILDING_HEIGHT = 15.0
MIN_BUILDING_HEIGHT = 3.0
BUILDING_HEIGHT_BUFFER = 12.0
BUILDING_H_DILATE = 3
OSM_FETCH_RADIUS = 250

# ===================== LBM Solver =================================
LBM_GRID_RES = 4.0
WIND_SPEED = 10.0              # fallback; overridden per-solve

# ===================== API Keys (all from .env) ====================
NIM_API_KEY = os.environ.get("NIM_API_KEY", "")
OPENWEATHER_API_KEY = os.environ.get("OPEN_WEATHER_API_KEY", "")
CESIUM_ION_TOKEN = os.environ.get("CESIUM_ION_TOKEN", "")

# ===================== Nemotron / NIM =============================
NEMOTRON_TARGET = 3
NEMOTRON_THRESHOLD = 0.7

# ===================== Streamline Configuration ====================
STREAMLINE_DT = 0.5
STREAMLINE_MAX_STEPS = 1000
STREAMLINE_MIN_SPEED = 0.5
STREAMLINE_MIN_LENGTH = 20.0

SEED_INLET_COUNT = 150
SEED_BUILDING_COUNT = 300
SEED_STREET_COUNT = 400

OPACITY_MIN_SPEED = 1.0
OPACITY_MAX_SPEED = 15.0

# ===================== Drone Parameters ===========================
DRONE_MASS = 5.0               # kg
DRONE_CRUISE_SPEED = 15.0      # m/s
DRONE_DISC_AREA = 0.5          # m² (rotor disc)
DRONE_FRONTAL_AREA = 0.1       # m² (body frontal area)
DRONE_CD = 0.5                 # drag coefficient

# ===================== Routing / cuOpt ============================
ROUTING_GRID_RES = 8.0         # metres – coarser grid for pathfinding
CUOPT_TIME_LIMIT = 5           # seconds allowed for cuOpt solver
NUM_ROUTE_WAYPOINTS = 80       # candidate intermediate waypoints
WIND_COST_ALPHA = 0.4           # headwind penalty weight
WIND_COST_BETA = 0.25          # speed-magnitude penalty weight
CUOPT_ENDPOINT = "https://optimize.api.nvidia.com/v1/nvidia/cuopt"

# ===================== Heightmap Sampling ==========================
HEIGHTMAP_SAMPLE_RES = 4.0
HEIGHTMAP_MIN_BUILDING = 3.0

# ===================== Server =====================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080

# ===================== Paths ======================================
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, "data")
BUILDINGS_DIR = os.path.join(DATA_DIR, "buildings")
DOMAIN_DIR = os.path.join(DATA_DIR, "domain")
STREAMLINE_DIR = os.path.join(DATA_DIR, "streamlines")
ROUTES_DIR = os.path.join(DATA_DIR, "routes")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

GROUND_ELLIPSOID_HEIGHT = 58.0


def domain_cache_dir() -> str:
    """Return a cache directory keyed by the current domain bounds.
    Geometry stored here persists across runs for the same area."""
    key = f"{DOMAIN_CENTER_LAT:.6f}_{DOMAIN_CENTER_LON:.6f}_{DOMAIN_HALF_X:.1f}_{DOMAIN_HALF_Y:.1f}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    d = os.path.join(CACHE_DIR, h)
    os.makedirs(d, exist_ok=True)
    return d


def has_cached_geometry() -> bool:
    """Check if voxelised geometry exists in the cache for the current domain."""
    d = domain_cache_dir()
    return (os.path.exists(os.path.join(d, "occupancy.npy"))
            and os.path.exists(os.path.join(d, "buildings.json")))
