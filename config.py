import os
import math
import threading
import hashlib
from dotenv import load_dotenv

load_dotenv()

NIM_API_KEY          = os.environ.get("NIM_API_KEY", "")
OPENWEATHER_API_KEY  = os.environ.get("OPEN_WEATHER_API_KEY", "")
CESIUM_ION_TOKEN     = os.environ.get("CESIUM_ION_TOKEN", "")

# Domain defaults — overwritten by /api/coords on each request
DOMAIN_CENTER_LAT = 51.5135
DOMAIN_CENTER_LON = -0.0850
DOMAIN_HALF_X     = 200.0   # metres
DOMAIN_HALF_Y     = 200.0
DOMAIN_HEIGHT     = 150.0

ORIGIN_LAT    = 51.5130
ORIGIN_LON    = -0.0890
ORIGIN_HEIGHT = 30.0        # metres AGL
DEST_LAT      = 51.5150
DEST_LON      = -0.0820
DEST_HEIGHT   = 30.0

GROUND_ELLIPSOID_HEIGHT = 58.0   # updated after heightmap scan

_domain_lock = threading.Lock()


def set_domain(center_lat, center_lon, half_x, half_y,
               origin_lat, origin_lon, origin_h,
               dest_lat, dest_lon, dest_h):
    global DOMAIN_CENTER_LAT, DOMAIN_CENTER_LON
    global DOMAIN_HALF_X, DOMAIN_HALF_Y
    global ORIGIN_LAT, ORIGIN_LON, ORIGIN_HEIGHT
    global DEST_LAT, DEST_LON, DEST_HEIGHT
    with _domain_lock:
        DOMAIN_CENTER_LAT = center_lat
        DOMAIN_CENTER_LON = center_lon
        DOMAIN_HALF_X     = half_x
        DOMAIN_HALF_Y     = half_y
        ORIGIN_LAT        = origin_lat
        ORIGIN_LON        = origin_lon
        ORIGIN_HEIGHT     = origin_h
        DEST_LAT          = dest_lat
        DEST_LON          = dest_lon
        DEST_HEIGHT       = dest_h


def snapshot():
    with _domain_lock:
        return dict(
            center_lat=DOMAIN_CENTER_LAT,
            center_lon=DOMAIN_CENTER_LON,
            half_x=DOMAIN_HALF_X,
            half_y=DOMAIN_HALF_Y,
            height=DOMAIN_HEIGHT,
            origin_lat=ORIGIN_LAT,
            origin_lon=ORIGIN_LON,
            origin_height=ORIGIN_HEIGHT,
            dest_lat=DEST_LAT,
            dest_lon=DEST_LON,
            dest_height=DEST_HEIGHT,
            ground_ellipsoid_height=GROUND_ELLIPSOID_HEIGHT,
        )


# Voxelisation
VOXEL_RESOLUTION      = 2.0   # metres
HEIGHTMAP_SAMPLE_RES  = 4.0
HEIGHTMAP_MIN_BUILDING= 3.0
BUILDING_H_DILATE     = 2

# LBM solver
LBM_GRID_RES  = 4.0   # metres per cell
LBM_U_LATTICE = 0.05  # lattice velocity (keep < 0.1 for stability)
LBM_RE        = 500.0
LBM_N_STEPS   = 3000

# Streamlines
STREAMLINE_DT        = 1.0    # max integration step (m)
STREAMLINE_MAX_STEPS = 800
STREAMLINE_MIN_SPEED = 0.3    # m/s
STREAMLINE_MIN_LEN   = 15.0   # m

N_SEEDS_INLET    = 120
N_SEEDS_BUILDING = 200
N_SEEDS_STREET   = 300

OPACITY_MIN_SPEED = 0.5
OPACITY_MAX_SPEED = 12.0

# Drone energy model
DRONE_MASS        = 5.0   # kg
DRONE_CRUISE_SPEED= 12.0  # m/s
DRONE_DISC_AREA   = 0.5   # m²
DRONE_FRONTAL_AREA= 0.1   # m²
DRONE_CD          = 0.5

# Routing
ROUTING_GRID_RES    = 3.0   # metres horizontal
ROUTING_GRID_RES_Z  = 3.0   # metres vertical
MIN_ROUTE_CLEARANCE = 5.0   # minimum SDF distance from buildings (m)
WIND_COST_ALPHA     = 0.35  # headwind penalty weight
WIND_COST_BETA      = 0.15  # turbulence penalty weight
NUM_ROUTE_WAYPOINTS = 60

CUOPT_ENDPOINT  = "https://optimize.api.nvidia.com/v1/nvidia/cuopt"
CUOPT_TIME_LIMIT= 30

# Nemotron
NEMOTRON_TARGET    = 3
NEMOTRON_THRESHOLD = 0.7

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080

# Paths
_BASE          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(_BASE, "data")
DOMAIN_DIR     = os.path.join(DATA_DIR, "domain")
ROUTES_DIR     = os.path.join(DATA_DIR, "routes")
STREAMLINE_DIR = os.path.join(DATA_DIR, "streamlines")
CACHE_DIR      = os.path.join(DATA_DIR, "cache")


def _ensure_dirs():
    for d in (DATA_DIR, DOMAIN_DIR, ROUTES_DIR, STREAMLINE_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)


def domain_cache_key():
    k = f"{DOMAIN_CENTER_LAT:.5f}_{DOMAIN_CENTER_LON:.5f}_{DOMAIN_HALF_X:.0f}_{DOMAIN_HALF_Y:.0f}"
    return hashlib.md5(k.encode()).hexdigest()[:10]


def domain_cache_dir():
    d = os.path.join(CACHE_DIR, domain_cache_key())
    os.makedirs(d, exist_ok=True)
    return d


def has_cached_geometry():
    d = domain_cache_dir()
    return (os.path.exists(os.path.join(d, "occupancy.npy")) and
            os.path.exists(os.path.join(d, "meta.json")))


def local_to_lonlat(x, y, dom=None):
    if dom is None:
        clat, clon = DOMAIN_CENTER_LAT, DOMAIN_CENTER_LON
    else:
        clat, clon = dom["center_lat"], dom["center_lon"]
    lat_m = 1.0 / 111_320.0
    lon_m = 1.0 / (111_320.0 * math.cos(math.radians(clat)))
    return clon + x * lon_m, clat + y * lat_m
