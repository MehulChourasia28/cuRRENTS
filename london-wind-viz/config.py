"""Configuration for London wind visualization system."""
import os

# ===================== Domain Configuration =====================
DOMAIN_CENTER_LAT = 51.5135
DOMAIN_CENTER_LON = -0.0850

DOMAIN_HALF_X = 200.0   # East-West half-extent in metres (400 m total)
DOMAIN_HALF_Y = 200.0   # North-South half-extent
DOMAIN_HEIGHT = 150.0    # Vertical extent

VOXEL_RESOLUTION = 2.0   # metres per voxel

# ===================== Building Configuration =====================
DEFAULT_BUILDING_HEIGHT = 15.0
MIN_BUILDING_HEIGHT = 3.0
BUILDING_HEIGHT_BUFFER = 12.0
BUILDING_H_DILATE = 3
OSM_FETCH_RADIUS = 250

# ===================== Wind / LBM =====================
WIND_SPEED = 10.0              # m/s reference wind speed
WIND_DIRECTIONS = [0, 270]     # only 2 angles for now
LBM_GRID_RES = 4.0             # metres – LBM evaluation grid resolution

# ===================== Streamline Configuration =====================
STREAMLINE_DT = 0.5
STREAMLINE_MAX_STEPS = 1000
STREAMLINE_MIN_SPEED = 0.5
STREAMLINE_MIN_LENGTH = 20.0

SEED_INLET_COUNT = 150
SEED_BUILDING_COUNT = 300
SEED_STREET_COUNT = 400

OPACITY_MIN_SPEED = 1.0
OPACITY_MAX_SPEED = 15.0

# ===================== Heightmap Sampling =====================
HEIGHTMAP_SAMPLE_RES = 4.0
HEIGHTMAP_MIN_BUILDING = 3.0

# ===================== Server =====================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080

# ===================== Paths =====================
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, "data")
BUILDINGS_DIR = os.path.join(DATA_DIR, "buildings")
DOMAIN_DIR = os.path.join(DATA_DIR, "domain")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
STREAMLINE_DIR = os.path.join(DATA_DIR, "streamlines")

GROUND_ELLIPSOID_HEIGHT = 58.0

CESIUM_ION_TOKEN = os.environ.get(
    "CESIUM_ION_TOKEN",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIyNjJhMjEyZS01NmZmLTQzMDItYWRhZi01NDhmNTE0ZDdkNDAiLCJpZCI6Mzk1ODMyLCJpYXQiOjE3NzIyODM0NDV9.j0H8VaqE5OgIzcH-_Sk6RWVFm3JofXpz7NqPRrrLwbs",
)
