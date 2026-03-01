"""Fetch and process London building geometry from OpenStreetMap."""
import os
import json
import numpy as np
from pathlib import Path

import config


def _local_to_lonlat(x, y):
    """Convert local metres to lon/lat using linear approximation (accurate to ~cm over 400 m)."""
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * np.cos(np.radians(config.DOMAIN_CENTER_LAT)))
    return config.DOMAIN_CENTER_LON + x * lon_per_m, config.DOMAIN_CENTER_LAT + y * lat_per_m


def _lonlat_to_local(lon, lat):
    """Inverse of _local_to_lonlat."""
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * np.cos(np.radians(config.DOMAIN_CENTER_LAT)))
    return (lon - config.DOMAIN_CENTER_LON) / lon_per_m, (lat - config.DOMAIN_CENTER_LAT) / lat_per_m


def _parse_height(row):
    """Extract building height from an OSM row."""
    import math
    for key in ("height", "building:height"):
        val = row.get(key)
        if val is not None:
            try:
                h = float(str(val).replace("m", "").strip())
                if math.isfinite(h) and h > 0:
                    return h
            except (ValueError, TypeError):
                pass
    for key in ("building:levels",):
        val = row.get(key)
        if val is not None:
            try:
                h = float(str(val)) * 3.5
                if math.isfinite(h) and h > 0:
                    return h
            except (ValueError, TypeError):
                pass
    return config.DEFAULT_BUILDING_HEIGHT


# ── Fetch from OpenStreetMap ────────────────────────────────────────
def fetch_buildings():
    """Download building footprints from OSM for the configured London area."""
    import osmnx as ox

    print("Fetching building data from OpenStreetMap …")
    gdf = ox.features_from_point(
        (config.DOMAIN_CENTER_LAT, config.DOMAIN_CENTER_LON),
        tags={"building": True},
        dist=config.OSM_FETCH_RADIUS,
    )

    buildings = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        height = _parse_height(row)
        if height < config.MIN_BUILDING_HEIGHT:
            continue
        height = min(height, config.DOMAIN_HEIGHT - 5)

        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else (
            [geom] if geom.geom_type == "Polygon" else []
        )

        for poly in polys:
            coords = np.array(poly.exterior.coords)
            local = np.column_stack(_lonlat_to_local(coords[:, 0], coords[:, 1]))

            if (np.all(local[:, 0] > config.DOMAIN_HALF_X + 10) or
                np.all(local[:, 0] < -config.DOMAIN_HALF_X - 10) or
                np.all(local[:, 1] > config.DOMAIN_HALF_Y + 10) or
                np.all(local[:, 1] < -config.DOMAIN_HALF_Y - 10)):
                continue

            buildings.append({
                "footprint": local.tolist(),
                "height": float(height),
            })

    print(f"  Found {len(buildings)} buildings in domain")
    return buildings


# ── Synthetic fallback ──────────────────────────────────────────────
def create_synthetic_buildings():
    """Generate a London-like grid of buildings (no internet required)."""
    print("Creating synthetic building layout …")
    rng = np.random.RandomState(42)
    buildings = []

    block, street = 60.0, 15.0
    spacing = block + street
    heights = [15, 20, 25, 30, 40, 60, 80, 100, 120]
    probs = [0.20, 0.20, 0.15, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02]

    for bx in np.arange(-config.DOMAIN_HALF_X + 20, config.DOMAIN_HALF_X - 20, spacing):
        for by in np.arange(-config.DOMAIN_HALF_Y + 20, config.DOMAIN_HALF_Y - 20, spacing):
            for _ in range(rng.randint(1, 4)):
                w = rng.uniform(15, 45)
                d = rng.uniform(15, 45)
                h = float(rng.choice(heights, p=probs))
                ox_ = bx + rng.uniform(5, max(6, block - w - 5))
                oy_ = by + rng.uniform(5, max(6, block - d - 5))
                buildings.append({
                    "footprint": [[ox_, oy_], [ox_ + w, oy_],
                                  [ox_ + w, oy_ + d], [ox_, oy_ + d], [ox_, oy_]],
                    "height": h,
                })

    print(f"  Created {len(buildings)} synthetic buildings")
    return buildings


# ── Voxelisation ────────────────────────────────────────────────────
def voxelize_buildings(buildings):
    """Create 3-D occupancy grid and signed distance field."""
    from shapely.geometry import Polygon, Point
    from shapely.prepared import prep
    from scipy.ndimage import distance_transform_edt

    res = config.VOXEL_RESOLUTION
    nx = int(2 * config.DOMAIN_HALF_X / res)
    ny = int(2 * config.DOMAIN_HALF_Y / res)
    nz = int(config.DOMAIN_HEIGHT / res)
    print(f"  Voxelising on {nx}×{ny}×{nz} grid ({res} m resolution) …")

    occupancy = np.zeros((nx, ny, nz), dtype=np.float32)

    for bldg in buildings:
        poly = Polygon(bldg["footprint"])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        prepped = prep(poly)
        h = bldg["height"]
        if not (isinstance(h, (int, float)) and np.isfinite(h) and h > 0):
            continue
        minx, miny, maxx, maxy = poly.bounds
        ix0 = max(0, int((minx + config.DOMAIN_HALF_X) / res))
        ix1 = min(nx, int((maxx + config.DOMAIN_HALF_X) / res) + 1)
        iy0 = max(0, int((miny + config.DOMAIN_HALF_Y) / res))
        iy1 = min(ny, int((maxy + config.DOMAIN_HALF_Y) / res) + 1)
        iz1 = min(nz, int((h + config.BUILDING_HEIGHT_BUFFER) / res))

        for ix in range(ix0, ix1):
            cx = ix * res - config.DOMAIN_HALF_X + res * 0.5
            for iy in range(iy0, iy1):
                cy = iy * res - config.DOMAIN_HALF_Y + res * 0.5
                if prepped.contains(Point(cx, cy)):
                    occupancy[ix, iy, :iz1] = 1.0

    sdf_out = distance_transform_edt(1 - occupancy) * res
    sdf_in = distance_transform_edt(occupancy) * res
    sdf = sdf_out - sdf_in

    pct = occupancy.mean() * 100
    print(f"  Voxelisation complete – building coverage {pct:.1f}%")
    return occupancy, sdf


# ── Voxelise from browser-sampled heightmap ─────────────────────────
def voxelize_from_heightmap(height_grid, sample_res):
    """Convert a 2-D ellipsoidal-height grid (from CesiumJS tile sampling)
    into a 3-D occupancy grid + SDF.

    Parameters
    ----------
    height_grid : ndarray, shape (ny_sample, nx_sample)
        Ellipsoidal heights returned by scene.sampleHeightMostDetailed().
        NaN entries are treated as missing data (interpolated from neighbours).
    sample_res : float
        Spatial resolution of the height grid in metres.
    """
    from scipy.ndimage import distance_transform_edt, uniform_filter
    from scipy.interpolate import RegularGridInterpolator

    ny_s, nx_s = height_grid.shape
    print(f"  Heightmap received: {nx_s}×{ny_s} @ {sample_res} m")

    hmap = height_grid.copy()

    # Fill NaN holes with nearest-neighbour interpolation
    nan_mask = ~np.isfinite(hmap)
    if nan_mask.any():
        from scipy.ndimage import distance_transform_edt as _edt
        _, indices = _edt(nan_mask, return_indices=True)
        hmap[nan_mask] = hmap[tuple(indices[:, nan_mask])]
        print(f"  Filled {nan_mask.sum()} NaN samples via nearest-neighbour")

    # Estimate ground plane as a heavily smoothed version of the surface
    ground = uniform_filter(hmap, size=max(nx_s, ny_s) // 4)
    # Clamp ground to be no higher than the low percentile to avoid
    # rooftops pulling the ground estimate up
    ground_lo = np.nanpercentile(hmap, 8)
    ground = np.clip(ground, ground_lo - 5, ground_lo + 5)

    above_ground = hmap - ground
    above_ground = np.clip(above_ground, 0, config.DOMAIN_HEIGHT)

    # Up-sample to the voxel grid resolution if needed
    res = config.VOXEL_RESOLUTION
    nx = int(2 * config.DOMAIN_HALF_X / res)
    ny = int(2 * config.DOMAIN_HALF_Y / res)
    nz = int(config.DOMAIN_HEIGHT / res)

    if nx_s != nx or ny_s != ny:
        xs_s = np.linspace(0, 1, nx_s)
        ys_s = np.linspace(0, 1, ny_s)
        interp = RegularGridInterpolator((ys_s, xs_s), above_ground,
                                         bounds_error=False, fill_value=0)
        xs_t = np.linspace(0, 1, nx)
        ys_t = np.linspace(0, 1, ny)
        yy, xx = np.meshgrid(ys_t, xs_t, indexing="ij")
        above_ground = interp(np.stack([yy.ravel(), xx.ravel()], axis=1)
                              ).reshape(ny, nx)
        print(f"  Upsampled to {nx}×{ny}")

    # The height_grid is indexed (iy, ix) but occupancy is (ix, iy, iz)
    above_ground = above_ground.T  # now (nx, ny)

    print(f"  Voxelising on {nx}×{ny}×{nz} grid ({res} m) …")
    occupancy = np.zeros((nx, ny, nz), dtype=np.float32)

    for ix in range(nx):
        for iy in range(ny):
            h = above_ground[ix, iy]
            if h > config.HEIGHTMAP_MIN_BUILDING:
                iz_max = min(nz, int(h / res))
                occupancy[ix, iy, :iz_max] = 1.0

    sdf_out = distance_transform_edt(1 - occupancy) * res
    sdf_in = distance_transform_edt(occupancy) * res
    sdf = sdf_out - sdf_in

    pct = occupancy.mean() * 100
    print(f"  Voxelisation complete – {pct:.1f}% coverage")

    # Update ground ellipsoid height from the actual sampled data
    config.GROUND_ELLIPSOID_HEIGHT = float(ground_lo)
    print(f"  Ground ellipsoid height updated to {ground_lo:.1f} m")

    return occupancy, sdf


def save_heightmap_geometry(occupancy, sdf):
    """Save heightmap-derived geometry to both the legacy paths AND the
    domain-keyed cache so subsequent runs with the same area skip the scan."""
    payload = {
        "buildings": [],
        "source": "heightmap",
        "domain": {
            "half_x": config.DOMAIN_HALF_X,
            "half_y": config.DOMAIN_HALF_Y,
            "height": config.DOMAIN_HEIGHT,
            "center_lat": config.DOMAIN_CENTER_LAT,
            "center_lon": config.DOMAIN_CENTER_LON,
            "ground_ellipsoid_height": config.GROUND_ELLIPSOID_HEIGHT,
        },
    }

    for d in (config.BUILDINGS_DIR, config.DOMAIN_DIR):
        os.makedirs(d, exist_ok=True)

    # Legacy location (used by load_geometry)
    with open(os.path.join(config.BUILDINGS_DIR, "buildings.json"), "w") as f:
        json.dump(payload, f)
    np.save(os.path.join(config.DOMAIN_DIR, "occupancy.npy"), occupancy)
    np.save(os.path.join(config.DOMAIN_DIR, "sdf.npy"), sdf)

    # Persistent domain-keyed cache
    cache = config.domain_cache_dir()
    with open(os.path.join(cache, "buildings.json"), "w") as f:
        json.dump(payload, f)
    np.save(os.path.join(cache, "occupancy.npy"), occupancy)
    np.save(os.path.join(cache, "sdf.npy"), sdf)

    print(f"  Geometry saved to {config.DATA_DIR} + cache {cache}")


# ── I/O helpers ─────────────────────────────────────────────────────

def save_geometry(buildings, occupancy, sdf):
    for d in (config.BUILDINGS_DIR, config.DOMAIN_DIR):
        os.makedirs(d, exist_ok=True)
    payload = {
        "buildings": buildings,
        "domain": {
            "half_x": config.DOMAIN_HALF_X,
            "half_y": config.DOMAIN_HALF_Y,
            "height": config.DOMAIN_HEIGHT,
            "center_lat": config.DOMAIN_CENTER_LAT,
            "center_lon": config.DOMAIN_CENTER_LON,
        },
    }
    with open(os.path.join(config.BUILDINGS_DIR, "buildings.json"), "w") as f:
        json.dump(payload, f)
    np.save(os.path.join(config.DOMAIN_DIR, "occupancy.npy"), occupancy)
    np.save(os.path.join(config.DOMAIN_DIR, "sdf.npy"), sdf)
    print(f"  Geometry saved to {config.DATA_DIR}")


def load_geometry():
    """Load from the legacy location."""
    with open(os.path.join(config.BUILDINGS_DIR, "buildings.json")) as f:
        data = json.load(f)
    occ = np.load(os.path.join(config.DOMAIN_DIR, "occupancy.npy"))
    sdf = np.load(os.path.join(config.DOMAIN_DIR, "sdf.npy"))
    return data, occ, sdf


def load_cached_geometry():
    """Load from the domain-keyed cache and also copy to the legacy paths."""
    cache = config.domain_cache_dir()
    with open(os.path.join(cache, "buildings.json")) as f:
        data = json.load(f)
    occ = np.load(os.path.join(cache, "occupancy.npy"))
    sdf = np.load(os.path.join(cache, "sdf.npy"))

    # Restore ground height
    geh = data.get("domain", {}).get("ground_ellipsoid_height")
    if geh is not None:
        config.GROUND_ELLIPSOID_HEIGHT = geh

    # Copy to legacy paths so the rest of the pipeline works unchanged
    for d in (config.BUILDINGS_DIR, config.DOMAIN_DIR):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(config.BUILDINGS_DIR, "buildings.json"), "w") as f:
        json.dump(data, f)
    np.save(os.path.join(config.DOMAIN_DIR, "occupancy.npy"), occ)
    np.save(os.path.join(config.DOMAIN_DIR, "sdf.npy"), sdf)

    return data, occ, sdf


def run():
    try:
        buildings = fetch_buildings()
    except Exception as exc:
        print(f"  OSM fetch failed ({exc}); falling back to synthetic buildings.")
        buildings = create_synthetic_buildings()
    occ, sdf = voxelize_buildings(buildings)
    save_geometry(buildings, occ, sdf)
    return buildings, occ, sdf


if __name__ == "__main__":
    run()
