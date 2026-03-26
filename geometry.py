"""Building geometry — voxelise from CesiumJS heightmap or from OSM."""
import os
import json
import math
import numpy as np

import config


# ── Heightmap → 3-D occupancy ─────────────────────────────────────────

def voxelize_from_heightmap(height_grid, sample_res, domain):
    """Convert a 2-D ellipsoidal-height grid into a 3-D occupancy grid.

    Parameters
    ----------
    height_grid : ndarray (ny_s, nx_s)  ellipsoidal heights from Cesium
    sample_res  : float   grid spacing in metres
    domain      : dict    config snapshot (half_x, half_y, height, …)

    Returns
    -------
    occupancy : float32 (nx, ny, nz)   1 = solid, 0 = air
    ground_h  : float                  median ground ellipsoidal height
    """
    from scipy.ndimage import uniform_filter, distance_transform_edt
    from scipy.interpolate import RegularGridInterpolator

    ny_s, nx_s = height_grid.shape
    half_x = domain["half_x"]
    half_y = domain["half_y"]
    height = domain["height"]

    print(f"  Heightmap {nx_s}×{ny_s} @ {sample_res} m")

    hmap = height_grid.copy().astype(np.float64)

    # Fill NaN with nearest-neighbour
    bad = ~np.isfinite(hmap)
    if bad.any():
        _, idx = distance_transform_edt(bad, return_indices=True)
        hmap[bad] = hmap[tuple(idx[:, bad])]
        print(f"  Filled {bad.sum()} NaN samples")

    # Ground = low-percentile smoothed surface (avoids rooftops pulling it up)
    ground_lo = float(np.nanpercentile(hmap, 8))
    ground    = np.clip(uniform_filter(hmap, size=max(nx_s, ny_s) // 4),
                        ground_lo - 3, ground_lo + 3)
    above = np.clip(hmap - ground, 0, height)

    # Upsample to voxel grid
    res = config.VOXEL_RESOLUTION
    nx  = max(2, int(2 * half_x / res))
    ny  = max(2, int(2 * half_y / res))
    nz  = max(2, int(height / res))

    if nx_s != nx or ny_s != ny:
        xs_s = np.linspace(0, 1, nx_s)
        ys_s = np.linspace(0, 1, ny_s)
        interp = RegularGridInterpolator((ys_s, xs_s), above,
                                         bounds_error=False, fill_value=0.0)
        xs_t = np.linspace(0, 1, nx)
        ys_t = np.linspace(0, 1, ny)
        yy, xx = np.meshgrid(ys_t, xs_t, indexing="ij")
        above = interp(np.stack([yy.ravel(), xx.ravel()], axis=1)).reshape(ny, nx)
        print(f"  Resampled to {nx}×{ny}")

    above = above.T  # (nx, ny)

    occupancy = np.zeros((nx, ny, nz), dtype=np.float32)
    for ix in range(nx):
        for iy in range(ny):
            h = above[ix, iy]
            if h > config.HEIGHTMAP_MIN_BUILDING:
                iz_max = min(nz, int(h / res))
                occupancy[ix, iy, :iz_max] = 1.0

    pct = occupancy.mean() * 100
    print(f"  Voxelised {nx}×{ny}×{nz} — {pct:.1f}% covered")
    return occupancy, ground_lo


# ── Cache I/O ──────────────────────────────────────────────────────────

def save_geometry(occupancy, domain, ground_h):
    """Save geometry to domain-keyed cache and legacy domain dir."""
    config._ensure_dirs()
    meta = {
        "domain": domain,
        "ground_ellipsoid_height": ground_h,
        "source": "heightmap",
    }
    cache = config.domain_cache_dir()
    np.save(os.path.join(cache, "occupancy.npy"), occupancy)
    with open(os.path.join(cache, "meta.json"), "w") as f:
        json.dump(meta, f)
    # Also write to legacy domain dir so LBM subprocess can find it
    np.save(os.path.join(config.DOMAIN_DIR, "occupancy.npy"), occupancy)
    print(f"  Geometry cached → {cache}")


def load_cached_geometry():
    """Load from domain-keyed cache.  Returns (occupancy, meta)."""
    cache = config.domain_cache_dir()
    occ   = np.load(os.path.join(cache, "occupancy.npy"))
    with open(os.path.join(cache, "meta.json")) as f:
        meta = json.load(f)
    # Refresh legacy path
    config._ensure_dirs()
    np.save(os.path.join(config.DOMAIN_DIR, "occupancy.npy"), occ)
    return occ, meta
