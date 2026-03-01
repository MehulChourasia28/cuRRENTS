"""
Streamline computation from a predicted velocity field.

Implements:
  • Hard-masked velocity field (zero inside buildings + 1-voxel buffer)
  • SDF-adaptive step sizing near building surfaces
  • Surface-tangent deflection so streamlines wrap around buildings
  • RK4 integration through the 3-D velocity field
  • Strategic seed-point placement (inlet / building-surface / street-level)
  • Turbo-colourmap colour mapping with speed-based alpha
  • Export to CesiumJS-ready JSON
"""
import os
import json
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, distance_transform_edt

import config


# ═══════════════════════════════════════════════════════════════════
# Colour mapping
# ═══════════════════════════════════════════════════════════════════

def turbo_colormap(t: np.ndarray) -> np.ndarray:
    """Map t ∈ [0, 1] → RGB ∈ [0, 1] using the Turbo colourmap."""
    try:
        from matplotlib.cm import get_cmap
        return get_cmap("turbo")(np.clip(t, 0, 1))[..., :3]
    except ImportError:
        r = np.clip(np.where(t < 0.5, 2 * t, 2 - 2 * t) + 0.1, 0, 1)
        g = np.clip(1 - 2 * np.abs(t - 0.5), 0, 1)
        b = np.clip(1.5 - 2 * t, 0, 1)
        return np.stack([r, g, b], -1)


# ═══════════════════════════════════════════════════════════════════
# Velocity interpolation — with hard building mask
# ═══════════════════════════════════════════════════════════════════

def _build_interps(coords, velocities, shape, occupancy):
    """
    Build tri-linear velocity interpolators after zeroing out any velocity
    inside buildings (+ a 1-voxel dilated safety margin).
    """
    nx, ny, nz = shape
    xs = np.linspace(-config.DOMAIN_HALF_X, config.DOMAIN_HALF_X, nx)
    ys = np.linspace(-config.DOMAIN_HALF_Y, config.DOMAIN_HALF_Y, ny)
    zs = np.linspace(0, config.DOMAIN_HEIGHT, nz)

    vel = velocities.copy()

    occ_res = config.VOXEL_RESOLUTION
    blocked = binary_dilation(occupancy > 0.5,
                              iterations=config.BUILDING_H_DILATE).astype(np.float32)

    ix = np.clip(((coords[:, 0] + config.DOMAIN_HALF_X) / occ_res).astype(int),
                 0, blocked.shape[0] - 1)
    iy = np.clip(((coords[:, 1] + config.DOMAIN_HALF_Y) / occ_res).astype(int),
                 0, blocked.shape[1] - 1)
    iz = np.clip((coords[:, 2] / occ_res).astype(int),
                 0, blocked.shape[2] - 1)

    inside = blocked[ix, iy, iz] > 0.5
    vel[inside] = 0.0

    masked_pct = inside.sum() / len(inside) * 100
    print(f"    Hard-masked {masked_pct:.1f}% of velocity grid points (building + buffer)")

    interps = []
    for c in range(3):
        vol = vel[:, c].reshape(nx, ny, nz)
        interps.append(
            RegularGridInterpolator((xs, ys, zs), vol,
                                    bounds_error=False, fill_value=0.0))
    return interps


def _build_sdf_interp(occupancy):
    """Build an SDF interpolator from the occupancy grid (positive = outside)."""
    occ = occupancy > 0.5
    sdf_out = distance_transform_edt(~occ) * config.VOXEL_RESOLUTION
    sdf_in = distance_transform_edt(occ) * config.VOXEL_RESOLUTION
    sdf = sdf_out - sdf_in

    nnx, nny, nnz = sdf.shape
    xs = np.linspace(-config.DOMAIN_HALF_X, config.DOMAIN_HALF_X, nnx)
    ys = np.linspace(-config.DOMAIN_HALF_Y, config.DOMAIN_HALF_Y, nny)
    zs = np.linspace(0, config.DOMAIN_HEIGHT, nnz)
    return RegularGridInterpolator((xs, ys, zs), sdf,
                                   bounds_error=False, fill_value=100.0)


def _vel(interps, p):
    pt = p.reshape(1, -1)
    return np.array([float(f(pt)) for f in interps])


def _sdf_at(sdf_interp, p):
    try:
        return float(sdf_interp(p.reshape(1, -1)))
    except Exception:
        return 100.0


# ═══════════════════════════════════════════════════════════════════
# RK4 + building-aware tracer
# ═══════════════════════════════════════════════════════════════════

_OCC_RES = config.VOXEL_RESOLUTION
_HX = config.DOMAIN_HALF_X
_HY = config.DOMAIN_HALF_Y


def _euler(interps, pos, dt):
    """Simple Euler step — used when close to walls where RK4 substeps
    could stray into buildings."""
    return pos + _vel(interps, pos) * dt


def _rk4(interps, pos, dt):
    k1 = _vel(interps, pos)
    k2 = _vel(interps, pos + 0.5 * dt * k1)
    k3 = _vel(interps, pos + 0.5 * dt * k2)
    k4 = _vel(interps, pos + dt * k3)
    return pos + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _inside(pos, occ):
    ix = int((pos[0] + _HX) / _OCC_RES)
    iy = int((pos[1] + _HY) / _OCC_RES)
    iz = int(pos[2] / _OCC_RES)
    if 0 <= ix < occ.shape[0] and 0 <= iy < occ.shape[1] and 0 <= iz < occ.shape[2]:
        return occ[ix, iy, iz] > 0.5
    return False


def _segment_clear(p0, p1, occ):
    """Ray-march the segment p0→p1 at ~1 m spacing; return False if any
    sample lands inside a building."""
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 0.5:
        return not _inside(p1, occ)
    n_checks = max(3, int(length / 1.0))
    for i in range(1, n_checks + 1):
        pt = p0 + d * (i / n_checks)
        if _inside(pt, occ):
            return False
    return True


def _sdf_gradient(sdf_interp, pos):
    eps = 1.0
    gx = _sdf_at(sdf_interp, pos + [eps, 0, 0]) - _sdf_at(sdf_interp, pos - [eps, 0, 0])
    gy = _sdf_at(sdf_interp, pos + [0, eps, 0]) - _sdf_at(sdf_interp, pos - [0, eps, 0])
    gz = _sdf_at(sdf_interp, pos + [0, 0, eps]) - _sdf_at(sdf_interp, pos - [0, 0, eps])
    g = np.array([gx, gy, gz]) / (2 * eps)
    gn = np.linalg.norm(g)
    return (g / gn) if gn > 1e-8 else np.zeros(3)


def _try_step(interps, sdf_interp, pos, occ, spd):
    """Compute a single forward step that does not cross any building.

    Strategy:
      1. Pick an adaptive dt (shrinks quadratically with SDF proximity).
      2. Take the step (Euler when very close, RK4 otherwise).
      3. Validate the segment with a ray-march.
      4. If it fails, halve dt and retry up to 5 times.
      5. If all retries fail, attempt tangent-plane deflection.

    Returns (new_pos, ok).
    """
    sdf_val = max(_sdf_at(sdf_interp, pos), 0.5)

    # Base dt capped by speed; further limited proportional to SDF²
    base_dt = min(config.STREAMLINE_DT, 2.0 / (spd + 1e-6))
    proximity_dt = (sdf_val ** 2) * 0.02 / (spd + 1e-6)
    dt = min(base_dt, max(0.01, proximity_dt)) if sdf_val < 25.0 else base_dt

    use_euler = sdf_val < 6.0

    for _ in range(5):
        step_fn = _euler if use_euler else _rk4
        npos = step_fn(interps, pos, dt)

        if (abs(npos[0]) > _HX or abs(npos[1]) > _HY
                or npos[2] < 0 or npos[2] > config.DOMAIN_HEIGHT):
            return npos, False  # out of bounds

        if _segment_clear(pos, npos, occ) and _sdf_at(sdf_interp, npos) > 0:
            return npos, True

        dt *= 0.4  # shrink and retry

    # All retries failed — try tangent-plane deflection
    v = _vel(interps, pos)
    n = _sdf_gradient(sdf_interp, pos)
    if np.linalg.norm(n) < 1e-8:
        return pos, False
    v_tang = v - np.dot(v, n) * n
    if np.linalg.norm(v_tang) < config.STREAMLINE_MIN_SPEED:
        return pos, False

    for frac in (0.25, 0.1, 0.03):
        npos = pos + v_tang * base_dt * frac
        if (_sdf_at(sdf_interp, npos) > 1.0
                and _segment_clear(pos, npos, occ)):
            return npos, True

    return pos, False


def _trace(interps, sdf_interp, start, occ):
    """Trace one streamline; each step is ray-march validated."""
    pos = start.copy()
    path = [pos.copy()]
    speeds = [np.linalg.norm(_vel(interps, pos))]
    stall = 0

    for _ in range(config.STREAMLINE_MAX_STEPS):
        v = _vel(interps, pos)
        spd = np.linalg.norm(v)
        if spd < config.STREAMLINE_MIN_SPEED:
            break

        npos, ok = _try_step(interps, sdf_interp, pos, occ, spd)
        if not ok:
            stall += 1
            if stall > 3:
                break
            continue
        stall = 0

        pos = npos
        path.append(pos.copy())
        speeds.append(np.linalg.norm(_vel(interps, pos)))

    return np.asarray(path), np.asarray(speeds)


# ═══════════════════════════════════════════════════════════════════
# Seed-point strategies
# ═══════════════════════════════════════════════════════════════════

def _seeds(wind_deg, occ):
    angle = math.radians(wind_deg)
    ca, sa = math.cos(angle), math.sin(angle)
    rng = np.random.RandomState(int(wind_deg))
    res = config.VOXEL_RESOLUTION
    seeds = []

    # 1) Inlet face — mostly below 50 m to keep streamlines between buildings
    n = config.SEED_INLET_COUNT
    max_inlet_z = 50.0
    if abs(ca) > 0.1:
        x0 = -np.sign(ca) * config.DOMAIN_HALF_X * 0.95
        for y, z in zip(rng.uniform(-0.9, 0.9, n // 2) * config.DOMAIN_HALF_Y,
                        rng.uniform(2, max_inlet_z, n // 2)):
            seeds.append(np.array([x0, y, z]))
    if abs(sa) > 0.1:
        y0 = -np.sign(sa) * config.DOMAIN_HALF_Y * 0.95
        for x, z in zip(rng.uniform(-0.9, 0.9, n // 2) * config.DOMAIN_HALF_X,
                        rng.uniform(2, max_inlet_z, n // 2)):
            seeds.append(np.array([x, y0, z]))

    # 2) Near building surfaces (offset outward by a few metres)
    surface = binary_dilation(occ > 0.5) & (occ < 0.5)
    spts = np.argwhere(surface).astype(np.float64)
    if len(spts):
        spts[:, 0] = spts[:, 0] * res - config.DOMAIN_HALF_X
        spts[:, 1] = spts[:, 1] * res - config.DOMAIN_HALF_Y
        spts[:, 2] = spts[:, 2] * res
        chosen = spts[rng.choice(len(spts),
                                 min(config.SEED_BUILDING_COUNT, len(spts)),
                                 replace=False)]
        for pt in chosen:
            offset = rng.randn(3) * 3.0
            s = pt + offset
            s[2] = max(1.0, s[2])
            ix = int((s[0] + config.DOMAIN_HALF_X) / res)
            iy = int((s[1] + config.DOMAIN_HALF_Y) / res)
            iz = int(s[2] / res)
            if (0 <= ix < occ.shape[0] and 0 <= iy < occ.shape[1]
                    and 0 <= iz < occ.shape[2] and occ[ix, iy, iz] < 0.5):
                seeds.append(s)

    # 3) Street level
    for _ in range(config.SEED_STREET_COUNT):
        x = rng.uniform(-0.8, 0.8) * config.DOMAIN_HALF_X
        y = rng.uniform(-0.8, 0.8) * config.DOMAIN_HALF_Y
        z = rng.uniform(1.5, 5.0)
        ix = int((x + config.DOMAIN_HALF_X) / res)
        iy = int((y + config.DOMAIN_HALF_Y) / res)
        iz = int(z / res)
        if (0 <= ix < occ.shape[0] and 0 <= iy < occ.shape[1]
                and 0 <= iz < occ.shape[2] and occ[ix, iy, iz] < 0.5):
            seeds.append(np.array([x, y, z]))

    return seeds


# ═══════════════════════════════════════════════════════════════════
# Compute all streamlines for one wind direction
# ═══════════════════════════════════════════════════════════════════

def compute(coords, velocities, shape, occ, wind_deg):
    print(f"  Computing streamlines for {wind_deg}° …")
    interps = _build_interps(coords, velocities, shape, occ)
    sdf_interp = _build_sdf_interp(occ)
    seeds = _seeds(wind_deg, occ)
    print(f"    {len(seeds)} seed points")

    results = []
    for s in seeds:
        if _inside(s, occ):
            continue
        pos, spd = _trace(interps, sdf_interp, s, occ)
        if len(pos) < 3:
            continue
        length = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        if length < config.STREAMLINE_MIN_LENGTH:
            continue
        results.append({"positions": pos, "speeds": spd})

    print(f"    {len(results)} valid streamlines")
    return results


# ═══════════════════════════════════════════════════════════════════
# Export to CesiumJS JSON
# ═══════════════════════════════════════════════════════════════════

def _local_to_lonlat(x, y):
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(config.DOMAIN_CENTER_LAT)))
    return (config.DOMAIN_CENTER_LON + x * lon_per_m,
            config.DOMAIN_CENTER_LAT + y * lat_per_m)


def to_cesium_json(streamlines, wind_deg):
    if not streamlines:
        return {"wind_angle": wind_deg, "wind_speed": config.WIND_SPEED,
                "speed_range": [0, 1], "streamlines": []}
    all_spd = np.concatenate([s["speeds"] for s in streamlines])
    smin = float(np.percentile(all_spd, 5))
    smax = float(np.percentile(all_spd, 95))
    srange = max(smax - smin, 0.1)

    BASE_ALT = config.GROUND_ELLIPSOID_HEIGHT

    payload = {
        "wind_angle": wind_deg,
        "wind_speed": config.WIND_SPEED,
        "speed_range": [smin, smax],
        "streamlines": [],
    }

    for sl in streamlines:
        pos = sl["positions"]
        spd = sl["speeds"]
        flat = []
        for p in pos:
            lon, lat = _local_to_lonlat(p[0], p[1])
            flat.extend([round(lon, 7), round(lat, 7), round(p[2] + BASE_ALT, 2)])

        t = np.clip((spd - smin) / srange, 0, 1)
        rgb = turbo_colormap(t)
        alpha = np.clip((spd - config.OPACITY_MIN_SPEED) /
                        (config.OPACITY_MAX_SPEED - config.OPACITY_MIN_SPEED),
                        0.10, 1.0)
        rgba = []
        for i in range(len(spd)):
            rgba.extend([int(rgb[i, 0] * 255), int(rgb[i, 1] * 255),
                         int(rgb[i, 2] * 255), int(alpha[i] * 255)])

        payload["streamlines"].append({
            "positions": flat,
            "colors": rgba,
            "num_points": len(pos),
        })
    return payload


def save(data, wind_deg):
    os.makedirs(config.STREAMLINE_DIR, exist_ok=True)
    path = os.path.join(config.STREAMLINE_DIR, f"streamlines_{wind_deg}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"    Saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# Top-level runner
# ═══════════════════════════════════════════════════════════════════

def run_from_field(occupancy, coords, vel, shape):
    """Compute streamlines from a pre-computed (averaged) velocity field
    and save as ``streamlines_combined.json``."""
    print("\n── Combined averaged wind field ──")
    sls = compute(coords, vel, shape, occupancy, 0)
    data = to_cesium_json(sls, 0)

    os.makedirs(config.STREAMLINE_DIR, exist_ok=True)
    path = os.path.join(config.STREAMLINE_DIR, "streamlines_combined.json")
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"    Saved → {path}  ({len(sls)} streamlines)")

    meta = {
        "type": "combined",
        "domain": {
            "center_lat": config.DOMAIN_CENTER_LAT,
            "center_lon": config.DOMAIN_CENTER_LON,
            "half_x": config.DOMAIN_HALF_X,
            "half_y": config.DOMAIN_HALF_Y,
            "height": config.DOMAIN_HEIGHT,
        },
    }
    with open(os.path.join(config.STREAMLINE_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f)


def run(occupancy, predict_fn):
    """Generate streamlines per-direction (legacy interface)."""
    angles = getattr(config, "WIND_DIRECTIONS", [0, 270])
    for deg in angles:
        print(f"\n── Wind {deg}° ──")
        coords, vel, shape = predict_fn(deg)
        sls = compute(coords, vel, shape, occupancy, deg)
        data = to_cesium_json(sls, deg)
        save(data, deg)

    meta = {
        "available_angles": angles,
        "domain": {
            "center_lat": config.DOMAIN_CENTER_LAT,
            "center_lon": config.DOMAIN_CENTER_LON,
            "half_x": config.DOMAIN_HALF_X,
            "half_y": config.DOMAIN_HALF_Y,
            "height": config.DOMAIN_HEIGHT,
        },
    }
    with open(os.path.join(config.STREAMLINE_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f)
