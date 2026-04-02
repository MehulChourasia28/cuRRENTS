import os
import json
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, distance_transform_edt

import config


def _turbo(t):
    try:
        from matplotlib.cm import get_cmap
        return get_cmap("turbo")(np.clip(t, 0, 1))[..., :3]
    except Exception:
        r = np.clip(1.5 * t, 0, 1)
        g = np.clip(1 - 2 * abs(t - 0.5), 0, 1)
        b = np.clip(1.5 - 1.5 * t, 0, 1)
        return np.stack([r, g, b], -1)


def _build_interps(coords, vel, shape, occupancy, domain):
    nx, ny, nz = shape
    half_x = domain["half_x"]
    half_y = domain["half_y"]
    height = domain["height"]
    res    = config.VOXEL_RESOLUTION

    xs = np.linspace(-half_x, half_x, nx)
    ys = np.linspace(-half_y, half_y, ny)
    zs = np.linspace(0, height, nz)

    v = vel.copy()

    # Zero velocity inside buildings so streamlines naturally deflect away
    blocked = binary_dilation(occupancy > 0.5,
                              iterations=config.BUILDING_H_DILATE).astype(bool)
    ix = np.clip(((coords[:, 0] + half_x) / res).astype(int), 0, blocked.shape[0]-1)
    iy = np.clip(((coords[:, 1] + half_y) / res).astype(int), 0, blocked.shape[1]-1)
    iz = np.clip(( coords[:, 2]           / res).astype(int), 0, blocked.shape[2]-1)
    inside = blocked[ix, iy, iz]
    v[inside] = 0.0
    print(f"    Masked {inside.sum()} of {len(inside)} grid points (buildings)")

    interps = []
    for c in range(3):
        vol = v[:, c].reshape(nx, ny, nz)
        interps.append(RegularGridInterpolator(
            (xs, ys, zs), vol, bounds_error=False, fill_value=0.0))
    return interps


def _build_sdf(occupancy, domain):
    res    = config.VOXEL_RESOLUTION
    half_x = domain["half_x"]
    half_y = domain["half_y"]
    height = domain["height"]
    occ    = occupancy > 0.5
    sdf    = (distance_transform_edt(~occ) - distance_transform_edt(occ)) * res
    nnx, nny, nnz = sdf.shape
    xs = np.linspace(-half_x, half_x, nnx)
    ys = np.linspace(-half_y, half_y, nny)
    zs = np.linspace(0, height, nnz)
    return RegularGridInterpolator((xs, ys, zs), sdf,
                                   bounds_error=False, fill_value=50.0)


def _vel_at(interps, p):
    pt = p.reshape(1, -1)
    return np.array([f(pt).item() for f in interps])


def _sdf_at(sdf_interp, p):
    return sdf_interp(p.reshape(1, -1)).item()


def _rk4_step(interps, p, dt, occupancy=None, domain=None):
    def _v(pt):
        if occupancy is not None and _inside_building(pt, occupancy, domain):
            return np.zeros(3)
        return _vel_at(interps, pt)
    k1 = _v(p)
    k2 = _v(p + 0.5*dt*k1)
    k3 = _v(p + 0.5*dt*k2)
    k4 = _v(p + dt*k3)
    return p + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def _in_bounds(p, domain):
    hx = domain["half_x"]
    hy = domain["half_y"]
    ht = domain["height"]
    return (abs(p[0]) <= hx and abs(p[1]) <= hy and 0 <= p[2] <= ht)


def _inside_building(p, occupancy, domain):
    res    = config.VOXEL_RESOLUTION
    half_x = domain["half_x"]
    half_y = domain["half_y"]
    ix = int((p[0] + half_x) / res)
    iy = int((p[1] + half_y) / res)
    iz = int(p[2] / res)
    nx, ny, nz = occupancy.shape
    if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
        return occupancy[ix, iy, iz] > 0.5
    return False


def _trace(interps, sdf_interp, start, occupancy, domain):
    pos    = start.copy()
    path   = [pos.copy()]
    speeds = [np.linalg.norm(_vel_at(interps, pos))]
    stall  = 0

    _MAX_STEP_M = config.STREAMLINE_DT

    for _ in range(config.STREAMLINE_MAX_STEPS):
        v   = _vel_at(interps, pos)
        spd = np.linalg.norm(v)
        if spd < config.STREAMLINE_MIN_SPEED:
            break

        sdf_val = _sdf_at(sdf_interp, pos)
        if sdf_val < config.VOXEL_RESOLUTION:
            break

        # Limit step to 25% of SDF distance so we never jump through thin walls
        step_m = min(_MAX_STEP_M, max(0.1, sdf_val * 0.25))
        dt     = step_m / (spd + 1e-6)

        npos = _rk4_step(interps, pos, dt, occupancy, domain)

        if not _in_bounds(npos, domain):
            break
        if _inside_building(npos, occupancy, domain) or _sdf_at(sdf_interp, npos) < 0.0:
            npos_half = _rk4_step(interps, pos, dt * 0.25, occupancy, domain)
            if _inside_building(npos_half, occupancy, domain):
                stall += 1
                if stall > 3:
                    break
                continue
            npos = npos_half

        if _sdf_at(sdf_interp, npos) < config.VOXEL_RESOLUTION:
            break

        stall = 0
        pos   = npos
        path.append(pos.copy())
        speeds.append(np.linalg.norm(_vel_at(interps, pos)))

    return np.asarray(path), np.asarray(speeds)


def _make_seeds(occupancy, domain, wind_deg):
    rng    = np.random.RandomState(42)
    res    = config.VOXEL_RESOLUTION
    half_x = domain["half_x"]
    half_y = domain["half_y"]
    height = domain["height"]
    rad    = math.radians(wind_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    seeds  = []

    def _free(p):
        return not _inside_building(p, occupancy, domain)

    # Inlet face
    n_in = config.N_SEEDS_INLET
    if abs(ca) >= abs(sa):
        x0 = -math.copysign(half_x * 0.97, ca)
        ys_ = rng.uniform(-half_y * 0.9, half_y * 0.9, n_in)
        zs_ = np.concatenate([
            rng.uniform(2, min(40, height * 0.35), n_in * 2 // 3),
            rng.uniform(min(40, height * 0.35), height * 0.7, n_in // 3),
        ])
        for y, z in zip(ys_, zs_[:len(ys_)]):
            p = np.array([x0, y, z])
            if _free(p):
                seeds.append(p)
    else:
        y0 = -math.copysign(half_y * 0.97, sa)
        xs_ = rng.uniform(-half_x * 0.9, half_x * 0.9, n_in)
        zs_ = np.concatenate([
            rng.uniform(2, min(40, height * 0.35), n_in * 2 // 3),
            rng.uniform(min(40, height * 0.35), height * 0.7, n_in // 3),
        ])
        for x, z in zip(xs_, zs_[:len(xs_)]):
            p = np.array([x, y0, z])
            if _free(p):
                seeds.append(p)

    # Uniform grid to cover the downwind half / wake gaps
    grid_step = max(half_x, half_y) / 8.0
    for gx in np.arange(-half_x * 0.85, half_x * 0.85, grid_step):
        for gy in np.arange(-half_y * 0.85, half_y * 0.85, grid_step):
            for gz in [rng.uniform(3, 8), rng.uniform(15, 40)]:
                p = np.array([gx + rng.uniform(-grid_step*0.3, grid_step*0.3),
                               gy + rng.uniform(-grid_step*0.3, grid_step*0.3),
                               gz])
                if _in_bounds(p, domain) and _free(p):
                    seeds.append(p)

    # Just outside building surfaces
    surface = binary_dilation(occupancy > 0.5) & (occupancy < 0.5)
    pts = np.argwhere(surface).astype(np.float64)
    if len(pts):
        pts[:, 0] = pts[:, 0] * res - half_x
        pts[:, 1] = pts[:, 1] * res - half_y
        pts[:, 2] = pts[:, 2] * res
        chosen = pts[rng.choice(len(pts),
                                min(config.N_SEEDS_BUILDING, len(pts)),
                                replace=False)]
        for pt in chosen:
            off = rng.randn(3) * 2.5
            off[2] = abs(off[2])
            s = pt + off
            s[2] = max(1.0, s[2])
            if _in_bounds(s, domain) and _free(s):
                seeds.append(s)

    # Street-level random
    for _ in range(config.N_SEEDS_STREET):
        x = rng.uniform(-half_x * 0.85, half_x * 0.85)
        y = rng.uniform(-half_y * 0.85, half_y * 0.85)
        z = rng.uniform(2.0, 8.0)
        p = np.array([x, y, z])
        if _free(p):
            seeds.append(p)

    return seeds


def _to_cesium(streamlines, domain):
    if not streamlines:
        return {"streamlines": [], "speed_range": [0, 1]}

    all_spd = np.concatenate([s["speeds"] for s in streamlines])
    smin    = float(np.percentile(all_spd, 5))
    smax    = float(np.percentile(all_spd, 95))
    srange  = max(smax - smin, 0.1)

    geh = domain.get("ground_ellipsoid_height", config.GROUND_ELLIPSOID_HEIGHT)
    out = []

    for sl in streamlines:
        pos  = sl["positions"]
        spd  = sl["speeds"]
        flat = []
        for p in pos:
            lon, lat = config.local_to_lonlat(p[0], p[1], domain)
            flat.extend([round(lon, 7), round(lat, 7), round(p[2] + geh, 2)])

        t    = np.clip((spd - smin) / srange, 0, 1)
        rgb  = _turbo(t)
        alpha = np.clip((spd - config.OPACITY_MIN_SPEED) /
                        max(config.OPACITY_MAX_SPEED - config.OPACITY_MIN_SPEED, 1),
                        0.12, 1.0)
        rgba = []
        for i in range(len(spd)):
            rgba.extend([int(rgb[i, 0]*255), int(rgb[i, 1]*255),
                         int(rgb[i, 2]*255), int(alpha[i]*255)])

        out.append({"positions": flat, "colors": rgba, "num_points": len(pos)})

    return {"streamlines": out, "speed_range": [smin, smax]}


def run(occupancy, coords, vel, shape, domain):
    print("\n── Streamline computation ──")

    mean_v   = vel.mean(axis=0)
    wind_deg = math.degrees(math.atan2(mean_v[1], mean_v[0])) % 360
    print(f"    Dominant direction: {wind_deg:.0f}°  "
          f"(mean u={mean_v[0]:.2f} v={mean_v[1]:.2f} m/s)")

    interps    = _build_interps(coords, vel, shape, occupancy, domain)
    sdf_interp = _build_sdf(occupancy, domain)
    seeds      = _make_seeds(occupancy, domain, wind_deg)
    print(f"    {len(seeds)} seed points")

    geh = domain.get("ground_ellipsoid_height", config.GROUND_ELLIPSOID_HEIGHT)
    seed_geo = []
    for s in seeds:
        lon, lat = config.local_to_lonlat(s[0], s[1], domain)
        seed_geo.append([round(lon, 6), round(lat, 6), round(float(s[2]) + geh, 1)])
    config._ensure_dirs()
    seed_path = os.path.join(config.STREAMLINE_DIR, "seed_points.json")
    with open(seed_path, "w") as fh:
        json.dump({"points": seed_geo}, fh)

    results = []
    n_short = n_loop = 0
    for s in seeds:
        if _inside_building(s, occupancy, domain):
            continue
        path, spd = _trace(interps, sdf_interp, s, occupancy, domain)
        if len(path) < 3:
            continue
        diffs  = np.diff(path, axis=0)
        length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        if length < config.STREAMLINE_MIN_LEN:
            n_short += 1
            continue
        disp = float(np.linalg.norm(path[-1] - path[0]))
        if disp / length < 0.12:
            n_loop += 1
            continue
        results.append({"positions": path, "speeds": spd})

    print(f"    {len(results)} streamlines  "
          f"({n_short} too short, {n_loop} loops filtered)")

    data = _to_cesium(results, domain)

    config._ensure_dirs()
    sl_path   = os.path.join(config.STREAMLINE_DIR, "streamlines_combined.json")
    meta_path = os.path.join(config.STREAMLINE_DIR, "metadata.json")

    with open(sl_path, "w") as f:
        json.dump(data, f)

    meta = {
        "type":          "combined",
        "n_streamlines": len(results),
        "wind_deg":      round(wind_deg, 1),
        "domain":        domain,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"    Saved {len(results)} streamlines → {sl_path}")
    return len(results)
