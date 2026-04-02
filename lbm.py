"""
GPU Lattice-Boltzmann wind solver (D3Q27 / BGK via XLB + Warp).

Each solve runs in a subprocess so that Warp's kernel cache never gets
polluted by a previous solve in the same process.

Public API
----------
    solve_wind(occupancy, wind_angle_deg, wind_speed_ms, domain, num_steps)
        → (coords_m, velocities_m_s, (nx, ny, nz))

    coords_m        float32 (N, 3)   — cell centres in local metres
    velocities_m_s  float32 (N, 3)   — physical m/s [u, v, w]
    shape           (nx, ny, nz)     — interior grid dimensions
"""
import os
import sys
import math
import time
import tempfile
import subprocess
import numpy as np

import config

_N_STEPS   = config.LBM_N_STEPS
_U_LATTICE = config.LBM_U_LATTICE
_RE        = config.LBM_RE


# In-process solve (runs inside subprocess)

def _solve_inproc(occ_path, out_dir, angle_deg, grid_res, num_steps,
                  wind_speed_ms, half_x, half_y, height, u_lat=None):
    import warp as wp
    import xlb
    from xlb import ComputeBackend, PrecisionPolicy
    from xlb.grid import grid_factory
    from xlb.operator.stepper import IncompressibleNavierStokesStepper
    from xlb.operator.boundary_condition import (
        FullwayBounceBackBC, HalfwayBounceBackBC,
        RegularizedBC, ExtrapolationOutflowBC,
    )
    from scipy.ndimage import zoom

    u_lat = u_lat or _U_LATTICE

    occ = np.load(occ_path)
    nx  = max(4, int(2 * half_x / grid_res))
    ny  = max(4, int(2 * half_y / grid_res))
    nz  = max(4, int(height    / grid_res))

    pp = PrecisionPolicy.FP32FP32
    cb = ComputeBackend.WARP
    vs = xlb.velocity_set.D3Q27(precision_policy=pp, compute_backend=cb)
    xlb.init(velocity_set=vs, default_backend=cb, default_precision_policy=pp)
    grid = grid_factory((nx, ny, nz), compute_backend=cb)

    visc  = u_lat * (min(nx, ny) - 1) / _RE
    omega = 1.0 / (3.0 * visc + 0.5)
    omega = min(omega, 1.7)  # cap for stability

    box    = grid.bounding_box_indices()
    box_ne = grid.bounding_box_indices(remove_edges=True)

    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)

    # Determine inlet/outlet faces from wind direction
    # We only ever use cardinal directions (0° or 90°) from run_pipeline,
    # so abs(ca) or abs(sa) will always be ~1.
    bcs = []

    if abs(ca) > 0.1:
        inlet_x  = "left"  if ca > 0 else "right"
        outlet_x = "right" if ca > 0 else "left"
        sign_x   = 1.0 if ca > 0 else -1.0
        bcs.append(RegularizedBC("velocity",
            prescribed_value=(sign_x * u_lat * abs(ca), 0.0, 0.0),
            indices=box_ne[inlet_x]))
        bcs.append(ExtrapolationOutflowBC(indices=box_ne[outlet_x]))

    if abs(sa) > 0.1:
        inlet_y  = "front" if sa > 0 else "back"
        outlet_y = "back"  if sa > 0 else "front"
        sign_y   = 1.0 if sa > 0 else -1.0
        bcs.append(RegularizedBC("velocity",
            prescribed_value=(0.0, sign_y * u_lat * abs(sa), 0.0),
            indices=box_ne[inlet_y]))
        bcs.append(ExtrapolationOutflowBC(indices=box_ne[outlet_y]))

    # Walls (top + bottom + any unused lateral faces)
    used_faces = set()
    for face in ("left","right","front","back"):
        if abs(ca) > 0.1 and face in (inlet_x, outlet_x):
            used_faces.add(face)
        if abs(sa) > 0.1 and face in (inlet_y, outlet_y):
            used_faces.add(face)
    wall_faces = ["bottom", "top"] + [f for f in ("left","right","front","back") if f not in used_faces]
    w = [box[wall_faces[0]][i] for i in range(vs.d)]
    for k in wall_faces[1:]:
        for i in range(vs.d):
            w[i] = tuple(list(w[i]) + list(box[k][i]))
    w = np.unique(np.array(w), axis=-1).tolist()
    bcs.insert(0, FullwayBounceBackBC(indices=w))

    # Buildings
    occ_ds = zoom(occ, (nx/occ.shape[0], ny/occ.shape[1], nz/occ.shape[2]), order=0)
    bidx   = np.where(occ_ds > 0.5)
    interior = ((bidx[0] > 0) & (bidx[0] < nx-1) &
                (bidx[1] > 0) & (bidx[1] < ny-1) &
                (bidx[2] > 0) & (bidx[2] < nz-1))
    bldg = [tuple(bidx[i][interior]) for i in range(vs.d)]
    if len(bldg[0]) > 0:
        bcs.append(HalfwayBounceBackBC(indices=bldg))

    stepper = IncompressibleNavierStokesStepper(
        grid=grid, boundary_conditions=bcs, collision_type="BGK")
    f0, f1, bc_mask, missing_mask = stepper.prepare_fields()

    print(f"  LBM {nx}×{ny}×{nz} | angle={angle_deg}° | u_lat={u_lat:.3f} | omega={omega:.4f}")
    wp.synchronize()
    t0 = time.time()
    for step in range(num_steps):
        f0, f1 = stepper(f0, f1, bc_mask, missing_mask, omega, step)
        f0, f1 = f1, f0
    wp.synchronize()
    dt_s = time.time() - t0
    print(f"  {num_steps} steps in {dt_s:.1f}s  ({nx*ny*nz*num_steps/dt_s/1e6:.0f} MLUPS)")

    c_np  = np.array(vs.c).reshape(vs.d, vs.q)  # (3, Q)
    f_np  = f0.numpy()                            # (Q, nx, ny, nz)
    rho   = f_np.sum(axis=0)
    rho   = np.where(rho > 1e-10, rho, 1.0)

    ux = np.einsum("q,qxyz->xyz", c_np[0].astype(np.float32), f_np) / rho
    uy = np.einsum("q,qxyz->xyz", c_np[1].astype(np.float32), f_np) / rho
    uz = np.einsum("q,qxyz->xyz", c_np[2].astype(np.float32), f_np) / rho

    # Trim boundary layer
    ux, uy, uz = ux[1:-1,1:-1,1:-1], uy[1:-1,1:-1,1:-1], uz[1:-1,1:-1,1:-1]
    ux, uy, uz = (np.nan_to_num(v, 0.0) for v in (ux, uy, uz))

    inx, iny, inz = ux.shape
    phys = wind_speed_ms / u_lat

    vel = np.stack([ux.ravel() * phys,
                    uy.ravel() * phys,
                    uz.ravel() * phys], axis=1).astype(np.float32)

    xs = np.linspace(-half_x, half_x, inx, dtype=np.float32)
    ys = np.linspace(-half_y, half_y, iny, dtype=np.float32)
    zs = np.linspace(0, height, inz,  dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

    spd = np.linalg.norm(vel, axis=1)
    print(f"  Speed range: [{spd.min():.2f}, {spd.max():.2f}] m/s")

    np.save(os.path.join(out_dir, "coords.npy"), coords)
    np.save(os.path.join(out_dir, "vel.npy"),    vel)
    np.save(os.path.join(out_dir, "shape.npy"),  np.array([inx, iny, inz]))


# Public API — spawns subprocess

def solve_wind(occupancy, angle_deg, wind_speed_ms,
               domain=None, grid_res=None, num_steps=None):
    """Run LBM in a subprocess and return (coords_m, vel_m_s, shape).

    Parameters
    ----------
    domain : dict from config.snapshot() — used for half_x/y/height.
             Falls back to live config globals if None.
    """
    grid_res  = grid_res  or config.LBM_GRID_RES
    num_steps = num_steps or _N_STEPS

    if domain is None:
        half_x = config.DOMAIN_HALF_X
        half_y = config.DOMAIN_HALF_Y
        height = config.DOMAIN_HEIGHT
    else:
        half_x = domain["half_x"]
        half_y = domain["half_y"]
        height = domain["height"]

    occ_path = os.path.join(config.DOMAIN_DIR, "occupancy.npy")
    np.save(occ_path, occupancy)

    out_dir   = tempfile.mkdtemp(prefix="lbm_out_")
    cache_env = tempfile.mkdtemp(prefix="warp_cache_")
    script    = os.path.abspath(__file__)
    sub_env   = os.environ.copy()
    sub_env["WARP_CACHE_PATH"] = cache_env

    cmd = [sys.executable, script,
           str(angle_deg), str(grid_res), str(num_steps),
           str(wind_speed_ms), occ_path, out_dir,
           str(half_x), str(half_y), str(height)]

    proc = subprocess.run(cmd, env=sub_env, capture_output=False)
    if proc.returncode != 0:
        raise RuntimeError(f"LBM subprocess failed (exit {proc.returncode})")

    coords = np.load(os.path.join(out_dir, "coords.npy"))
    vel    = np.load(os.path.join(out_dir, "vel.npy"))
    shape  = tuple(np.load(os.path.join(out_dir, "shape.npy")).astype(int))

    # Sanity-check: if max speed is unrealistically high → diverged
    max_spd = float(np.linalg.norm(vel, axis=1).max())
    if max_spd > wind_speed_ms * 10:
        raise RuntimeError(
            f"LBM diverged at {angle_deg}° (max {max_spd:.0f} m/s)")

    # If all-zero, something went wrong
    if max_spd < 1e-4:
        raise RuntimeError(f"LBM returned zero velocity field at {angle_deg}°")

    return coords, vel, shape


# Subprocess entry point

if __name__ == "__main__":
    cache_root = os.environ.get("WARP_CACHE_PATH")
    if cache_root:
        os.environ["WARP_CACHE_PATH"] = cache_root

    angle    = float(sys.argv[1])
    grid_res = float(sys.argv[2])
    n_steps  = int(sys.argv[3])
    spd      = float(sys.argv[4])
    occ_p    = sys.argv[5]
    out_d    = sys.argv[6]
    hx       = float(sys.argv[7])
    hy       = float(sys.argv[8])
    ht       = float(sys.argv[9])

    _solve_inproc(occ_p, out_d, angle, grid_res, n_steps, spd, hx, hy, ht)
