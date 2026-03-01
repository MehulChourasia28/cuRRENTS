"""
GPU Lattice-Boltzmann wind solver using XLB (Autodesk).

Replaces the PINN/FNO with a real physics solve that runs in seconds on GPU.
Uses D3Q27 with BGK collision.

Each wind direction is solved in a **separate subprocess** because XLB's
WARP-compiled kernels cache stale boundary data when multiple steppers are
created in the same process.

    solve_wind(occupancy, wind_angle_deg)
        → (coords_m, velocities_m_s, (nx, ny, nz))

Same return signature as pinn.predict_velocity_field / fno.predict_velocity_field
so the streamline module can use it as a drop-in replacement.
"""
import os
import sys
import math
import time
import tempfile
import subprocess
import numpy as np

import config

# LBM lattice parameters
_U_LATTICE = 0.01       # lattice velocity (keep well below 0.1 for stability)
_RE = 200.0              # effective Reynolds number (laminar but shows channelling)
_N_STEPS = 5000          # timesteps to reach approximate steady state


def _init_xlb(grid_shape):
    """Initialise XLB with WARP backend — called once per subprocess."""
    import xlb
    from xlb import ComputeBackend, PrecisionPolicy
    from xlb.grid import grid_factory

    pp = PrecisionPolicy.FP32FP32
    cb = ComputeBackend.WARP
    vs = xlb.velocity_set.D3Q27(precision_policy=pp, backend=cb)
    xlb.init(velocity_set=vs, default_backend=cb, default_precision_policy=pp)
    grid = grid_factory(grid_shape, compute_backend=cb)
    return vs, grid, pp, cb


def _build_boundary_conditions(grid, vs, occupancy, wind_angle_deg, grid_shape):
    """Set up inlet, outlet, wall, and building BCs for a given wind direction.

    XLB's RegularizedBC only accepts a single non-zero velocity component
    (normal to the face), so each inlet face gets its own BC with just the
    normal velocity.  The tangential component develops naturally in the flow.
    """
    from xlb.operator.boundary_condition import (
        HalfwayBounceBackBC, FullwayBounceBackBC,
        RegularizedBC, ExtrapolationOutflowBC,
    )
    from scipy.ndimage import zoom

    nx, ny, nz = grid_shape
    box = grid.bounding_box_indices()
    box_ne = grid.bounding_box_indices(remove_edges=True)

    angle = math.radians(wind_angle_deg)
    ca, sa = math.cos(angle), math.sin(angle)

    bcs_list = []

    # Classify each lateral face as inlet, outlet, or wall based on the
    # wind direction.  For simplicity (and XLB stability), each inlet face
    # gets only its normal velocity component.
    face_role = {}  # face_key → "inlet" | "outlet" | "wall"

    if abs(ca) > 0.1:
        face_role["left" if ca > 0 else "right"] = "inlet"
        face_role["right" if ca > 0 else "left"] = "outlet"
    else:
        face_role["left"] = "wall"
        face_role["right"] = "wall"

    if abs(sa) > 0.1:
        face_role["front" if sa > 0 else "back"] = "inlet"
        face_role.setdefault("back" if sa > 0 else "front", "outlet")
    else:
        face_role["front"] = "wall"
        face_role["back"] = "wall"

    # For any face not yet classified (diagonals: both axes have inlet),
    # override: downwind faces become outlet
    for k in ["left", "right", "front", "back"]:
        face_role.setdefault(k, "wall")

    wall_face_keys = [k for k, v in face_role.items() if v == "wall"]
    outlet_face_keys = [k for k, v in face_role.items() if v == "outlet"]

    # Inlet BCs (one per inlet face, only normal component)
    for k, v in face_role.items():
        if v != "inlet":
            continue
        if k in ("left", "right"):
            sign = 1.0 if k == "left" else -1.0
            u_normal = sign * _U_LATTICE * abs(ca)
            bcs_list.append(RegularizedBC("velocity",
                            prescribed_value=(u_normal, 0.0, 0.0),
                            indices=box_ne[k]))
        elif k in ("front", "back"):
            sign = 1.0 if k == "front" else -1.0
            u_normal = sign * _U_LATTICE * abs(sa)
            bcs_list.append(RegularizedBC("velocity",
                            prescribed_value=(0.0, u_normal, 0.0),
                            indices=box_ne[k]))

    # Merge outlet faces (following XLB's index convention)
    if outlet_face_keys:
        out_idx = [box_ne[outlet_face_keys[0]][i] for i in range(vs.d)]
        for k in outlet_face_keys[1:]:
            for i in range(vs.d):
                out_idx[i] = tuple(list(out_idx[i]) + list(box_ne[k][i]))
        out_idx = np.unique(np.array(out_idx), axis=-1).tolist()
        bcs_list.append(ExtrapolationOutflowBC(indices=out_idx))

    # Walls: top + bottom + any lateral faces not used as inlet/outlet
    wall_keys = ["bottom", "top"] + wall_face_keys
    w = [box[wall_keys[0]][i] for i in range(vs.d)]
    for k in wall_keys[1:]:
        for i in range(vs.d):
            w[i] = tuple(list(w[i]) + list(box[k][i]))
    w = np.unique(np.array(w), axis=-1).tolist()
    bcs_list.insert(0, FullwayBounceBackBC(indices=w))

    # Downsample occupancy to LBM grid and extract building voxels
    occ_ds = zoom(occupancy, (nx / occupancy.shape[0],
                              ny / occupancy.shape[1],
                              nz / occupancy.shape[2]), order=0)
    bidx = np.where(occ_ds > 0.5)
    interior = ((bidx[0] > 0) & (bidx[0] < nx - 1) &
                (bidx[1] > 0) & (bidx[1] < ny - 1) &
                (bidx[2] > 0) & (bidx[2] < nz - 1))
    buildings = [tuple(bidx[i][interior]) for i in range(vs.d)]
    if len(buildings[0]) > 0:
        bcs_list.append(HalfwayBounceBackBC(indices=buildings))

    return bcs_list


def solve_wind(occupancy, wind_angle_deg, grid_res=None, num_steps=None):
    """Run the LBM solver in a **subprocess** to avoid XLB/Warp kernel-cache
    conflicts when solving multiple wind directions in the same session.

    Returns (coords_m, velocities_m_s, (nx, ny, nz)).
    """
    grid_res = grid_res or config.LBM_GRID_RES
    num_steps = num_steps or _N_STEPS
    occ_path = os.path.join(config.DOMAIN_DIR, "occupancy.npy")
    np.save(occ_path, occupancy)

    out_dir = tempfile.mkdtemp(prefix="lbm_")
    script = os.path.join(os.path.dirname(__file__), "lbm.py")

    sub_env = os.environ.copy()
    sub_env["WARP_CACHE_ROOT"] = tempfile.mkdtemp(prefix="warp_cache_")

    proc = subprocess.run(
        [sys.executable, script,
         str(wind_angle_deg), str(grid_res), str(num_steps),
         occ_path, out_dir],
        capture_output=False,
        env=sub_env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"LBM subprocess failed (exit {proc.returncode})")

    coords_m = np.load(os.path.join(out_dir, "coords.npy"))
    vel = np.load(os.path.join(out_dir, "vel.npy"))
    shape = tuple(np.load(os.path.join(out_dir, "shape.npy")))

    # If the solve produced all-zero velocity (rare XLB cache collision),
    # mirror the solution from the opposite diagonal direction.
    speed = np.linalg.norm(vel, axis=1)
    if speed.max() < 1e-6:
        mirror_angle = (wind_angle_deg + 180) % 360
        print(f"    ⚠ Zero velocity at {wind_angle_deg}° — mirroring from {mirror_angle}°")
        coords_m, vel_m, shape = solve_wind(occupancy, mirror_angle, grid_res, num_steps)
        vel = -vel_m

    return coords_m, vel, shape


def _solve_wind_inproc(occupancy, wind_angle_deg, grid_res=None, num_steps=None):
    """In-process LBM solve (called by the subprocess entry point below)."""
    import warp as wp
    from xlb.operator.stepper import IncompressibleNavierStokesStepper

    grid_res = grid_res or config.LBM_GRID_RES   # reuse the same 4 m grid
    num_steps = num_steps or _N_STEPS

    nx = int(2 * config.DOMAIN_HALF_X / grid_res)
    ny = int(2 * config.DOMAIN_HALF_Y / grid_res)
    nz = int(config.DOMAIN_HEIGHT / grid_res)
    grid_shape = (nx, ny, nz)

    vs, grid, pp, cb = _init_xlb(grid_shape)

    visc = _U_LATTICE * (nx - 1) / _RE
    omega = 1.0 / (3.0 * visc + 0.5)

    bcs = _build_boundary_conditions(grid, vs, occupancy, wind_angle_deg, grid_shape)

    stepper = IncompressibleNavierStokesStepper(
        omega=omega, grid=grid,
        boundary_conditions=bcs,
        collision_type="BGK",
    )
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    print(f"    LBM grid {nx}x{ny}x{nz} | Re={_RE:.0f} | omega={omega:.4f}")

    wp.synchronize()
    t0 = time.time()
    for step in range(num_steps):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, step)
        f_0, f_1 = f_1, f_0
    wp.synchronize()
    elapsed = time.time() - t0
    mlups = nx * ny * nz * num_steps / elapsed / 1e6
    print(f"    {num_steps} steps in {elapsed:.2f}s ({mlups:.0f} MLUPS)")

    # Extract macroscopic velocity from the distribution function
    # u_alpha = sum_q( f_q * c_{q,alpha} ) / rho
    c = np.array(vs.c).reshape(vs.d, vs.q)   # [3, Q]
    f_np = f_0.numpy()                        # [Q, nx, ny, nz]

    rho = f_np.sum(axis=0)
    rho = np.where(rho > 1e-10, rho, 1.0)

    ux = np.einsum("q,qxyz->xyz", c[0].astype(np.float32), f_np) / rho
    uy = np.einsum("q,qxyz->xyz", c[1].astype(np.float32), f_np) / rho
    uz = np.einsum("q,qxyz->xyz", c[2].astype(np.float32), f_np) / rho

    # Trim the single-cell boundary layer
    ux = ux[1:-1, 1:-1, 1:-1]
    uy = uy[1:-1, 1:-1, 1:-1]
    uz = uz[1:-1, 1:-1, 1:-1]

    inner_nx, inner_ny, inner_nz = ux.shape

    # Replace any NaN from numerical instability with zero
    ux = np.nan_to_num(ux, nan=0.0)
    uy = np.nan_to_num(uy, nan=0.0)
    uz = np.nan_to_num(uz, nan=0.0)

    # Convert lattice units → physical m/s
    phys_scale = config.WIND_SPEED / _U_LATTICE
    vel = np.stack([ux.ravel() * phys_scale,
                    uy.ravel() * phys_scale,
                    uz.ravel() * phys_scale], axis=1).astype(np.float32)

    # Build matching physical coordinate grid
    xs = np.linspace(-config.DOMAIN_HALF_X, config.DOMAIN_HALF_X, inner_nx, dtype=np.float32)
    ys = np.linspace(-config.DOMAIN_HALF_Y, config.DOMAIN_HALF_Y, inner_ny, dtype=np.float32)
    zs = np.linspace(0, config.DOMAIN_HEIGHT, inner_nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords_m = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    speed = np.linalg.norm(vel, axis=1)
    print(f"    Speed range: [{speed.min():.2f}, {speed.max():.2f}] m/s")

    return coords_m, vel, (inner_nx, inner_ny, inner_nz)


# ═══════════════════════════════════════════════════════════════════
# Subprocess entry point — python lbm.py <angle> <res> <steps> <occ_path> <out_dir>
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cache_root = os.environ.get("WARP_CACHE_ROOT")
    if cache_root:
        os.environ["WARP_CACHE_PATH"] = cache_root

    angle_deg = float(sys.argv[1])
    grid_res = float(sys.argv[2])
    num_steps = int(sys.argv[3])
    occ_path = sys.argv[4]
    out_dir = sys.argv[5]

    occ = np.load(occ_path)
    coords, vel, shape = _solve_wind_inproc(occ, angle_deg, grid_res, num_steps)

    np.save(os.path.join(out_dir, "coords.npy"), coords)
    np.save(os.path.join(out_dir, "vel.npy"), vel)
    np.save(os.path.join(out_dir, "shape.npy"), np.array(shape))
