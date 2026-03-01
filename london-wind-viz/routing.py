"""
3-D drone route optimisation through an urban wind field.

Builds a navigation graph from the LBM velocity field, computes two
routes (distance-only via A*, wind-optimised via cuOpt + A* fallback),
and estimates energy consumption for each.

Public API
----------
    compute_routes(occupancy, vel, coords, shape, origin_3d, dest_3d)
        → dict  (ready to serialise as /api/routes JSON)
"""
from __future__ import annotations

import heapq
import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

import config

log = logging.getLogger(__name__)

AIR_DENSITY = 1.225  # kg/m³
G = 9.81


# =====================================================================
# 3-D navigation graph
# =====================================================================

class _NavGraph:
    """Coarse 3-D voxel graph built from the LBM velocity field."""

    _NEIGHBOURS_26 = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    def __init__(self, occupancy: np.ndarray, vel_field: np.ndarray,
                 coords: np.ndarray, shape: tuple):
        res = config.ROUTING_GRID_RES
        occ_res = config.VOXEL_RESOLUTION

        self.nx = max(1, int(2 * config.DOMAIN_HALF_X / res))
        self.ny = max(1, int(2 * config.DOMAIN_HALF_Y / res))
        self.nz = max(1, int(config.DOMAIN_HEIGHT / res))
        self.res = res
        self.total = self.nx * self.ny * self.nz

        occ_ds = zoom(occupancy,
                      (self.nx / occupancy.shape[0],
                       self.ny / occupancy.shape[1],
                       self.nz / occupancy.shape[2]), order=0)
        self.blocked = occ_ds > 0.5

        vel_3c = vel_field.reshape(shape[0], shape[1], shape[2], 3)
        xs_v = np.linspace(-config.DOMAIN_HALF_X, config.DOMAIN_HALF_X, shape[0])
        ys_v = np.linspace(-config.DOMAIN_HALF_Y, config.DOMAIN_HALF_Y, shape[1])
        zs_v = np.linspace(0, config.DOMAIN_HEIGHT, shape[2])
        self._vel_interps = [
            RegularGridInterpolator((xs_v, ys_v, zs_v), vel_3c[..., c],
                                    bounds_error=False, fill_value=0.0)
            for c in range(3)
        ]

        self._xs = np.linspace(-config.DOMAIN_HALF_X, config.DOMAIN_HALF_X, self.nx)
        self._ys = np.linspace(-config.DOMAIN_HALF_Y, config.DOMAIN_HALF_Y, self.ny)
        self._zs = np.linspace(0, config.DOMAIN_HEIGHT, self.nz)

        log.info("  NavGraph %dx%dx%d = %d nodes (%.0f%% blocked)",
                 self.nx, self.ny, self.nz, self.total,
                 self.blocked.sum() / self.blocked.size * 100)

        self._sparse_dist, self._sparse_wind = self._build_sparse()

    def _build_sparse(self):
        """Build sparse adjacency matrices (distance and wind-aware) for the
        full 3-D grid.  Fully vectorised with NumPy for speed."""
        nx, ny, nz = self.nx, self.ny, self.nz

        free_mask = ~self.blocked
        free_ijk = np.argwhere(free_mask)  # (N_free, 3)
        n_free = len(free_ijk)
        if n_free == 0:
            n = self.total
            empty = csr_matrix((n, n))
            return empty, empty

        flat_of = (free_ijk[:, 0] * (ny * nz)
                   + free_ijk[:, 1] * nz
                   + free_ijk[:, 2])

        offsets = np.array(self._NEIGHBOURS_26, dtype=np.int32)  # (26, 3)
        all_rows, all_cols, all_dist, all_wind = [], [], [], []

        pos_arr = np.stack([
            self._xs[free_ijk[:, 0]],
            self._ys[free_ijk[:, 1]],
            self._zs[free_ijk[:, 2]],
        ], axis=1)  # (N_free, 3)

        for off in offsets:
            nb = free_ijk + off[None, :]  # (N_free, 3)
            valid = ((nb[:, 0] >= 0) & (nb[:, 0] < nx) &
                     (nb[:, 1] >= 0) & (nb[:, 1] < ny) &
                     (nb[:, 2] >= 0) & (nb[:, 2] < nz))
            idx_v = np.where(valid)[0]
            if len(idx_v) == 0:
                continue
            nb_v = nb[idx_v]
            nb_free = free_mask[nb_v[:, 0], nb_v[:, 1], nb_v[:, 2]]
            idx_v = idx_v[nb_free]
            if len(idx_v) == 0:
                continue
            nb_v = nb[idx_v]

            src_flat = flat_of[idx_v]
            dst_flat = nb_v[:, 0] * (ny * nz) + nb_v[:, 1] * nz + nb_v[:, 2]

            p0 = pos_arr[idx_v]
            p1 = np.stack([
                self._xs[nb_v[:, 0]],
                self._ys[nb_v[:, 1]],
                self._zs[nb_v[:, 2]],
            ], axis=1)

            diff = p1 - p0
            dist = np.linalg.norm(diff, axis=1)

            mid = 0.5 * (p0 + p1)
            mid_pts = mid.reshape(-1, 1, 3)
            wind = np.zeros_like(mid)
            for c in range(3):
                wind[:, c] = self._vel_interps[c](mid).ravel()

            dn = np.clip(dist, 1e-8, None)
            direction = diff / dn[:, None]
            headwind = -np.sum(wind * direction, axis=1)
            w_speed = np.linalg.norm(wind, axis=1)
            max_ws = max(config.WIND_SPEED, 1.0)

            # Penalise headwinds and high wind speed (turbulence avoidance)
            hw_pen = np.clip(headwind, 0, None) / max_ws
            sp_pen = w_speed / max_ws
            # Give a small discount for tailwinds
            tailwind_benefit = np.clip(-headwind, 0, None) / max_ws * 0.3
            wind_cost = dist * np.clip(
                1.0 + config.WIND_COST_ALPHA * hw_pen
                + config.WIND_COST_BETA * sp_pen
                - tailwind_benefit,
                0.1, None)

            all_rows.append(src_flat)
            all_cols.append(dst_flat)
            all_dist.append(dist)
            all_wind.append(wind_cost)

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        dists = np.concatenate(all_dist).astype(np.float32)
        winds = np.concatenate(all_wind).astype(np.float32)

        n = self.total
        sp_dist = csr_matrix((dists, (rows, cols)), shape=(n, n))
        sp_wind = csr_matrix((winds, (rows, cols)), shape=(n, n))
        log.info("  Sparse graph: %d edges", len(rows))
        return sp_dist, sp_wind

    # -- coordinate helpers ------------------------------------------

    def idx(self, ix, iy, iz) -> int:
        return ix * (self.ny * self.nz) + iy * self.nz + iz

    def ijk(self, flat: int):
        iz = flat % self.nz
        rem = flat // self.nz
        iy = rem % self.ny
        ix = rem // self.ny
        return ix, iy, iz

    def pos(self, ix, iy, iz) -> np.ndarray:
        return np.array([self._xs[ix], self._ys[iy], self._zs[iz]])

    def nearest_ijk(self, local_xyz: np.ndarray):
        ix = int(np.clip(np.searchsorted(self._xs, local_xyz[0]) - 0,
                         0, self.nx - 1))
        iy = int(np.clip(np.searchsorted(self._ys, local_xyz[1]) - 0,
                         0, self.ny - 1))
        iz = int(np.clip(np.searchsorted(self._zs, local_xyz[2]) - 0,
                         0, self.nz - 1))
        return ix, iy, iz

    def wind_at(self, xyz: np.ndarray) -> np.ndarray:
        pt = xyz.reshape(1, -1)
        return np.array([float(f(pt)) for f in self._vel_interps])

    # -- edge costs --------------------------------------------------

    def _edge_distance(self, p0, p1):
        return float(np.linalg.norm(p1 - p0))

    def _edge_wind_cost(self, p0, p1):
        dist = self._edge_distance(p0, p1)
        mid = 0.5 * (p0 + p1)
        w = self.wind_at(mid)
        direction = (p1 - p0)
        dn = np.linalg.norm(direction)
        if dn < 1e-8:
            return dist
        direction /= dn
        headwind = -np.dot(w, direction)
        w_speed = np.linalg.norm(w)
        max_ws = max(config.WIND_SPEED, 1.0)
        hw_penalty = max(0.0, headwind) / max_ws
        sp_penalty = w_speed / max_ws
        return dist * (1.0
                       + config.WIND_COST_ALPHA * hw_penalty
                       + config.WIND_COST_BETA * sp_penalty)


# =====================================================================
# A* pathfinding
# =====================================================================

def _astar(graph: _NavGraph, start_ijk, goal_ijk, use_wind: bool):
    sp = graph._sparse_wind if use_wind else graph._sparse_dist
    si = graph.idx(*start_ijk)
    gi = graph.idx(*goal_ijk)

    open_set = [(0.0, si)]
    g_score = {si: 0.0}
    came_from: dict[int, int] = {}

    goal_pos = graph.pos(*goal_ijk)
    visited = set()

    while open_set:
        f_cur, cur = heapq.heappop(open_set)
        if cur == gi:
            break
        if cur in visited:
            continue
        visited.add(cur)

        row_start = sp.indptr[cur]
        row_end = sp.indptr[cur + 1]
        neighbours = sp.indices[row_start:row_end]
        edge_costs = sp.data[row_start:row_end]

        for k in range(len(neighbours)):
            ni = int(neighbours[k])
            if ni in visited:
                continue
            ec = float(edge_costs[k])
            tentative = g_score[cur] + ec
            if tentative < g_score.get(ni, float("inf")):
                g_score[ni] = tentative
                came_from[ni] = cur
                nijk = graph.ijk(ni)
                nb_pos = graph.pos(*nijk)
                h = float(np.linalg.norm(nb_pos - goal_pos))
                heapq.heappush(open_set, (tentative + h, ni))

    if gi not in came_from and si != gi:
        return None

    path_idx = [gi]
    while path_idx[-1] != si:
        path_idx.append(came_from[path_idx[-1]])
    path_idx.reverse()

    path_3d = []
    for fi in path_idx:
        path_3d.append(graph.pos(*graph.ijk(fi)))
    return np.array(path_3d)


def _smooth_path(pts: np.ndarray, window: int = 5) -> np.ndarray:
    if len(pts) <= window:
        return pts
    smoothed = pts.copy()
    hw = window // 2
    for i in range(hw, len(pts) - hw):
        smoothed[i] = pts[max(0, i - hw):i + hw + 1].mean(axis=0)
    smoothed[0] = pts[0]
    smoothed[-1] = pts[-1]
    return smoothed


def _validate_path(path: np.ndarray, graph: _NavGraph) -> np.ndarray:
    """Ensure the smoothed path stays above buildings.

    Checks each waypoint AND interpolated midpoints between consecutive
    waypoints, pushing z upward when a voxel is occupied.  After lifting
    midpoints a second pass re-checks all waypoints to propagate fixes.
    """
    res = graph.res

    def _lift(pt):
        ix, iy, iz = graph.nearest_ijk(pt)
        if (0 <= ix < graph.nx and 0 <= iy < graph.ny
                and 0 <= iz < graph.nz and graph.blocked[ix, iy, iz]):
            for dz in range(1, 20):
                trial_iz = min(iz + dz, graph.nz - 1)
                if not graph.blocked[ix, iy, trial_iz]:
                    pt[2] = graph._zs[trial_iz]
                    return True
        return False

    for i in range(len(path)):
        _lift(path[i])

    new_pts = [path[0]]
    for i in range(len(path) - 1):
        seg = path[i + 1] - path[i]
        seg_len = float(np.linalg.norm(seg))
        n_sub = max(1, int(seg_len / res))
        for s in range(1, n_sub):
            mid = path[i] + seg * (s / n_sub)
            if _lift(mid):
                new_pts.append(mid.copy())
        new_pts.append(path[i + 1])

    result = np.array(new_pts)
    for i in range(len(result)):
        _lift(result[i])
    return result


def _validate_path_fine(path: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Second-pass validation against the original fine-resolution occupancy.

    Checks each point and its immediate XY neighbours (±1 voxel) to account
    for coordinate rounding in the local-to-geo conversion.  Lifts by an
    extra voxel beyond the first clear one for safety margin.
    """
    vox = config.VOXEL_RESOLUTION
    hx = config.DOMAIN_HALF_X
    hy = config.DOMAIN_HALF_Y
    nxf, nyf, nzf = occupancy.shape

    for i in range(len(path)):
        cx = int((path[i][0] + hx) / vox)
        cy = int((path[i][1] + hy) / vox)
        cz = int(path[i][2] / vox)

        blocked = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                ix = np.clip(cx + dx, 0, nxf - 1)
                iy = np.clip(cy + dy, 0, nyf - 1)
                iz = np.clip(cz, 0, nzf - 1)
                if occupancy[ix, iy, iz]:
                    blocked = True
                    break
            if blocked:
                break

        if blocked:
            for dz in range(1, 30):
                tz = min(cz + dz, nzf - 1)
                clear = True
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        ix = np.clip(cx + dx, 0, nxf - 1)
                        iy = np.clip(cy + dy, 0, nyf - 1)
                        if occupancy[ix, iy, tz]:
                            clear = False
                            break
                    if not clear:
                        break
                if clear:
                    path[i][2] = (tz + 1) * vox
                    break
    return path


# =====================================================================
# cuOpt NIM integration
# =====================================================================

def _select_waypoints(graph: _NavGraph, start_ijk, goal_ijk,
                      n_waypoints: int) -> list[tuple]:
    """Pick candidate waypoints spread across the route corridor."""
    s_pos = graph.pos(*start_ijk)
    g_pos = graph.pos(*goal_ijk)

    corridor_dir = g_pos - s_pos
    corridor_len = np.linalg.norm(corridor_dir)
    if corridor_len < 1e-6:
        return []
    corridor_dir /= corridor_len

    perp_xy = np.array([-corridor_dir[1], corridor_dir[0], 0.0])
    pn = np.linalg.norm(perp_xy)
    if pn > 1e-8:
        perp_xy /= pn

    rng = np.random.RandomState(42)
    candidates = []

    n_along = max(5, int(n_waypoints * 0.6))
    for t in np.linspace(0.05, 0.95, n_along):
        base = s_pos + t * corridor_len * corridor_dir
        lateral = rng.uniform(-0.4, 0.4) * corridor_len * 0.5
        alt_offset = rng.uniform(-20, 30)
        pt = base + lateral * perp_xy
        pt[2] = np.clip(pt[2] + alt_offset, 5.0, config.DOMAIN_HEIGHT - 10)
        ix, iy, iz = graph.nearest_ijk(pt)
        if (0 <= ix < graph.nx and 0 <= iy < graph.ny
                and 0 <= iz < graph.nz and not graph.blocked[ix, iy, iz]):
            candidates.append((ix, iy, iz))

    n_alt = n_waypoints - len(candidates)
    for _ in range(n_alt * 3):
        if len(candidates) >= n_waypoints:
            break
        t = rng.uniform(0.05, 0.95)
        base = s_pos + t * corridor_len * corridor_dir
        lateral = rng.uniform(-0.6, 0.6) * corridor_len * 0.5
        alt = rng.uniform(10, config.DOMAIN_HEIGHT * 0.7)
        pt = base + lateral * perp_xy
        pt[2] = alt
        ix, iy, iz = graph.nearest_ijk(pt)
        if (0 <= ix < graph.nx and 0 <= iy < graph.ny
                and 0 <= iz < graph.nz and not graph.blocked[ix, iy, iz]):
            candidates.append((ix, iy, iz))

    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:n_waypoints]


def _build_cost_matrix(graph: _NavGraph, waypoints_ijk: list[tuple]):
    """Compute all-pairs wind-aware cost between waypoints using scipy
    sparse shortest paths (fast Dijkstra on the pre-built graph)."""
    flat_indices = np.array([graph.idx(*w) for w in waypoints_ijk], dtype=np.int32)
    dist_mat = shortest_path(graph._sparse_wind, method="D",
                             directed=False, indices=flat_indices)
    n = len(waypoints_ijk)
    cost = np.full((n, n), 1e9, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            v = dist_mat[i, flat_indices[j]]
            if np.isfinite(v):
                cost[i, j] = v
    return cost


def _call_cuopt(cost_matrix: np.ndarray, origin_idx: int,
                dest_idx: int) -> Optional[list[int]]:
    """Call cuOpt NIM API. Returns ordered waypoint indices or None."""
    api_key = config.NIM_API_KEY
    if not api_key:
        log.warning("  No NIM_API_KEY – skipping cuOpt")
        return None

    n = cost_matrix.shape[0]
    task_indices = [i for i in range(n) if i != origin_idx]

    task_locations = task_indices
    n_tasks = len(task_indices)
    demand = [[1] * n_tasks]
    prizes = [999999.0 if ti == dest_idx else 0.0
              for ti in task_indices]

    payload = {
        "action": "cuOpt_OptimizedRouting",
        "data": {
            "cost_matrix_data": {
                "data": {
                    "0": cost_matrix.tolist()
                }
            },
            "fleet_data": {
                "vehicle_locations": [[origin_idx, dest_idx]],
                "capacities": [[n_tasks + 1]],
                "vehicle_time_windows": [[0, 999999]],
            },
            "task_data": {
                "task_locations": task_locations,
                "demand": demand,
                "prizes": prizes,
            },
            "solver_config": {
                "time_limit": config.CUOPT_TIME_LIMIT,
            },
        },
    }

    try:
        log.info("  Calling cuOpt NIM API …")
        t0 = time.time()
        resp = requests.post(
            config.CUOPT_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=30,
        )
        elapsed = time.time() - t0
        log.info("  cuOpt responded in %.1fs (status %d)", elapsed, resp.status_code)

        if resp.status_code == 202:
            result = resp.json()
            req_id = result.get("reqId")
            if req_id:
                log.info("  cuOpt queued (reqId=%s), polling …", req_id)
                for _ in range(60):
                    time.sleep(2)
                    poll = requests.get(
                        f"{config.CUOPT_ENDPOINT}/status/{req_id}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=15,
                    )
                    if poll.status_code == 200:
                        result = poll.json()
                        if "response" in result:
                            break
                else:
                    log.warning("  cuOpt polling timed out")
                    return None
        elif resp.status_code == 200:
            result = resp.json()
        else:
            log.warning("  cuOpt error %d: %s", resp.status_code,
                        resp.text[:500])
            return None

        solver_resp = result.get("response", result).get("solver_response", {})
        vehicle_data = solver_resp.get("vehicle_data", {})
        if not vehicle_data:
            log.warning("  cuOpt returned no vehicle data")
            return None

        route = list(vehicle_data.values())[0].get("route", [])
        if not route:
            return None

        log.info("  cuOpt route through %d waypoints (cost %.1f)",
                 len(route), solver_resp.get("solution_cost", -1))
        return route

    except Exception as exc:
        log.warning("  cuOpt call failed: %s", exc)
        return None


def _reconstruct_cuopt_route(graph: _NavGraph, waypoints_ijk: list[tuple],
                              route_indices: list[int]) -> Optional[np.ndarray]:
    """Stitch A* segments between consecutive cuOpt waypoints."""
    full_path = []
    for i in range(len(route_indices) - 1):
        a = waypoints_ijk[route_indices[i]]
        b = waypoints_ijk[route_indices[i + 1]]
        seg = _astar(graph, a, b, use_wind=True)
        if seg is None:
            continue
        if full_path:
            full_path.extend(seg[1:].tolist())
        else:
            full_path.extend(seg.tolist())

    if not full_path:
        return None
    return np.array(full_path)


# =====================================================================
# Energy model
# =====================================================================

def _compute_energy(path: np.ndarray, graph: _NavGraph) -> dict:
    """Estimate drone energy along a 3-D path."""
    m = config.DRONE_MASS
    v_cruise = config.DRONE_CRUISE_SPEED
    A_disc = config.DRONE_DISC_AREA
    A_front = config.DRONE_FRONTAL_AREA
    Cd = config.DRONE_CD
    rho = AIR_DENSITY

    P_hover = (m * G) ** 1.5 / math.sqrt(2 * rho * A_disc)

    total_energy_j = 0.0
    total_dist = 0.0
    total_time = 0.0

    for i in range(len(path) - 1):
        seg = path[i + 1] - path[i]
        dist = float(np.linalg.norm(seg))
        if dist < 0.01:
            continue

        direction = seg / dist
        wind = graph.wind_at(0.5 * (path[i] + path[i + 1]))
        headwind = -np.dot(wind, direction)
        v_air = v_cruise + headwind
        v_air = max(v_air, 1.0)

        dh = float(seg[2])
        v_ground = max(v_cruise, 1.0)
        dt = dist / v_ground

        P_para = 0.5 * rho * Cd * A_front * v_air ** 3
        P_climb = m * G * max(0, dh / dt) if dt > 0 else 0
        P_total = P_hover + P_para + P_climb

        total_energy_j += P_total * dt
        total_dist += dist
        total_time += dt

    energy_wh = total_energy_j / 3600.0

    return {
        "energy_wh": round(energy_wh, 2),
        "distance_m": round(total_dist, 1),
        "time_s": round(total_time, 1),
    }


# =====================================================================
# Local-to-geo conversion
# =====================================================================

def _local_to_lonlat(x, y):
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(config.DOMAIN_CENTER_LAT)))
    return (config.DOMAIN_CENTER_LON + x * lon_per_m,
            config.DOMAIN_CENTER_LAT + y * lat_per_m)


def _wind_exposure(path: np.ndarray, graph: _NavGraph) -> tuple[float, float]:
    """Return (peak_wind_ms, mean_wind_ms) along a path."""
    if path is None or len(path) < 2:
        return 0.0, 0.0
    speeds = []
    for i in range(len(path)):
        w = graph.wind_at(path[i])
        speeds.append(float(np.linalg.norm(w)))
    return max(speeds), sum(speeds) / len(speeds)


def _path_to_geo(path: np.ndarray) -> list[list[float]]:
    geo = []
    for p in path:
        lon, lat = _local_to_lonlat(p[0], p[1])
        alt = p[2] + config.GROUND_ELLIPSOID_HEIGHT
        geo.append([round(lon, 7), round(lat, 7), round(alt, 2)])
    return geo


# =====================================================================
# Public API
# =====================================================================

def compute_routes(occupancy: np.ndarray, vel: np.ndarray,
                   coords: np.ndarray, shape: tuple,
                   origin_local: np.ndarray, dest_local: np.ndarray) -> dict:
    """Compute distance-only and wind-optimised routes.

    Parameters
    ----------
    origin_local, dest_local : np.ndarray shape (3,)
        [x_metres, y_metres, z_metres] in the local domain frame.

    Returns dict ready for JSON serialisation.
    """
    log.info("Building 3-D navigation graph …")
    graph = _NavGraph(occupancy, vel, coords, shape)

    s_ijk = graph.nearest_ijk(origin_local)
    g_ijk = graph.nearest_ijk(dest_local)

    if graph.blocked[s_ijk]:
        for dz in range(1, 20):
            trial = (s_ijk[0], s_ijk[1], min(s_ijk[2] + dz, graph.nz - 1))
            if not graph.blocked[trial]:
                s_ijk = trial
                break
    if graph.blocked[g_ijk]:
        for dz in range(1, 20):
            trial = (g_ijk[0], g_ijk[1], min(g_ijk[2] + dz, graph.nz - 1))
            if not graph.blocked[trial]:
                g_ijk = trial
                break

    # -- Distance-only route (A*) ------------------------------------
    log.info("Computing distance-only route (A*) …")
    dist_path = _astar(graph, s_ijk, g_ijk, use_wind=False)
    if dist_path is not None:
        dist_path = _smooth_path(dist_path)
        dist_path = _validate_path(dist_path, graph)
        dist_path = _validate_path_fine(dist_path, occupancy)
        dist_path[0] = origin_local
        dist_path[-1] = dest_local
        dist_energy = _compute_energy(dist_path, graph)
        dist_geo = _path_to_geo(dist_path)
        log.info("  Distance route: %.0fm, %.1fs, %.2f Wh",
                 dist_energy["distance_m"], dist_energy["time_s"],
                 dist_energy["energy_wh"])
    else:
        log.warning("  A* failed for distance route")
        dist_energy = {"energy_wh": 0, "distance_m": 0, "time_s": 0}
        dist_geo = []

    # -- Wind-optimised route (A* wind + cuOpt, pick best energy) -----
    log.info("Computing wind-optimised route …")
    wind_path = None

    # A* with wind-aware edge costs (always computed)
    log.info("  A* with wind-aware costs …")
    astar_wind_path = _astar(graph, s_ijk, g_ijk, use_wind=True)
    astar_energy = None
    if astar_wind_path is not None:
        ap = _smooth_path(astar_wind_path)
        ap = _validate_path(ap, graph)
        ap = _validate_path_fine(ap, occupancy)
        ap[0] = origin_local
        ap[-1] = dest_local
        astar_energy = _compute_energy(ap, graph)
        log.info("    A* wind route: %.0fm, %.1fs, %.2f Wh",
                 astar_energy["distance_m"], astar_energy["time_s"],
                 astar_energy["energy_wh"])

    # cuOpt waypoint optimisation (optional, compared with A*)
    cuopt_energy = None
    cuopt_wind_path = None
    if config.NIM_API_KEY:
        waypoints_ijk = [s_ijk] + _select_waypoints(
            graph, s_ijk, g_ijk, config.NUM_ROUTE_WAYPOINTS) + [g_ijk]
        origin_wp_idx = 0
        dest_wp_idx = len(waypoints_ijk) - 1

        log.info("  Building %d-waypoint cost matrix …", len(waypoints_ijk))
        cost_mat = _build_cost_matrix(graph, waypoints_ijk)
        cuopt_route = _call_cuopt(cost_mat, origin_wp_idx, dest_wp_idx)
        if cuopt_route is not None:
            cp = _reconstruct_cuopt_route(graph, waypoints_ijk, cuopt_route)
            if cp is not None:
                cp = _smooth_path(cp)
                cp = _validate_path(cp, graph)
                cp = _validate_path_fine(cp, occupancy)
                cp[0] = origin_local
                cp[-1] = dest_local
                cuopt_energy = _compute_energy(cp, graph)
                cuopt_wind_path = cp
                log.info("    cuOpt route: %.0fm, %.1fs, %.2f Wh",
                         cuopt_energy["distance_m"], cuopt_energy["time_s"],
                         cuopt_energy["energy_wh"])

    # Pick whichever candidate uses less energy
    if (cuopt_energy is not None and astar_energy is not None
            and cuopt_energy["energy_wh"] < astar_energy["energy_wh"]):
        log.info("  Using cuOpt route (better energy)")
        wind_path = cuopt_wind_path
        wind_energy = cuopt_energy
        wind_geo = _path_to_geo(wind_path)
    elif astar_wind_path is not None:
        log.info("  Using A* wind route (better energy)")
        wind_path = ap
        wind_energy = astar_energy
        wind_geo = _path_to_geo(wind_path)
    else:
        log.warning("  Wind route computation failed")
        wind_energy = {"energy_wh": 0, "distance_m": 0, "time_s": 0}
        wind_geo = []

    log.info("  Final wind route: %.0fm, %.1fs, %.2f Wh",
             wind_energy["distance_m"], wind_energy["time_s"],
             wind_energy["energy_wh"])

    # -- Energy comparison + wind exposure ----------------------------
    e_dist = dist_energy["energy_wh"]
    e_wind = wind_energy["energy_wh"]
    if e_dist > 0:
        savings = round((e_dist - e_wind) / e_dist * 100, 1)
    else:
        savings = 0.0

    # Wind exposure metrics
    if dist_path is not None:
        max_w_d, mean_w_d = _wind_exposure(dist_path, graph)
    else:
        max_w_d, mean_w_d = 0, 0
    if wind_path is not None and wind_geo:
        max_w_w, mean_w_w = _wind_exposure(wind_path, graph)
    else:
        max_w_w, mean_w_w = 0, 0

    if mean_w_d > 0:
        wind_reduction_pct = round((mean_w_d - mean_w_w) / mean_w_d * 100, 1)
    else:
        wind_reduction_pct = 0.0

    log.info("  Wind exposure — dist route: peak %.1f / mean %.1f m/s, "
             "wind route: peak %.1f / mean %.1f m/s  → %.1f%% reduction",
             max_w_d, mean_w_d, max_w_w, mean_w_w, wind_reduction_pct)

    result = {
        "distance_route": {
            "path": dist_geo,
            **dist_energy,
        },
        "wind_route": {
            "path": wind_geo,
            **wind_energy,
        },
        "energy_savings_pct": savings,
        "max_wind_on_dist_route": round(max_w_d, 1),
        "max_wind_on_wind_route": round(max_w_w, 1),
        "mean_wind_on_dist_route": round(mean_w_d, 1),
        "mean_wind_on_wind_route": round(mean_w_w, 1),
        "wind_reduction_pct": wind_reduction_pct,
    }

    os.makedirs(config.ROUTES_DIR, exist_ok=True)
    with open(os.path.join(config.ROUTES_DIR, "routes.json"), "w") as f:
        json.dump(result, f)
    log.info("  Routes saved. Energy saving: %.1f%%", savings)

    return result
