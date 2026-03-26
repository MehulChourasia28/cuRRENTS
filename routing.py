"""
Drone route optimisation through a 3-D wind field.

Two routes are computed:
  1. Distance-only  — A* on Euclidean edge costs
  2. Wind-optimised — A* on wind-aware edge costs
         (+ cuOpt waypoint optimisation if NIM_API_KEY is set)

The route with lower computed energy is chosen as the wind route.

Public API
----------
    compute_routes(occupancy, vel, shape, origin_local, dest_local, domain)
        → dict  (written to data/routes/routes.json)
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

_AIR_RHO = 1.225  # kg/m³
_G       = 9.81


# ══════════════════════════════════════════════════════════════════════
# Navigation graph
# ══════════════════════════════════════════════════════════════════════

class _NavGraph:
    _NEIGHBOURS = [(dx, dy, dz)
                   for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
                   if not (dx==0 and dy==0 and dz==0)]

    def __init__(self, occupancy, vel, shape, domain):
        res    = config.ROUTING_GRID_RES
        half_x = domain["half_x"]
        half_y = domain["half_y"]
        height = domain["height"]

        self.nx = max(2, int(2 * half_x / res))
        self.ny = max(2, int(2 * half_y / res))
        self.nz = max(2, int(height / res))
        self.res = res

        # Max-pool occupancy to routing resolution
        self.blocked = self._maxpool(occupancy > 0.5,
                                     self.nx, self.ny, self.nz)

        # Velocity interpolators from LBM field
        # vel shape: (N, 3),  shape: (inx, iny, inz)
        inx, iny, inz = shape
        vel_3d = vel.reshape(inx, iny, inz, 3)
        xs_v = np.linspace(-half_x, half_x, inx)
        ys_v = np.linspace(-half_y, half_y, iny)
        zs_v = np.linspace(0, height, inz)
        self._vel_interps = [
            RegularGridInterpolator((xs_v, ys_v, zs_v), vel_3d[..., c],
                                    bounds_error=False, fill_value=0.0)
            for c in range(3)
        ]
        self._xs = np.linspace(-half_x, half_x, self.nx)
        self._ys = np.linspace(-half_y, half_y, self.ny)
        self._zs = np.linspace(0, height, self.nz)

        log.info("NavGraph %dx%dx%d  %.0f%% blocked",
                 self.nx, self.ny, self.nz,
                 self.blocked.sum()/self.blocked.size*100)

        self._sp_dist, self._sp_wind = self._build_sparse()

    @staticmethod
    def _maxpool(occ_bool, tnx, tny, tnz):
        snx, sny, snz = occ_bool.shape
        out = np.zeros((tnx, tny, tnz), dtype=bool)
        bx, by, bz = snx/tnx, sny/tny, snz/tnz
        for ix in range(tnx):
            sx0, sx1 = int(ix*bx), min(snx, int((ix+1)*bx))
            for iy in range(tny):
                sy0, sy1 = int(iy*by), min(sny, int((iy+1)*by))
                for iz in range(tnz):
                    sz0, sz1 = int(iz*bz), min(snz, int((iz+1)*bz))
                    if occ_bool[sx0:sx1, sy0:sy1, sz0:sz1].any():
                        out[ix, iy, iz] = True
        return out

    def _build_sparse(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        free_mask  = ~self.blocked
        free_ijk   = np.argwhere(free_mask)
        if len(free_ijk) == 0:
            empty = csr_matrix((nx*ny*nz, nx*ny*nz))
            return empty, empty

        flat_of = (free_ijk[:,0]*(ny*nz) + free_ijk[:,1]*nz + free_ijk[:,2])
        pos_arr = np.stack([self._xs[free_ijk[:,0]],
                            self._ys[free_ijk[:,1]],
                            self._zs[free_ijk[:,2]]], axis=1)

        all_r, all_c, all_d, all_w = [], [], [], []
        for off in np.array(self._NEIGHBOURS, dtype=np.int32):
            nb    = free_ijk + off
            valid = ((nb[:,0]>=0)&(nb[:,0]<nx) &
                     (nb[:,1]>=0)&(nb[:,1]<ny) &
                     (nb[:,2]>=0)&(nb[:,2]<nz))
            idx_v = np.where(valid)[0]
            if not len(idx_v): continue
            nb_v  = nb[idx_v]
            nb_fr = free_mask[nb_v[:,0], nb_v[:,1], nb_v[:,2]]
            idx_v = idx_v[nb_fr];  nb_v = nb[idx_v]
            if not len(idx_v): continue

            src = flat_of[idx_v]
            dst = nb_v[:,0]*(ny*nz) + nb_v[:,1]*nz + nb_v[:,2]

            p0  = pos_arr[idx_v]
            p1  = np.stack([self._xs[nb_v[:,0]],
                            self._ys[nb_v[:,1]],
                            self._zs[nb_v[:,2]]], axis=1)
            diff = p1 - p0
            dist = np.linalg.norm(diff, axis=1)

            mid  = 0.5 * (p0 + p1)
            wind = np.zeros_like(mid)
            for c in range(3):
                wind[:,c] = self._vel_interps[c](mid).ravel()

            dn        = np.clip(dist, 1e-8, None)
            direction = diff / dn[:,None]
            headwind  = -np.sum(wind * direction, axis=1)
            w_spd     = np.linalg.norm(wind, axis=1)
            max_ws    = max(float(w_spd.max()), 1.0)

            hw_pen   = np.clip(headwind,  0, None) / max_ws
            sp_pen   = w_spd / max_ws
            tw_bonus = np.clip(-headwind, 0, None) / max_ws * 0.25
            wind_cost = dist * np.clip(
                1.0 + config.WIND_COST_ALPHA * hw_pen
                    + config.WIND_COST_BETA  * sp_pen
                    - tw_bonus, 0.1, None)

            all_r.append(src);  all_c.append(dst)
            all_d.append(dist); all_w.append(wind_cost)

        if not all_r:
            empty = csr_matrix((nx*ny*nz, nx*ny*nz))
            return empty, empty

        R    = np.concatenate(all_r)
        C    = np.concatenate(all_c)
        D    = np.concatenate(all_d).astype(np.float32)
        W    = np.concatenate(all_w).astype(np.float32)
        n    = nx * ny * nz
        log.info("Sparse graph: %d edges", len(R))
        return csr_matrix((D,(R,C)), shape=(n,n)), csr_matrix((W,(R,C)), shape=(n,n))

    def idx(self, ix, iy, iz):
        return ix*(self.ny*self.nz) + iy*self.nz + iz

    def ijk(self, flat):
        iz  = flat % self.nz
        rem = flat // self.nz
        iy  = rem  % self.ny
        ix  = rem  // self.ny
        return ix, iy, iz

    def pos(self, ix, iy, iz):
        return np.array([self._xs[ix], self._ys[iy], self._zs[iz]])

    def nearest_free(self, xyz):
        """Snap xyz to the nearest free (unblocked) cell."""
        ix = int(np.clip(np.searchsorted(self._xs, xyz[0]), 0, self.nx-1))
        iy = int(np.clip(np.searchsorted(self._ys, xyz[1]), 0, self.ny-1))
        iz = int(np.clip(np.searchsorted(self._zs, xyz[2]), 0, self.nz-1))
        if not self.blocked[ix, iy, iz]:
            return ix, iy, iz
        # Search upward first, then spiral outward
        for dz in range(1, self.nz):
            tz = min(iz+dz, self.nz-1)
            if not self.blocked[ix, iy, tz]:
                return ix, iy, tz
        return ix, iy, self.nz-1

    def wind_at(self, xyz):
        pt = xyz.reshape(1,-1)
        return np.array([f(pt).item() for f in self._vel_interps])


# ══════════════════════════════════════════════════════════════════════
# A* pathfinding
# ══════════════════════════════════════════════════════════════════════

def _astar(graph, start_ijk, goal_ijk, use_wind):
    sp = graph._sp_wind if use_wind else graph._sp_dist
    si = graph.idx(*start_ijk)
    gi = graph.idx(*goal_ijk)

    if si == gi:
        return np.array([graph.pos(*start_ijk)])

    g_score   = {si: 0.0}
    came_from = {}
    visited   = set()
    open_set  = [(0.0, si)]
    goal_pos  = graph.pos(*goal_ijk)
    h_scale   = 0.1 if use_wind else 1.0  # admissible even with tailwind

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == gi:
            break
        if cur in visited:
            continue
        visited.add(cur)

        r0, r1 = sp.indptr[cur], sp.indptr[cur+1]
        for k in range(r1-r0):
            ni  = int(sp.indices[r0+k])
            ec  = float(sp.data[r0+k])
            if ni in visited:
                continue
            tg = g_score[cur] + ec
            if tg < g_score.get(ni, 1e18):
                g_score[ni]    = tg
                came_from[ni]  = cur
                h = h_scale * float(np.linalg.norm(graph.pos(*graph.ijk(ni)) - goal_pos))
                heapq.heappush(open_set, (tg+h, ni))

    if gi not in came_from and si != gi:
        return None

    idx_list = [gi]
    while idx_list[-1] != si:
        idx_list.append(came_from[idx_list[-1]])
    idx_list.reverse()

    return np.array([graph.pos(*graph.ijk(fi)) for fi in idx_list])


# ── Path post-processing ──────────────────────────────────────────────

def _smooth(pts, window=7):
    if len(pts) <= window:
        return pts
    hw = window // 2
    out = pts.copy()
    for i in range(hw, len(pts)-hw):
        out[i] = pts[max(0,i-hw):i+hw+1].mean(axis=0)
    out[0] = pts[0];  out[-1] = pts[-1]
    return out


def _lift_blocked(path, graph):
    """Push any waypoint that lands in a blocked cell upward."""
    out = path.copy()
    for i in range(len(out)):
        ix, iy, iz = graph.nearest_free(out[i])
        out[i] = graph.pos(ix, iy, iz)
    out[0]  = path[0]
    out[-1] = path[-1]
    return out


# ── Energy model ──────────────────────────────────────────────────────

def _energy(path, graph):
    m        = config.DRONE_MASS
    v_cruise = config.DRONE_CRUISE_SPEED
    A_disc   = config.DRONE_DISC_AREA
    A_front  = config.DRONE_FRONTAL_AREA
    Cd       = config.DRONE_CD

    P_hover = (m * _G)**1.5 / math.sqrt(2 * _AIR_RHO * A_disc)

    total_e = total_d = total_t = 0.0
    for i in range(len(path)-1):
        seg  = path[i+1] - path[i]
        dist = float(np.linalg.norm(seg))
        if dist < 0.01:
            continue
        direction = seg / dist
        wind      = graph.wind_at(0.5*(path[i]+path[i+1]))
        headwind  = -float(np.dot(wind, direction))
        v_air     = max(v_cruise + headwind, 1.0)
        dh        = float(seg[2])
        dt        = dist / max(v_cruise, 1.0)
        P_para    = 0.5 * _AIR_RHO * Cd * A_front * v_air**3
        P_climb   = m * _G * max(0, dh/dt) if dt > 0 else 0
        total_e  += (P_hover + P_para + P_climb) * dt
        total_d  += dist
        total_t  += dt

    return {
        "energy_wh":  round(total_e / 3600.0, 2),
        "distance_m": round(total_d, 1),
        "time_s":     round(total_t, 1),
    }


def _mean_wind_speed(path, graph):
    if path is None or len(path) < 2:
        return 0.0
    return float(np.mean([np.linalg.norm(graph.wind_at(p)) for p in path]))


# ── cuOpt (optional) ──────────────────────────────────────────────────

def _call_cuopt(cost_matrix, origin_idx, dest_idx):
    if not config.NIM_API_KEY:
        return None
    n = cost_matrix.shape[0]
    task_idx = [i for i in range(n) if i != origin_idx]
    prizes   = [999999.0 if i == dest_idx else 0.0 for i in task_idx]

    payload = {
        "action": "cuOpt_OptimizedRouting",
        "data": {
            "cost_matrix_data": {"data": {"0": cost_matrix.tolist()}},
            "fleet_data": {
                "vehicle_locations": [[origin_idx, dest_idx]],
                "capacities": [[len(task_idx)+1]],
                "vehicle_time_windows": [[0, 999999]],
            },
            "task_data": {
                "task_locations": task_idx,
                "demand": [[1]*len(task_idx)],
                "prizes": prizes,
            },
            "solver_config": {"time_limit": config.CUOPT_TIME_LIMIT},
        },
    }
    try:
        r = requests.post(
            config.CUOPT_ENDPOINT,
            headers={"Authorization": f"Bearer {config.NIM_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload, timeout=30)
        if r.status_code == 200:
            resp = r.json()
        elif r.status_code == 202:
            req_id = r.json().get("reqId")
            resp = None
            for _ in range(30):
                time.sleep(2)
                pr = requests.get(f"{config.CUOPT_ENDPOINT}/status/{req_id}",
                                  headers={"Authorization": f"Bearer {config.NIM_API_KEY}"},
                                  timeout=15)
                if pr.status_code == 200 and "response" in pr.json():
                    resp = pr.json();  break
        else:
            log.warning("cuOpt returned %d", r.status_code)
            return None
        if resp is None:
            return None
        vehicle_data = (resp.get("response", resp)
                           .get("solver_response", {})
                           .get("vehicle_data", {}))
        if not vehicle_data:
            return None
        return list(vehicle_data.values())[0].get("route", None)
    except Exception as exc:
        log.warning("cuOpt error: %s", exc)
        return None


def _cuopt_route(graph, start_ijk, goal_ijk):
    """Build waypoints → call cuOpt → stitch with A* → return path."""
    s_pos = graph.pos(*start_ijk)
    g_pos = graph.pos(*goal_ijk)
    corridor = g_pos - s_pos
    c_len = np.linalg.norm(corridor)
    if c_len < 1:
        return None
    c_dir = corridor / c_len
    perp  = np.array([-c_dir[1], c_dir[0], 0.0])
    pn    = np.linalg.norm(perp)
    if pn > 1e-8:
        perp /= pn

    rng  = np.random.RandomState(7)
    wpts = [start_ijk]
    n    = config.NUM_ROUTE_WAYPOINTS
    for t in np.linspace(0.05, 0.95, max(5, int(n*0.6))):
        base  = s_pos + t * c_len * c_dir
        lat   = rng.uniform(-0.35, 0.35) * c_len * 0.5
        dz    = rng.uniform(-15, 25)
        pt    = base + lat * perp
        pt[2] = np.clip(pt[2]+dz, 5.0, graph._zs[-1]-5)
        ix, iy, iz = graph.nearest_free(pt)
        wpts.append((ix, iy, iz))
    wpts.append(goal_ijk)

    # Deduplicate
    seen = set(); unique = []
    for w in wpts:
        if w not in seen:
            seen.add(w); unique.append(w)
    wpts = unique

    # All-pairs wind cost via Dijkstra
    flat  = np.array([graph.idx(*w) for w in wpts], dtype=np.int32)
    d_mat = shortest_path(graph._sp_wind, method="D",
                          directed=False, indices=flat)
    n_w = len(wpts)
    cost = np.full((n_w, n_w), 1e9)
    for i in range(n_w):
        for j in range(n_w):
            v = d_mat[i, flat[j]]
            if np.isfinite(v):
                cost[i, j] = v

    route_idx = _call_cuopt(cost, 0, n_w-1)
    if route_idx is None:
        return None

    # Stitch A* segments
    full = []
    for i in range(len(route_idx)-1):
        a = wpts[route_idx[i]]
        b = wpts[route_idx[i+1]]
        seg = _astar(graph, a, b, use_wind=True)
        if seg is None:
            continue
        full.extend(seg[1:].tolist() if full else seg.tolist())

    return np.array(full) if full else None


# ── Local → geo ───────────────────────────────────────────────────────

def _to_geo(path, domain):
    geh = domain.get("ground_ellipsoid_height", config.GROUND_ELLIPSOID_HEIGHT)
    out = []
    for p in path:
        lon, lat = config.local_to_lonlat(p[0], p[1], domain)
        out.append([round(lon,7), round(lat,7), round(p[2]+geh, 2)])
    return out


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def compute_routes(occupancy, vel, shape, origin_local, dest_local, domain):
    """Compute and save both routes.  Returns result dict."""
    log.info("Building nav graph …")
    graph  = _NavGraph(occupancy, vel, shape, domain)
    s_ijk  = graph.nearest_free(origin_local)
    g_ijk  = graph.nearest_free(dest_local)

    # ── Distance route ────────────────────────────────────────────
    log.info("A* distance route …")
    dp = _astar(graph, s_ijk, g_ijk, use_wind=False)
    if dp is not None:
        dp = _smooth(_lift_blocked(dp, graph))
        dp[0] = origin_local;  dp[-1] = dest_local
        d_energy = _energy(dp, graph)
        d_geo    = _to_geo(dp, domain)
    else:
        log.warning("Distance A* failed — no path found")
        d_energy = {"energy_wh":0, "distance_m":0, "time_s":0}
        d_geo    = []

    log.info("  Dist route: %.0f m  %.1f Wh",
             d_energy["distance_m"], d_energy["energy_wh"])

    # ── Wind-optimised route ──────────────────────────────────────
    log.info("A* wind-aware route …")
    wp = _astar(graph, s_ijk, g_ijk, use_wind=True)
    if wp is not None:
        wp = _smooth(_lift_blocked(wp, graph))
        wp[0] = origin_local;  wp[-1] = dest_local
        w_energy = _energy(wp, graph)
    else:
        log.warning("Wind A* failed")
        w_energy = None
        wp = None

    # Try cuOpt if API key present
    if config.NIM_API_KEY:
        log.info("cuOpt route …")
        cp = _cuopt_route(graph, s_ijk, g_ijk)
        if cp is not None:
            cp = _smooth(_lift_blocked(cp, graph))
            cp[0] = origin_local;  cp[-1] = dest_local
            c_energy = _energy(cp, graph)
            if w_energy is None or c_energy["energy_wh"] < w_energy["energy_wh"]:
                wp, w_energy = cp, c_energy
                log.info("  Using cuOpt route (lower energy)")

    if wp is None or w_energy is None:
        wp       = dp if dp is not None else np.array([origin_local, dest_local])
        w_energy = _energy(wp, graph) if dp is not None else {"energy_wh":0,"distance_m":0,"time_s":0}

    w_geo = _to_geo(wp, domain)
    log.info("  Wind route: %.0f m  %.1f Wh",
             w_energy["distance_m"], w_energy["energy_wh"])

    # ── Savings ───────────────────────────────────────────────────
    e_d = d_energy["energy_wh"]
    e_w = w_energy["energy_wh"]
    savings = round((e_d - e_w) / e_d * 100, 1) if e_d > 0 else 0.0

    # Mean wind exposure
    mean_w_d = _mean_wind_speed(dp, graph) if dp is not None else 0.0
    mean_w_w = _mean_wind_speed(wp, graph)
    wind_red = round((mean_w_d - mean_w_w) / mean_w_d * 100, 1) if mean_w_d > 0 else 0.0

    log.info("  Energy saving: %.1f%%   Wind exposure reduction: %.1f%%",
             savings, wind_red)

    result = {
        "distance_route": {"path": d_geo, **d_energy},
        "wind_route":     {"path": w_geo, **w_energy},
        "energy_savings_pct":      savings,
        "wind_reduction_pct":      wind_red,
        "mean_wind_dist_route_ms": round(mean_w_d, 2),
        "mean_wind_wind_route_ms": round(mean_w_w, 2),
    }

    config._ensure_dirs()
    with open(os.path.join(config.ROUTES_DIR, "routes.json"), "w") as f:
        json.dump(result, f)

    return result
