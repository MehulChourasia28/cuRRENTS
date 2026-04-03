#!/usr/bin/env python3
"""
Urban Wind Visualisation — pipeline orchestrator.

Usage:
    python run_pipeline.py                  # full pipeline
    python run_pipeline.py --skip-nemotron  # skip Nemotron, use 2 fallback angles
    python run_pipeline.py --serve-only     # web server only
    python run_pipeline.py --lbm-steps N   # override timestep count
"""
import argparse
import asyncio
import logging
import math
import os
import threading

import numpy as np

import config
import server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _run_pipeline(args):
    dom = config.snapshot()
    log.info("Domain: center (%.5f, %.5f)  extent %.0f×%.0f m",
             dom["center_lat"], dom["center_lon"],
             dom["half_x"]*2, dom["half_y"]*2)

    import geometry
    server.set_status("geometry", "Loading geometry …")

    if config.has_cached_geometry():
        log.info("Cache HIT — loading geometry")
        occ, meta = geometry.load_cached_geometry()
        gh = meta.get("ground_ellipsoid_height")
        if gh is not None:
            config.GROUND_ELLIPSOID_HEIGHT = gh
            dom["ground_ellipsoid_height"] = gh
    else:
        log.info("Cache MISS — waiting for heightmap from browser")
        server.set_status("scanning", "Scanning tile geometry …")
        server.wait_for_heightmap()
        occ = np.load(os.path.join(config.DOMAIN_DIR, "occupancy.npy"))
        dom["ground_ellipsoid_height"] = config.GROUND_ELLIPSOID_HEIGHT

    log.info("Occupancy grid %s  %.1f%% covered", occ.shape, occ.mean() * 100)

    server.set_status("wind", "Fetching wind data …")
    from wind_data import fetch_wind_profile, log_wind_profile, WindProfile, PROFILE_HEIGHTS
    wind_ov = dom.get("wind_override")
    if wind_ov:
        speed_10m = float(wind_ov["speed_ms"])
        dir_deg   = float(wind_ov["direction_deg"])
        profile   = WindProfile(direction_deg=dir_deg,
                                center_lat=dom["center_lat"],
                                center_lon=dom["center_lon"],
                                num_samples=0)
        for z in PROFILE_HEIGHTS:
            profile.speed_at_height[z] = round(log_wind_profile(speed_10m, z), 2)
        base_speed = profile.speed_at_height.get(50.0, speed_10m)
        log.info("Wind override: %.2f m/s @ 10 m  dir=%.0f°  → %.1f m/s @ 50 m",
                 speed_10m, dir_deg, base_speed)
    else:
        try:
            profile = asyncio.run(fetch_wind_profile(
                dom["center_lat"], dom["center_lon"],
                dom["half_x"], dom["half_y"]))
            base_speed = profile.speed_at_height.get(50.0, config.LBM_U_LATTICE * 100)
            log.info("Wind: %.1f m/s @ 50 m  dir=%.0f°", base_speed, profile.direction_deg)
        except Exception as exc:
            log.warning("Wind fetch failed (%s) — using 8 m/s fallback", exc)
            base_speed = 8.0
            profile    = None

    scenarios = None
    if not args.skip_nemotron and config.NIM_API_KEY and profile is not None:
        log.info("STEP 3 — Nemotron scenario generation")
        server.set_status("nemotron", "Generating wind scenarios …")
        try:
            from nemotron import generate_variations
            scenarios = asyncio.run(generate_variations(profile))
            log.info("Nemotron: %d scenarios accepted", len(scenarios))
        except Exception as exc:
            log.warning("Nemotron failed (%s) — using fallback", exc)
            scenarios = None

    if not scenarios:
        from dataclasses import dataclass

        @dataclass
        class _Sc:
            u_ref: float
            direction_deg: float

        scenarios = [_Sc(base_speed, 0.0), _Sc(base_speed, 90.0)]
        log.info("Using 2 fallback scenarios (0° and 90°)")

    log.info("STEP 4 — LBM solves (%d scenarios)", len(scenarios))
    import lbm

    lbm_cache: dict = {}
    all_vel:   list = []
    coords_m = shape = None
    n_steps  = args.lbm_steps or config.LBM_N_STEPS

    def _cardinal_solve(angle, speed, label):
        key = (int(round(angle)), round(speed, 1))
        if key in lbm_cache:
            result = lbm_cache[key]
            if result == "failed":
                raise RuntimeError(f"previously failed at {angle}°")
            log.info("    %s — cached", label)
            return result
        log.info("    %s", label)
        try:
            result = lbm.solve_wind(occ, angle, speed, domain=dom, num_steps=n_steps)
            lbm_cache[key] = result
            return result
        except Exception as exc:
            lbm_cache[key] = "failed"
            raise RuntimeError(str(exc)) from exc

    for i, sc in enumerate(scenarios, 1):
        angle = sc.direction_deg
        speed = max(sc.u_ref, 1.0)
        rad   = math.radians(angle)
        cx, cy = abs(math.cos(rad)), abs(math.sin(rad))
        log.info("Scenario %d/%d: %.1f m/s @ %.0f° (cx=%.2f cy=%.2f)",
                 i, len(scenarios), speed, angle, cx, cy)
        server.set_status("lbm", f"Fluid sim {i}/{len(scenarios)} …")

        vel_combined = None
        if cx > 0.05:
            try:
                c0, v0, s0 = _cardinal_solve(0, speed, f"0°  {speed*cx:.1f} m/s")
                sign_x = 1.0 if math.cos(rad) >= 0 else -1.0
                vel_combined = v0 * (cx * sign_x)
                if coords_m is None:
                    coords_m, shape = c0, s0
            except RuntimeError as e:
                log.warning("    0° component failed: %s", e)

        if cy > 0.05:
            try:
                c90, v90, s90 = _cardinal_solve(90, speed, f"90° {speed*cy:.1f} m/s")
                sign_y = 1.0 if math.sin(rad) >= 0 else -1.0
                contrib = v90 * (cy * sign_y)
                vel_combined = contrib if vel_combined is None else vel_combined + contrib
                if coords_m is None:
                    coords_m, shape = c90, s90
            except RuntimeError as e:
                log.warning("    90° component failed: %s", e)

        if vel_combined is not None:
            all_vel.append(vel_combined.astype(np.float32))

    if not all_vel:
        server.set_status("error", "All LBM solves failed")
        log.error("All LBM solves failed — aborting")
        return

    avg_vel = np.mean(all_vel, axis=0).astype(np.float32)
    log.info("Averaged %d velocity fields → shape %s", len(all_vel), avg_vel.shape)

    log.info("STEP 5 — Streamlines")
    server.set_status("streamlines", "Computing streamlines …")
    import streamlines
    n_sl = streamlines.run(occ, coords_m, avg_vel, shape, dom)
    if n_sl == 0:
        log.warning("No streamlines generated")

    log.info("STEP 6 — Route optimisation")
    server.set_status("routing", "Computing drone routes …")

    lon_per_m = 1.0 / (111_320.0 * math.cos(math.radians(dom["center_lat"])))
    lat_per_m = 1.0 / 111_320.0
    ox = (dom["origin_lon"] - dom["center_lon"]) / lon_per_m
    oy = (dom["origin_lat"] - dom["center_lat"]) / lat_per_m
    dx = (dom["dest_lon"]   - dom["center_lon"]) / lon_per_m
    dy = (dom["dest_lat"]   - dom["center_lat"]) / lat_per_m
    origin_local = np.array([ox, oy, dom["origin_height"]])
    dest_local   = np.array([dx, dy, dom["dest_height"]])

    import routing
    try:
        result = routing.compute_routes(
            occ, avg_vel, shape, origin_local, dest_local, dom)
        log.info("Routes done — energy saving %.1f%%  wind reduction %.1f%%",
                 result["energy_savings_pct"], result["wind_reduction_pct"])
    except Exception as exc:
        log.error("Routing failed: %s", exc, exc_info=True)
        server.set_status("error", f"Routing failed: {exc}")
        return

    server.set_status("done",
        f"{len(all_vel)} variations — "
        f"{n_sl} streamlines — "
        f"{result['energy_savings_pct']:.1f}% energy saving")
    log.info("Pipeline complete  http://localhost:%d", config.SERVER_PORT)


def main():
    parser = argparse.ArgumentParser(description="Urban Wind Viz pipeline")
    parser.add_argument("--skip-nemotron", action="store_true")
    parser.add_argument("--serve-only",    action="store_true")
    parser.add_argument("--lbm-steps",    type=int, default=None)
    args = parser.parse_args()

    config._ensure_dirs()

    if args.serve_only:
        server.start()
        return

    t = threading.Thread(target=server.start, daemon=True)
    t.start()

    while True:
        log.info("Waiting for route request  http://localhost:%d", config.SERVER_PORT)
        server.wait_for_coords()

        try:
            _run_pipeline(args)
        except Exception as exc:
            log.error("Pipeline error: %s", exc, exc_info=True)
            server.set_status("error", str(exc))

        server.reset_events()


if __name__ == "__main__":
    main()
