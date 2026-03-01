#!/usr/bin/env python3
"""
Urban Wind Visualisation – full pipeline.

  python run_pipeline.py              # interactive: coords → scan → wind → nemotron → LBM → viz
  python run_pipeline.py --skip-geometry   # reuse cached geometry
  python run_pipeline.py --skip-nemotron   # use fallback wind angles instead of Nemotron
  python run_pipeline.py --serve-only      # just start the web server
"""
import os
import sys
import asyncio
import threading
import argparse
import logging
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _ensure_dirs():
    for d in (config.DATA_DIR, config.BUILDINGS_DIR, config.DOMAIN_DIR,
              config.STREAMLINE_DIR, config.ROUTES_DIR):
        os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Urban Wind Viz pipeline")
    parser.add_argument("--skip-geometry", action="store_true",
                        help="Reuse cached geometry (skip heightmap scan)")
    parser.add_argument("--skip-nemotron", action="store_true",
                        help="Skip Nemotron generation; use 2 fallback angles")
    parser.add_argument("--lbm-steps", type=int, default=None)
    parser.add_argument("--serve-only", action="store_true",
                        help="Only start the web server")
    args = parser.parse_args()

    _ensure_dirs()

    if args.serve_only:
        import server
        server.start()
        return

    # Start the server in background (needed for browser interaction)
    import server
    srv = threading.Thread(target=server.start, daemon=True)
    srv.start()

    # ── 1. Geometry (from cache or browser scan) ──────────────────
    import geometry

    if not args.skip_geometry:
        log.info("=" * 60)
        log.info("  STEP 1 · Waiting for coords from browser")
        log.info("=" * 60)
        log.info('  Open http://localhost:%d', config.SERVER_PORT)
        log.info('  Enter origin/dest → Analyze Route')

        server.wait_for_coords()
        log.info("  Coords received: (%.5f, %.5f) ± %.0f×%.0f m",
                 config.DOMAIN_CENTER_LAT, config.DOMAIN_CENTER_LON,
                 config.DOMAIN_HALF_X, config.DOMAIN_HALF_Y)

        if config.has_cached_geometry():
            log.info("  Cache HIT — loading geometry from %s",
                     config.domain_cache_dir())
            server.set_pipeline_status("cache", "Loading cached geometry …")
            _, occupancy, _ = geometry.load_cached_geometry()
        else:
            log.info("  Cache MISS — waiting for heightmap scan from browser")
            server.set_pipeline_status("scanning", "Scanning tile geometry …")
            server.wait_for_heightmap()
            log.info("  Heightmap received and cached")
            occupancy = np.load(os.path.join(config.DOMAIN_DIR, "occupancy.npy"))
    else:
        try:
            data, occupancy, _ = geometry.load_geometry()
            geh = data.get("domain", {}).get("ground_ellipsoid_height")
            if geh is not None:
                config.GROUND_ELLIPSOID_HEIGHT = geh
        except Exception:
            log.error("No cached geometry found. Run without --skip-geometry.")
            return

    # ── 2. Fetch real wind data ─────────────────────────────────
    log.info("=" * 60)
    log.info("  STEP 2 · Fetching real-time wind data")
    log.info("=" * 60)
    server.set_pipeline_status("wind", "Fetching wind data from OpenWeather …")

    from wind_data import fetch_wind_profile
    profile = asyncio.run(fetch_wind_profile(
        config.DOMAIN_CENTER_LAT, config.DOMAIN_CENTER_LON,
        config.DOMAIN_HALF_X, config.DOMAIN_HALF_Y))

    base_speed = profile.speed_at_height.get(50.0, config.WIND_SPEED)
    base_dir = profile.direction_deg

    # ── 3. Nemotron variations (or fallback) ────────────────────
    if not args.skip_nemotron and config.NIM_API_KEY:
        log.info("=" * 60)
        log.info("  STEP 3 · Nemotron scenario generation")
        log.info("=" * 60)
        server.set_pipeline_status("nemotron", "Generating wind variations …")

        from nemotron import generate_variations
        scenarios = asyncio.run(generate_variations(profile))
        if not scenarios:
            log.warning("Nemotron returned 0 scenarios — using fallback angles")
            scenarios = None
    else:
        log.info("  Skipping Nemotron (--skip-nemotron or no NIM_API_KEY)")
        scenarios = None

    if scenarios is None:
        from dataclasses import dataclass

        @dataclass
        class _FB:
            u_ref: float
            direction_deg: float

        # Use cardinal directions (0° and 90°) as fallback — these are
        # single-axis inlets that compile fastest in XLB/Warp.
        scenarios = [
            _FB(u_ref=base_speed, direction_deg=0),
            _FB(u_ref=base_speed, direction_deg=90),
        ]

    # ── 4. LBM solves + averaging ───────────────────────────────
    log.info("=" * 60)
    log.info("  STEP 4 · LBM solves (%d variations)", len(scenarios))
    log.info("=" * 60)

    import lbm

    all_vel = []
    coords_m = shape = None

    for i, sc in enumerate(scenarios, 1):
        angle = sc.direction_deg
        speed = sc.u_ref
        log.info("  Variation %d/%d: %.1f m/s @ %.0f°", i, len(scenarios), speed, angle)
        server.set_pipeline_status("lbm", f"Solving variation {i}/{len(scenarios)} …")

        try:
            c, v, s = lbm.solve_wind(occupancy, angle,
                                     wind_speed_ms=speed,
                                     num_steps=args.lbm_steps)
            all_vel.append(v)
            if coords_m is None:
                coords_m, shape = c, s
        except Exception as exc:
            log.warning("  LBM failed for variation %d: %s", i, exc)

    if not all_vel:
        log.error("All LBM solves failed")
        server.set_pipeline_status("error", "All LBM solves failed")
        srv.join()
        return

    avg_vel = np.mean(all_vel, axis=0).astype(np.float32)
    log.info("  Averaged %d velocity fields → shape %s", len(all_vel), avg_vel.shape)

    # ── 5. Streamlines from averaged field ──────────────────────
    log.info("=" * 60)
    log.info("  STEP 5 · Streamline computation")
    log.info("=" * 60)
    server.set_pipeline_status("streamlines", "Computing streamlines …")

    import streamlines as sl_mod
    sl_mod.run_from_field(occupancy, coords_m, avg_vel, shape)

    # ── 6. Route optimisation ────────────────────────────────────
    log.info("=" * 60)
    log.info("  STEP 6 · Route optimisation (A* + cuOpt)")
    log.info("=" * 60)
    server.set_pipeline_status("routing", "Computing drone routes …")

    import math as _m
    import routing

    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * _m.cos(_m.radians(config.DOMAIN_CENTER_LAT)))
    ox = (config.ORIGIN_LON - config.DOMAIN_CENTER_LON) / lon_per_m
    oy = (config.ORIGIN_LAT - config.DOMAIN_CENTER_LAT) / lat_per_m
    dx = (config.DEST_LON - config.DOMAIN_CENTER_LON) / lon_per_m
    dy = (config.DEST_LAT - config.DOMAIN_CENTER_LAT) / lat_per_m
    origin_local = np.array([ox, oy, config.ORIGIN_HEIGHT])
    dest_local = np.array([dx, dy, config.DEST_HEIGHT])

    try:
        server.set_pipeline_status("cuopt", "Optimising routes with cuOpt …")
        route_result = routing.compute_routes(
            occupancy, avg_vel, coords_m, shape, origin_local, dest_local)
        log.info("  Energy saving: %.1f%%", route_result["energy_savings_pct"])
    except Exception as exc:
        log.error("  Route optimisation failed: %s", exc, exc_info=True)
        server.set_pipeline_status("error", f"Routing failed: {exc}")

    # ── Done ────────────────────────────────────────────────────
    server.set_pipeline_status("done", f"{len(all_vel)} variations averaged")
    log.info("=" * 60)
    log.info("  Pipeline complete — server at http://localhost:%d", config.SERVER_PORT)
    log.info("=" * 60)

    try:
        srv.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
