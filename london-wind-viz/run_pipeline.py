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


def _run_pipeline(args, server, geometry_mod):
    """Execute the full pipeline once for the current config values."""
    import math as _m

    # ── 1. Geometry (from cache or browser scan) ──────────────────
    if config.has_cached_geometry():
        log.info("  Cache HIT — loading geometry from %s",
                 config.domain_cache_dir())
        server.set_pipeline_status("cache", "Loading cached geometry …")
        _, occupancy, _ = geometry_mod.load_cached_geometry()
    else:
        log.info("  Cache MISS — waiting for heightmap scan from browser")
        server.set_pipeline_status("scanning", "Scanning tile geometry …")
        server.wait_for_heightmap()
        log.info("  Heightmap received and cached")
        occupancy = np.load(os.path.join(config.DOMAIN_DIR, "occupancy.npy"))

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

    # ── 3. Nemotron variations (or fallback) ────────────────────
    scenarios = None
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

    if scenarios is None:
        from dataclasses import dataclass

        @dataclass
        class _FB:
            u_ref: float
            direction_deg: float

        scenarios = [
            _FB(u_ref=base_speed, direction_deg=0),
            _FB(u_ref=base_speed, direction_deg=90),
        ]

    # ── 4. LBM solves + averaging ───────────────────────────────
    # Decompose each Nemotron scenario into two cardinal-direction solves
    # (0° and 90°) weighted by cos/sin.  XLB's RegularizedBC is stable for
    # single-face inlets but diverges for diagonal angles, so this gives
    # physically sound results via linear superposition while always stable.
    log.info("=" * 60)
    log.info("  STEP 4 · LBM solves (%d variations × 2 cardinal dirs)", len(scenarios))
    log.info("=" * 60)

    import lbm, math

    all_vel = []
    coords_m = shape = None
    lbm_cache = {}

    _DIVERGED = object()

    def _solve_cardinal(cardinal_angle, spd, label):
        """Solve a cardinal direction, caching results (including failures)."""
        key = (cardinal_angle, round(spd, 2))
        if key in lbm_cache:
            cached = lbm_cache[key]
            if cached is _DIVERGED:
                raise ValueError(f"previously diverged at {cardinal_angle}°")
            log.info("    %s — cached", label)
            return cached
        log.info("    %s", label)
        c, v, s = lbm.solve_wind(occupancy, cardinal_angle,
                                 wind_speed_ms=spd,
                                 num_steps=args.lbm_steps)
        max_spd = float(np.linalg.norm(v, axis=1).max())
        if max_spd > spd * 8:
            lbm_cache[key] = _DIVERGED
            raise ValueError(f"diverged (max {max_spd:.0f} m/s vs {spd:.1f} input)")
        lbm_cache[key] = (c, v, s)
        return c, v, s

    for i, sc in enumerate(scenarios, 1):
        angle = sc.direction_deg
        speed = sc.u_ref
        rad = math.radians(angle)
        cx, cy = abs(math.cos(rad)), abs(math.sin(rad))
        log.info("  Variation %d/%d: %.1f m/s @ %.0f° → decomposed (cx=%.2f, cy=%.2f)",
                 i, len(scenarios), speed, angle, cx, cy)
        server.set_pipeline_status("lbm", f"Solving variation {i}/{len(scenarios)} …")

        vel_combined = None
        if cx > 0.1:
            try:
                c0, v0, s0 = _solve_cardinal(0, speed, f"0° component ({speed*cx:.1f} m/s)")
                sign_x = 1.0 if math.cos(rad) > 0 else -1.0
                vel_combined = v0 * (cx * sign_x)
                if coords_m is None:
                    coords_m, shape = c0, s0
            except (ValueError, RuntimeError) as exc:
                log.warning("    0° component failed: %s — using 90° only", exc)

        if cy > 0.1:
            try:
                c90, v90, s90 = _solve_cardinal(90, speed, f"90° component ({speed*cy:.1f} m/s)")
                sign_y = 1.0 if math.sin(rad) > 0 else -1.0
                contrib = v90 * (cy * sign_y)
                vel_combined = contrib if vel_combined is None else vel_combined + contrib
                if coords_m is None:
                    coords_m, shape = c90, s90
            except (ValueError, RuntimeError) as exc:
                log.warning("    90° component failed: %s — using 0° only", exc)

        if vel_combined is not None:
            all_vel.append(vel_combined.astype(np.float32))

    if not all_vel:
        log.error("All LBM solves failed")
        server.set_pipeline_status("error", "All LBM solves failed")
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

    import server
    import geometry

    srv = threading.Thread(target=server.start, daemon=True)
    srv.start()

    while True:
        log.info("=" * 60)
        log.info("  Waiting for route request from browser …")
        log.info("=" * 60)
        log.info('  http://localhost:%d', config.SERVER_PORT)

        server.wait_for_coords()
        log.info("  Coords received: (%.5f, %.5f) ± %.0f×%.0f m",
                 config.DOMAIN_CENTER_LAT, config.DOMAIN_CENTER_LON,
                 config.DOMAIN_HALF_X, config.DOMAIN_HALF_Y)

        try:
            _run_pipeline(args, server, geometry)
        except Exception as exc:
            log.error("  Pipeline failed: %s", exc, exc_info=True)
            server.set_pipeline_status("error", str(exc))

        server.reset_pipeline_events()


if __name__ == "__main__":
    main()
