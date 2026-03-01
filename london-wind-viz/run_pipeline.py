#!/usr/bin/env python3
"""
London Wind Visualisation – pipeline.

  python run_pipeline.py                    # heightmap scan → LBM → streamlines → serve
  python run_pipeline.py --skip-geometry    # reuse cached geometry
  python run_pipeline.py --synthetic        # synthetic buildings (no internet)
  python run_pipeline.py --serve-only       # just start the web server
"""
import os
import threading
import argparse
import numpy as np

import config


def _ensure_dirs():
    for d in (config.DATA_DIR, config.BUILDINGS_DIR, config.DOMAIN_DIR,
              config.CHECKPOINT_DIR, config.STREAMLINE_DIR):
        os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="London Wind Viz pipeline")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic buildings (no internet)")
    parser.add_argument("--skip-geometry", action="store_true",
                        help="Reuse cached geometry")
    parser.add_argument("--skip-streamlines", action="store_true")
    parser.add_argument("--lbm-steps", type=int, default=None,
                        help="Override LBM timestep count")
    parser.add_argument("--serve-only", action="store_true",
                        help="Only start the web server")
    args = parser.parse_args()

    _ensure_dirs()

    if args.serve_only:
        import server
        server.start()
        return

    # ── 1. Geometry ─────────────────────────────────────────────
    if not args.skip_geometry:
        print("\n" + "=" * 60)
        print("  STEP 1 · Building geometry")
        print("=" * 60)
        print('  Starting server — open http://localhost:{} and click'
              ' "Scan Tile Geometry"'.format(config.SERVER_PORT))

        import server
        srv = threading.Thread(target=server.start, daemon=True)
        srv.start()
        server.wait_for_heightmap()

        print("  Heightmap received — loading voxelised geometry …")
        occupancy = np.load(os.path.join(config.DOMAIN_DIR, "occupancy.npy"))

        if args.synthetic:
            import geometry
            buildings = geometry.create_synthetic_buildings()
            occupancy, _ = geometry.voxelize_buildings(buildings)
            geometry.save_geometry(buildings, occupancy,
                                   np.zeros_like(occupancy))
    else:
        print("  Loading cached geometry …")
        import geometry
        data, occupancy, _ = geometry.load_geometry()
        if "ground_ellipsoid_height" in data.get("domain", {}):
            config.GROUND_ELLIPSOID_HEIGHT = data["domain"]["ground_ellipsoid_height"]

    # ── 2. LBM wind solve + streamlines ─────────────────────────
    if not args.skip_streamlines:
        print("\n" + "=" * 60)
        print("  STEP 2 · LBM wind solve + streamlines")
        print("=" * 60)
        import lbm
        import streamlines as sl_mod

        def _predict(deg):
            return lbm.solve_wind(occupancy, deg, num_steps=args.lbm_steps)

        sl_mod.run(occupancy, predict_fn=_predict)

    # ── 3. Web server ───────────────────────────────────────────
    if not args.skip_geometry:
        print("\n" + "=" * 60)
        print("  Pipeline complete — server already running at "
              f"http://localhost:{config.SERVER_PORT}")
        print("=" * 60)
        print("  Press Ctrl+C to exit.\n")
        try:
            srv.join()
        except KeyboardInterrupt:
            pass
    else:
        print("\n" + "=" * 60)
        print("  STEP 3 · Visualisation server")
        print("=" * 60)
        import server
        server.start()


if __name__ == "__main__":
    main()
