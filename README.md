# Urban Wind Visualisation

3D wind-aware drone route optimisation over real city geometry, using a GPU Lattice-Boltzmann CFD solver, NVIDIA cuOpt, and CesiumJS with Google Photorealistic 3D Tiles.

![Demo](ApianXcuRents.gif)

---

## Pipeline

1. **Set domain** — enter origin/destination coordinates; domain is computed as the route bounding box + 100 m padding on each side
2. **Scan geometry** — sample Google 3D Tile surface heights to build a voxelised building occupancy grid (2 m resolution, ceiling-rounded to avoid missed building tops)
3. **Fetch wind** — query OpenWeather at a 5×5 grid across the domain and extrapolate to multiple heights via log-law profile; preset routes use measured station data (EGLC / EGLL) instead
4. **Nemotron variations** — use NVIDIA NIM to generate physically-plausible wind scenario variations, score each with the reward model, keep those above the correctness threshold
5. **LBM solve** — run the GPU Lattice-Boltzmann solver for each validated scenario; two cardinal directions (0° and 90°) are solved and linearly combined for arbitrary wind angles
6. **Streamlines** — batch-vectorised RK4 tracer with SDF-based building avoidance; seeds both upwind faces proportionally to wind component magnitude; results progressively streamed to browser
7. **Route optimisation** — A* on a wind-cost graph (penalises headwind and crosswind, rewards tailwind) followed by cuOpt prize-collecting TSP for global waypoint ordering; centripetal Catmull-Rom post-smoothing removes grid artefacts
8. **Visualise** — CesiumJS viewer with toggleable layers; results load progressively as each pipeline stage completes

## Quick start

```bash
pip install -r requirements.txt

# Required API keys (in .env or environment)
export CESIUM_ION_TOKEN="..."           # https://cesium.com/ion/tokens
export NIM_API_KEY="nvapi-..."          # https://build.nvidia.com
export OPENWEATHER_API_KEY="..."        # https://openweathermap.org/api

python run_pipeline.py
# Open http://localhost:8080
# Pick a preset or enter origin + destination → Analyze Route
```

### Flags

| Flag | Effect |
|---|---|
| `--skip-nemotron` | Skip Nemotron; use 2 fallback wind angles (0° and 90°) |
| `--serve-only` | Start the web server only |
| `--lbm-steps N` | Override LBM timestep count |

## Architecture

| Module | Purpose |
|---|---|
| `config.py` | All configuration — domain, API keys, solver and routing parameters |
| `geometry.py` | Heightmap → voxelised occupancy grid |
| `wind_data.py` | OpenWeather multi-point fetch + log-law height profile |
| `nemotron.py` | Nemotron scenario generation + reward-model scoring |
| `lbm.py` | GPU Lattice-Boltzmann solver (XLB / Warp) |
| `streamlines.py` | Batch-vectorised RK4 streamline tracer with SDF building avoidance |
| `routing.py` | Wind-aware A* + cuOpt route optimiser + centripetal Catmull-Rom smoothing |
| `server.py` | Flask API + progressive pipeline status |
| `run_pipeline.py` | Pipeline orchestrator |
| `frontend/` | CesiumJS viewer — streamlines, routes, occupancy layer, particles |

## Key parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `DOMAIN_HEIGHT` | 150 m | Vertical extent of CFD domain and routing graph |
| `VOXEL_RESOLUTION` | 2 m | Building occupancy grid cell size |
| `ROUTING_GRID_RES` | 3 m | Horizontal nav graph resolution |
| `WIND_COST_ALPHA` | 0.35 | Headwind penalty weight (energy / drag) |
| `WIND_COST_BETA` | 0.30 | Crosswind penalty weight (destabilisation) |
| `LBM_N_STEPS` | 3000 | LBM solver timesteps |

## Preset routes

| Preset | Notes |
|---|---|
| Leadenhall → Heron Tower | City of London |
| Royal Exchange → Gherkin | City of London |
| Liverpool St → Bishopsgate | City of London |
| Tower 42 → The Scalpel | City of London |
| **Guy's → St Thomas's** | Uses EGLC (London City Airport) measured wind data |
| **The Nelson → St George's** | Uses EGLL (Heathrow) measured wind data |

## Visualisation layers

| Layer | Description |
|---|---|
| Photorealistic Textures | Google 3D Tiles |
| Streamlines | Wind flow coloured by speed (turbo colormap) |
| Particles | Animated tracers along streamlines |
| Distance Route | Shortest path ignoring wind (cyan) |
| Wind Route | cuOpt + wind-aware A* optimised path (green) |
| Routing Grid Nodes | Nav graph free/blocked cells |
| Streamline Seeds | RK4 integration seed points |
| **Occupancy Voxels** | Translucent building detection overlay for validation |
