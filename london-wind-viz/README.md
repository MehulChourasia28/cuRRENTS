# Urban Wind Visualisation

3D visualisation of wind flowing through city streets, using a GPU Lattice-Boltzmann solver (XLB) and CesiumJS with Google Photorealistic 3D Tiles.

## Pipeline

1. **Set domain** -- enter origin/destination coordinates in the browser; the system computes a 2x-expanded rectangle
2. **Scan geometry** -- sample Google 3D Tile surface heights to build a voxelised building occupancy grid
3. **Fetch wind** -- query OpenWeather at a 5x5 grid across the domain, extrapolate to multiple heights via log-law
4. **Nemotron variations** -- use NVIDIA NIM to generate 15 plausible wind scenario variations, score each with the reward model, keep those above the correctness threshold
5. **LBM solve** -- run the GPU Lattice-Boltzmann solver for each validated variation (~0.5 s per solve on L40S)
6. **Average + visualise** -- average all velocity fields, compute streamlines, render on the 3D tiles

## Quick start

```bash
pip install -r requirements.txt

# Required API keys (in .env or environment)
export CESIUM_ION_TOKEN="..."           # https://cesium.com/ion/tokens
export NIM_API_KEY="nvapi-..."          # https://build.nvidia.com
export OPENWEATHER_API_KEY="..."        # https://openweathermap.org/api

python run_pipeline.py
# Open http://localhost:8080
# Enter origin + dest → Set Domain → Scan Tile Geometry
# Pipeline runs automatically after scan
```

### Flags

| Flag | Effect |
|---|---|
| `--skip-geometry` | Reuse cached heightmap geometry |
| `--skip-nemotron` | Skip Nemotron; use 2 fallback wind angles |
| `--serve-only` | Just start the web server |
| `--lbm-steps N` | Override LBM timestep count |

## Architecture

| Module | Purpose |
|---|---|
| `config.py` | All configuration (dynamic domain, API keys, solver params) |
| `wind_data.py` | OpenWeather multi-point fetch + log-law height profile |
| `nemotron.py` | Nemotron scenario generation + reward-model scoring |
| `lbm.py` | GPU Lattice-Boltzmann solver (XLB/Warp) |
| `geometry.py` | Building geometry from heightmap or OSM |
| `streamlines.py` | RK4 streamline tracer with building-aware deflection |
| `server.py` | Flask server (API + frontend) |
| `run_pipeline.py` | Orchestrates the full pipeline |
| `frontend/` | CesiumJS viewer with streamline rendering |
