# Urban Wind Visualisation for Drone Delivery

## Product Summary

An AI-powered platform that simulates wind flowing through London's streets on the GPU and computes the most energy-efficient drone delivery route in real time. Built end-to-end on the **NVIDIA** accelerated computing stack.

Enter an origin and destination on a photorealistic 3D map of London. The system captures building geometry, fetches live weather, generates AI-driven wind scenarios, runs GPU fluid dynamics, visualises the airflow, and optimises the drone's flight path — all in a single automated pipeline.

---

## How It Works

### 1. Route Input & Geometry Capture

The user selects an origin and destination on a **CesiumJS** 3D globe rendered with Google Photorealistic Tiles. The system samples thousands of surface-height points from the tileset and voxelises them into a 3D building occupancy grid — the obstacle field for the fluid simulation.

### 2. Real-Time Wind Data

Live wind conditions are fetched from **OpenWeather** across a spatial grid centred on the route. Measurements at 10 m altitude are extrapolated to multiple heights (1.5 m – 50 m) using the logarithmic wind profile law with an urban surface-roughness length of 1.5 m.

### 3. AI Scenario Generation — NVIDIA Nemotron (NIM)

**Llama 3.1 Nemotron Nano 8B**, served through NVIDIA NIM, generates batches of physically plausible urban wind variations — channelling between buildings, Venturi acceleration, vortex shedding, corner effects, and thermal updrafts. Each scenario is then scored by the **Nemotron 70B Reward Model** for physical correctness. Only high-confidence variations (above a configurable threshold) proceed to the solver.

### 4. GPU Fluid Simulation — NVIDIA Warp + XLB

Each validated wind scenario is solved with a full 3D **Lattice Boltzmann Method** (D3Q27 lattice, BGK collision) running on the GPU via **NVIDIA Warp** and **XLB** (Autodesk). Thousands of timesteps converge to approximate steady-state in seconds rather than hours. The velocity fields from all scenarios are averaged into a single composite wind field.

### 5. Streamline Visualisation

Hundreds of streamlines are traced through the averaged wind field using **RK4 integration** with SDF-adaptive step sizing and building-aware collision avoidance. Streamlines are colour-mapped (Turbo colourmap — blue = slow, red = fast) and rendered on the 3D tileset with animated wind particles.

### 6. Route Optimisation — NVIDIA cuOpt (NIM)

Two drone routes are computed:

- **Distance route** — shortest path via A\* on a 3D voxel navigation graph.
- **Wind-optimised route** — A\* with wind-aware edge costs (headwind penalty, tailwind benefit), further refined by **NVIDIA cuOpt**, a GPU-accelerated combinatorial optimisation solver served through NIM. cuOpt reorders candidate waypoints to minimise total energy by exploiting tailwinds and avoiding headwinds.

A physics-based energy model (hover power, parasitic drag, climb cost) compares both routes and reports the percentage of energy saved.

---

## NVIDIA Tech Stack

| Technology | Role |
|---|---|
| **Nemotron Nano 8B** (NIM) | Generates physically plausible urban wind variations |
| **Nemotron 70B Reward** (NIM) | Validates scenario correctness with reward scoring |
| **NVIDIA Warp** | GPU-compiled kernels for the Lattice Boltzmann solver |
| **XLB** (Autodesk, Warp backend) | High-performance LBM framework (D3Q27, BGK) |
| **cuOpt** (NIM) | Combinatorial route optimisation for energy-minimal waypoint ordering |
| **NVIDIA NIM** | Unified inference microservice platform for all three AI services |

### Other Technologies

| Technology | Role |
|---|---|
| CesiumJS + Google 3D Tiles | Photorealistic 3D globe and building geometry source |
| OpenWeather API | Real-time surface wind observations |
| Flask | Backend API server |
| SciPy / NumPy | Streamline tracing, sparse graph pathfinding, interpolation |

---

## Quick Start

```bash
pip install -r requirements.txt

# Required API keys (in .env or environment)
export CESIUM_ION_TOKEN="..."           # https://cesium.com/ion/tokens
export NIM_API_KEY="nvapi-..."          # https://build.nvidia.com
export OPEN_WEATHER_API_KEY="..."       # https://openweathermap.org/api

python run_pipeline.py
# Open http://localhost:8080
# Pick a preset route or enter coordinates → Analyze Route
# Pipeline runs automatically
```

### CLI Flags

| Flag | Effect |
|---|---|
| `--skip-geometry` | Reuse cached heightmap geometry |
| `--skip-nemotron` | Skip Nemotron; use 2 fallback wind angles |
| `--serve-only` | Just start the web server |
| `--lbm-steps N` | Override LBM timestep count |

---

## Architecture

| Module | Purpose |
|---|---|
| `run_pipeline.py` | Orchestrates the full 6-step pipeline |
| `config.py` | All configuration (domain, API keys, solver & drone params) |
| `geometry.py` | Building geometry from heightmap scanning or OSM |
| `wind_data.py` | OpenWeather multi-point fetch + log-law height profile |
| `nemotron.py` | Nemotron scenario generation + reward-model validation |
| `lbm.py` | GPU Lattice Boltzmann solver (XLB / Warp, subprocess per solve) |
| `streamlines.py` | RK4 streamline tracer with SDF-adaptive building avoidance |
| `routing.py` | 3D navigation graph, A\* pathfinding, cuOpt integration, energy model |
| `server.py` | Flask API server + static frontend |
| `frontend/` | CesiumJS viewer, streamline / route rendering, pipeline progress UI |
