# London Wind Visualisation

Real-time 3D visualisation of wind flow through London city streets.

- **PINN** (Physics-Informed Neural Network) trained with NVIDIA Modulus methodology solves steady-state Navier–Stokes for arbitrary wind directions
- **CesiumJS** renders Google Photorealistic 3D Tiles as the building backdrop
- **Streamlines** are RK4-integrated from strategic seed points, coloured with the Turbo colourmap and speed-based opacity

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your free Cesium Ion token (https://cesium.com/ion/tokens)
export CESIUM_ION_TOKEN="your_token_here"

# 3. Run the full pipeline
python run_pipeline.py            # fetches buildings, trains PINN, generates streamlines, starts server

# — or use synthetic buildings (no internet needed for geometry) —
python run_pipeline.py --synthetic
```

Open **http://localhost:8080** in your browser.

### Useful flags

| Flag | Effect |
|---|---|
| `--synthetic` | Skip OSM, generate a procedural city block grid |
| `--iterations N` | Override PINN training iterations (default 30 000 GPU / 10 000 CPU) |
| `--skip-geometry` | Reuse cached building data |
| `--skip-training` | Reuse cached PINN checkpoint |
| `--skip-streamlines` | Reuse cached streamline JSON |
| `--serve-only` | Just launch the web server |

---

## How it works

### 1. Building geometry
Building footprints + heights are fetched from OpenStreetMap for a 400 m × 400 m area of the City of London and voxelised into a 3D occupancy grid at 2 m resolution.

### 2. PINN wind simulation
A Fourier-feature MLP with skip connections takes `(x, y, z, cos θ, sin θ)` and predicts `(u, v, w, p)`.  Training minimises the incompressible Navier–Stokes residual together with no-slip (buildings + ground), log-law inlet, and free-stream top boundary losses.  An artificially reduced Reynolds number (Re ≈ 2 000) ensures convergence without hours of training while still capturing channelling, recirculation, and speed-up effects.

### 3. Streamline computation
Virtual massless particles are seeded at the inlet face, near building surfaces, and at street level.  Their paths are traced through the predicted velocity field with adaptive-step RK4.  Colour is mapped via the Turbo colourmap and opacity scales with speed magnitude to hide calm regions.

### 4. Visualisation
CesiumJS overlays the streamlines on top of Google Photorealistic 3D Tiles.  A glow layer (wider, translucent duplicate) adds depth.  Optional animated particles travel along the streamlines for a flowing effect.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- GPU recommended (CUDA) for training; CPU works but is slower
- Free Cesium Ion account for Google 3D Tiles
