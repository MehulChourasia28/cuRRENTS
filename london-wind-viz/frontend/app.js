/* London Wind Visualisation – CesiumJS frontend
 *
 * Renders Google Photorealistic 3D Tiles via Cesium Ion and overlays
 * pre-computed wind streamlines as per-vertex-coloured polylines with
 * an optional glow layer and animated particle dots.
 *
 * Also supports scanning tile surface heights to build a heightmap that
 * the server can voxelise for perfectly-aligned wind simulation.
 */

let viewer, cfg, tileset;
let currentAngle = 270;
let currentData  = null;

const primaries = [];
let   particles = [];
let   pointCollection = null;

const DIR_NAME = {0:"N",45:"NE",90:"E",135:"SE",
                  180:"S",225:"SW",270:"W",315:"NW"};

// ═══════════════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════════════

async function boot() {
  try {
    const r = await fetch("/api/config");
    cfg = await r.json();
  } catch {
    status("Cannot reach backend – is the server running?");
    return;
  }

  const hasToken = cfg.cesium_token && !cfg.cesium_token.startsWith("YOUR_");

  if (hasToken) {
    Cesium.Ion.defaultAccessToken = cfg.cesium_token;
  }

  viewer = new Cesium.Viewer("cesiumContainer", {
    globe: !hasToken ? undefined : false,
    skyAtmosphere: hasToken ? false : undefined,
    shadows: false,
    animation: false,
    timeline: false,
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    requestRenderMode: false,
  });

  if (hasToken) {
    try {
      tileset = await Cesium.Cesium3DTileset.fromIonAssetId(2275207);
      viewer.scene.primitives.add(tileset);
      status("3D Tiles loaded");
    } catch (e) {
      console.error(e);
      status("3D Tiles failed – check your Cesium token");
    }
  } else {
    status("No Cesium token — using default globe. Set CESIUM_ION_TOKEN for 3D buildings.");
  }

  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(
      cfg.center_lon - 0.001, cfg.center_lat - 0.003, 550),
    orientation: {
      heading: Cesium.Math.toRadians(15),
      pitch:   Cesium.Math.toRadians(-32),
      roll: 0,
    },
    duration: 0,
  });

  wireControls();
  loadStreamlines(currentAngle);
  viewer.scene.preRender.addEventListener(tick);
}

// ═══════════════════════════════════════════════════════════════════
// Controls
// ═══════════════════════════════════════════════════════════════════

function wireControls() {
  const rose = document.getElementById("wind-rose");
  (cfg.available_angles || [0,45,90,135,180,225,270,315]).forEach(a => {
    const b = document.createElement("button");
    b.className = "wbtn" + (a === currentAngle ? " on" : "");
    b.textContent = DIR_NAME[a] || `${a}°`;
    b.onclick = () => {
      rose.querySelectorAll(".wbtn").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      currentAngle = a;
      document.getElementById("angle-label").textContent =
        `${a}° ${DIR_NAME[a]||""}`;
      loadStreamlines(a);
    };
    rose.appendChild(b);
  });

  document.getElementById("sl-opacity").oninput = rebuild;
  document.getElementById("sl-width").oninput   = rebuild;
  document.getElementById("ck-glow").onchange   = rebuild;
  document.getElementById("ck-particles").onchange = rebuild;

  document.getElementById("btn-scan").onclick = scanTileGeometry;
}

// ═══════════════════════════════════════════════════════════════════
// Heightmap scanning — sample Google 3D Tile surface heights
// ═══════════════════════════════════════════════════════════════════

async function scanTileGeometry() {
  const btn = document.getElementById("btn-scan");
  btn.disabled = true;

  if (!tileset) {
    status("No 3D tileset loaded");
    btn.disabled = false;
    return;
  }

  const sampleRes = cfg.heightmap_sample_res || 4.0;
  const halfX = cfg.domain_half_x;
  const halfY = cfg.domain_half_y;
  const nx = Math.floor(2 * halfX / sampleRes);
  const ny = Math.floor(2 * halfY / sampleRes);
  const total = nx * ny;

  const latPerM = 1.0 / 111320.0;
  const lonPerM = 1.0 / (111320.0 * Math.cos(cfg.center_lat * Math.PI / 180));

  status(`Preparing ${total} sample points …`);
  await sleep(50);

  const positions = new Array(total);
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const localX = -halfX + (ix + 0.5) * sampleRes;
      const localY = -halfY + (iy + 0.5) * sampleRes;
      const lon = cfg.center_lon + localX * lonPerM;
      const lat = cfg.center_lat + localY * latPerM;
      positions[iy * nx + ix] = Cesium.Cartographic.fromDegrees(lon, lat);
    }
  }

  const BATCH = 400;
  const heights = new Float64Array(total);
  let sampled = 0;

  for (let i = 0; i < total; i += BATCH) {
    const batch = positions.slice(i, Math.min(i + BATCH, total));
    try {
      const results = await viewer.scene.sampleHeightMostDetailed(batch);
      for (let j = 0; j < results.length; j++) {
        const h = results[j].height;
        heights[i + j] = (h !== undefined && isFinite(h)) ? h : NaN;
      }
    } catch (e) {
      console.warn("sampleHeight batch failed at offset", i, e);
      for (let j = 0; j < batch.length; j++) heights[i + j] = NaN;
    }
    sampled = Math.min(i + BATCH, total);
    status(`Scanning: ${sampled} / ${total}`);
    await sleep(10);
  }

  const validCount = Array.from(heights).filter(h => isFinite(h)).length;
  status(`Scan done (${validCount}/${total} valid). Uploading …`);

  try {
    const resp = await fetch("/api/heightmap", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        heights: Array.from(heights),
        nx, ny,
        resolution: sampleRes,
      }),
    });
    const result = await resp.json();
    if (result.error) {
      status("Server error: " + result.error);
    } else {
      status(`Geometry captured: ${result.coverage}% coverage. Re-run pipeline with --skip-geometry.`);
    }
  } catch (e) {
    console.error(e);
    status("Upload failed");
  }

  btn.disabled = false;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ═══════════════════════════════════════════════════════════════════
// Data loading
// ═══════════════════════════════════════════════════════════════════

async function loadStreamlines(angle) {
  status(`Loading ${angle}° …`);
  try {
    const r = await fetch(`/api/streamlines/${angle}`);
    if (!r.ok) throw new Error(r.status);
    currentData = await r.json();
    rebuild();
    status(`${currentData.streamlines.length} streamlines · ${angle}° wind`);
  } catch (e) {
    console.error(e);
    status("No streamline data – run the pipeline first");
  }
}

// ═══════════════════════════════════════════════════════════════════
// Rendering
// ═══════════════════════════════════════════════════════════════════

function rebuild() {
  clearPrimitives();
  if (!currentData || !currentData.streamlines.length) return;

  const opacity = parseInt(document.getElementById("sl-opacity").value) / 100;
  const width   = parseInt(document.getElementById("sl-width").value);
  const glow    = document.getElementById("ck-glow").checked;
  const showPar = document.getElementById("ck-particles").checked;

  addPolylines(currentData, opacity, width);
  if (glow) addPolylines(currentData, opacity * 0.25, width * 3);
  if (showPar) initParticles(currentData);
}

function addPolylines(data, opacity, width) {
  const insts = [];

  for (const sl of data.streamlines) {
    const n = sl.num_points;
    if (n < 2) continue;

    const pos = new Array(n);
    for (let i = 0; i < n; i++) {
      pos[i] = Cesium.Cartesian3.fromDegrees(
        sl.positions[i*3], sl.positions[i*3+1], sl.positions[i*3+2]);
    }

    const col = new Array(n);
    for (let i = 0; i < n; i++) {
      col[i] = new Cesium.Color(
        sl.colors[i*4]   / 255,
        sl.colors[i*4+1] / 255,
        sl.colors[i*4+2] / 255,
        (sl.colors[i*4+3] / 255) * opacity);
    }

    try {
      insts.push(new Cesium.GeometryInstance({
        geometry: new Cesium.PolylineGeometry({
          positions: pos,
          colors: col,
          width: width,
          colorsPerVertex: true,
          followSurface: false,
        }),
      }));
    } catch (_) {}
  }

  if (!insts.length) return;
  const p = new Cesium.Primitive({
    geometryInstances: insts,
    appearance: new Cesium.PolylineColorAppearance({ translucent: true }),
    asynchronous: true,
  });
  viewer.scene.primitives.add(p);
  primaries.push(p);
}

function clearPrimitives() {
  for (const p of primaries) viewer.scene.primitives.remove(p);
  primaries.length = 0;
  if (pointCollection) {
    viewer.scene.primitives.remove(pointCollection);
    pointCollection = null;
  }
  particles = [];
}

// ═══════════════════════════════════════════════════════════════════
// Animated wind particles
// ═══════════════════════════════════════════════════════════════════

function initParticles(data) {
  if (pointCollection) {
    viewer.scene.primitives.remove(pointCollection);
    pointCollection = null;
  }
  particles = [];

  pointCollection = new Cesium.PointPrimitiveCollection();
  viewer.scene.primitives.add(pointCollection);

  const PARTICLES_PER = 4;
  const MAX_SL = 120;
  const subset = data.streamlines.slice(0, MAX_SL);

  for (let si = 0; si < subset.length; si++) {
    const sl = subset[si];
    if (sl.num_points < 4) continue;
    for (let p = 0; p < PARTICLES_PER; p++) {
      const phase = p / PARTICLES_PER;
      const pt = pointCollection.add({
        position: Cesium.Cartesian3.ZERO,
        pixelSize: 5,
        color: Cesium.Color.WHITE,
      });
      particles.push({ point: pt, sl, phase, speed: 0.08 + Math.random() * 0.04 });
    }
  }
}

let _t = 0;
function tick() {
  if (!particles.length) return;
  _t += 0.016;

  for (const p of particles) {
    const t = (p.phase + _t * p.speed) % 1.0;
    const n = p.sl.num_points;
    const idx = Math.min(Math.floor(t * n), n - 1);
    const i3 = idx * 3;
    const i4 = idx * 4;
    p.point.position = Cesium.Cartesian3.fromDegrees(
      p.sl.positions[i3], p.sl.positions[i3+1], p.sl.positions[i3+2]);
    p.point.color = new Cesium.Color(
      p.sl.colors[i4]   / 255,
      p.sl.colors[i4+1] / 255,
      p.sl.colors[i4+2] / 255,
      1.0);
    p.point.pixelSize = 4 + (p.sl.colors[i4+3] / 255) * 4;
  }
}

// ═══════════════════════════════════════════════════════════════════
function status(msg) { document.getElementById("status").textContent = msg; }
boot();
