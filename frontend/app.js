/* Urban Wind Viz — frontend */

let viewer, cfg, tileset;
let currentData = null;
let routeData   = null;
const primaries      = [];
const routePrims     = [];
const routeEntities  = [];
let particles        = [];
let pointCollection  = null;

// ── Presets ───────────────────────────────────────────────────────────

const PRESETS = [
  { name: "Custom",                     origin: null },
  { name: "Leadenhall → Heron Tower",
    origin: "51.5130, -0.0830", dest: "51.5158, -0.0805", oh: 100, dh: 110 },
  { name: "Royal Exchange → Gherkin",
    origin: "51.5128, -0.0850", dest: "51.5150, -0.0800", oh: 90,  dh: 100 },
  { name: "Liverpool St → Bishopsgate",
    origin: "51.5180, -0.0835", dest: "51.5145, -0.0810", oh: 100, dh: 110 },
  { name: "Tower 42 → The Scalpel",
    origin: "51.5160, -0.0820", dest: "51.5130, -0.0815", oh: 95,  dh: 100 },
];

// Pipeline stage → stepper index
const STAGE_STEP = {
  scanning: 0, cache: 0, geometry: 0,
  wind: 1, nemotron: 2, lbm: 3, streamlines: 4, routing: 5, cuopt: 5,
};

// ── Bootstrap ─────────────────────────────────────────────────────────

async function boot() {
  try {
    const r = await fetch("/api/config");
    cfg = await r.json();
  } catch {
    setStatus("Cannot reach backend — is run_pipeline.py running?");
    return;
  }

  const hasToken = cfg.cesium_token && !cfg.cesium_token.startsWith("YOUR_");
  if (hasToken) Cesium.Ion.defaultAccessToken = cfg.cesium_token;

  viewer = new Cesium.Viewer("cesiumContainer", {
    globe:                  !hasToken ? undefined : false,
    skyAtmosphere:          hasToken  ? false : undefined,
    shadows:                false,
    animation:              false,
    timeline:               false,
    baseLayerPicker:        false,
    geocoder:               false,
    homeButton:             false,
    sceneModePicker:        false,
    navigationHelpButton:   false,
    requestRenderMode:      false,
  });

  if (hasToken) {
    try {
      tileset = await Cesium.Cesium3DTileset.fromIonAssetId(2275207);
      viewer.scene.primitives.add(tileset);
    } catch (e) { console.warn("3D tileset not loaded:", e); }
  }

  flyTo(cfg.center_lat, cfg.center_lon);
  wireControls();
  document.getElementById("ck-particles").checked = true;
  viewer.scene.preRender.addEventListener(tick);

  // Attempt to load any previously-computed results on startup
  loadStreamlines();
  loadRoutes();
}

function flyTo(lat, lon) {
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon - 0.001, lat - 0.003, 580),
    orientation: { heading: Cesium.Math.toRadians(15),
                   pitch:   Cesium.Math.toRadians(-32), roll: 0 },
    duration: 1.5,
  });
}

// ── Controls ──────────────────────────────────────────────────────────

function wireControls() {
  document.getElementById("btn-analyze").onclick  = analyzeRoute;
  document.getElementById("btn-scan").onclick     = scanTileGeometry;
  document.getElementById("sl-opacity").oninput   = rebuildStreamlines;
  document.getElementById("sl-width").oninput     = rebuildStreamlines;
  document.getElementById("ck-glow").onchange     = rebuildStreamlines;
  document.getElementById("ck-particles").onchange= rebuildStreamlines;
  document.getElementById("preset-route").onchange= applyPreset;
}

function applyPreset() {
  const idx    = document.getElementById("preset-route").selectedIndex;
  const fields = document.getElementById("coord-inputs-section");
  if (idx === 0) { fields && fields.classList.remove("hidden"); return; }
  const p = PRESETS[idx];
  if (!p || !p.origin) return;
  document.getElementById("inp-origin").value   = p.origin;
  document.getElementById("inp-dest").value     = p.dest;
  document.getElementById("inp-origin-h").value = String(p.oh ?? 30);
  document.getElementById("inp-dest-h").value   = String(p.dh ?? 30);
  fields && fields.classList.add("hidden");
}

// ── Pipeline stepper UI ───────────────────────────────────────────────

function showProgress() {
  const div = document.getElementById("pipeline-progress");
  if (!div) return;
  div.classList.remove("hidden");
  div.querySelectorAll(".step").forEach(s => s.classList.remove("done","active"));
}

function setStep(stageKey) {
  const idx = STAGE_STEP[stageKey] ?? -1;
  const div = document.getElementById("pipeline-progress");
  if (!div) return;
  div.querySelectorAll(".step").forEach((s, i) => {
    s.classList.remove("done","active");
    if (i < idx)      s.classList.add("done");
    else if (i === idx) s.classList.add("active");
  });
}

function finishProgress() {
  const div = document.getElementById("pipeline-progress");
  if (!div) return;
  div.querySelectorAll(".step").forEach(s => {
    s.classList.remove("active"); s.classList.add("done");
  });
  setTimeout(() => div.classList.add("hidden"), 1200);
}

// ── Analyze Route ─────────────────────────────────────────────────────

async function analyzeRoute() {
  const parse = id => {
    const parts = document.getElementById(id).value.split(",").map(s => parseFloat(s.trim()));
    return { lat: parts[0], lon: parts[1] };
  };
  const origin  = parse("inp-origin");
  const dest    = parse("inp-dest");
  const originH = parseFloat(document.getElementById("inp-origin-h").value) || 30;
  const destH   = parseFloat(document.getElementById("inp-dest-h").value)   || 30;

  if ([origin.lat, origin.lon, dest.lat, dest.lon].some(isNaN)) {
    setStatus("Invalid coordinates");
    return;
  }

  clearAll();
  setStatus("Setting domain …");
  document.getElementById("btn-analyze").disabled = true;
  showProgress();

  try {
    const r = await fetch("/api/coords", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ origin, dest, origin_height: originH, dest_height: destH }),
    });
    const result = await r.json();
    if (result.error) { setStatus("Error: " + result.error); return; }

    cfg.center_lat    = result.center_lat;
    cfg.center_lon    = result.center_lon;
    cfg.domain_half_x = result.half_x;
    cfg.domain_half_y = result.half_y;

    flyTo(result.center_lat, result.center_lon);
    showMarkers(origin, dest, originH, destH);

    if (result.cached) {
      setStatus("Cached geometry found — pipeline starting …");
      pollPipeline();
    } else if (tileset) {
      setStatus("Scanning tile geometry …");
      await scanTileGeometry();
    } else {
      setStatus("No tileset available. Click Scan Tile Geometry manually.");
      document.getElementById("btn-scan").classList.remove("hidden");
      document.getElementById("btn-scan").disabled = false;
    }
  } catch (e) {
    console.error(e);
    setStatus("Failed to set domain");
  } finally {
    document.getElementById("btn-analyze").disabled = false;
  }
}

function showMarkers(origin, dest, oh, dh) {
  const geh = (cfg && cfg.ground_ellipsoid_height) || 58;
  addMarker([origin.lon, origin.lat, geh + oh], "A", Cesium.Color.fromCssColorString("#00d2ff"));
  addMarker([dest.lon,   dest.lat,   geh + dh], "B", Cesium.Color.fromCssColorString("#76B900"));
}

// ── Heightmap scan ────────────────────────────────────────────────────

async function scanTileGeometry() {
  const btn = document.getElementById("btn-scan");
  btn.disabled = true;
  if (!tileset) { setStatus("No 3D tileset loaded"); btn.disabled = false; return; }

  const sRes  = cfg.heightmap_sample_res || 4.0;
  const hx    = cfg.domain_half_x;
  const hy    = cfg.domain_half_y;
  const nx    = Math.floor(2 * hx / sRes);
  const ny    = Math.floor(2 * hy / sRes);
  const total = nx * ny;

  const latM = 1.0 / 111320.0;
  const lonM = 1.0 / (111320.0 * Math.cos(cfg.center_lat * Math.PI / 180));

  setStatus(`Preparing ${total} sample points …`);
  await sleep(50);

  const positions = new Array(total);
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const lx = -hx + (ix + 0.5) * sRes;
      const ly = -hy + (iy + 0.5) * sRes;
      positions[iy * nx + ix] = Cesium.Cartographic.fromDegrees(
        cfg.center_lon + lx * lonM,
        cfg.center_lat + ly * latM);
    }
  }

  const BATCH   = 500;
  const heights = new Float64Array(total);
  for (let i = 0; i < total; i += BATCH) {
    const batch = positions.slice(i, Math.min(i + BATCH, total));
    try {
      const res = await viewer.scene.sampleHeightMostDetailed(batch);
      for (let j = 0; j < res.length; j++) {
        const h = res[j].height;
        heights[i + j] = (h !== undefined && isFinite(h)) ? h : NaN;
      }
    } catch { for (let j = 0; j < batch.length; j++) heights[i + j] = NaN; }
    setStatus(`Scanning: ${Math.min(i + BATCH, total)} / ${total}`);
    await sleep(10);
  }

  setStatus("Uploading heightmap …");
  try {
    const resp = await fetch("/api/heightmap", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ heights: Array.from(heights), nx, ny, resolution: sRes }),
    });
    const result = await resp.json();
    if (result.error) setStatus("Server error: " + result.error);
    else setStatus(`Geometry captured (${result.coverage}% buildings) — pipeline running …`);
  } catch (e) { setStatus("Upload failed: " + e); }

  btn.disabled = false;
  pollPipeline();
}

// ── Pipeline polling ──────────────────────────────────────────────────

async function pollPipeline() {
  const stageLabels = {
    scanning:    "Scanning tile geometry",
    cache:       "Loading cached geometry",
    geometry:    "Loading geometry",
    wind:        "Fetching wind data",
    nemotron:    "Nemotron scenario generation",
    lbm:         "GPU fluid simulation",
    streamlines: "Computing streamlines",
    routing:     "Computing drone routes",
    cuopt:       "cuOpt route optimisation",
  };

  for (let i = 0; i < 720; i++) {          // max 24 min
    await sleep(2000);
    try {
      const r = await fetch("/api/pipeline-status");
      const s = await r.json();
      const label = stageLabels[s.stage] || s.stage;
      setStatus(`${label} — ${s.detail}`);
      setStep(s.stage);

      if (s.stage === "done") {
        finishProgress();
        // Load results — retry a couple of times if files aren't flushed yet
        for (let retry = 0; retry < 3; retry++) {
          await sleep(500);
          const ok = await loadStreamlines();
          if (ok) break;
        }
        for (let retry = 0; retry < 3; retry++) {
          await sleep(500);
          const ok = await loadRoutes();
          if (ok) break;
        }
        return;
      }

      if (s.stage === "error") {
        finishProgress();
        setStatus("Pipeline error: " + s.detail);
        return;
      }
    } catch { /* network blip — keep polling */ }
  }
  setStatus("Pipeline timed out — check server logs");
}

// ── Streamline loading ────────────────────────────────────────────────

async function loadStreamlines() {
  try {
    const r = await fetch("/api/streamlines/combined");
    if (!r.ok) return false;
    currentData = await r.json();
    if (!currentData.streamlines || currentData.streamlines.length === 0) {
      console.warn("Streamlines response has no data");
      return false;
    }
    rebuildStreamlines();
    return true;
  } catch (e) {
    console.warn("loadStreamlines:", e);
    return false;
  }
}

// ── Route loading ─────────────────────────────────────────────────────

async function loadRoutes() {
  try {
    const r = await fetch("/api/routes");
    if (!r.ok) return false;
    routeData = await r.json();
    renderRoutes();
    updateRouteUI();
    return true;
  } catch (e) {
    console.warn("loadRoutes:", e);
    return false;
  }
}

function renderRoutes() {
  clearRoutes();
  if (!routeData) return;
  const dr = routeData.distance_route;
  const wr = routeData.wind_route;
  if (dr && dr.path && dr.path.length > 1)
    addRoutePolyline(dr.path, new Cesium.Color(0, 0.82, 1.0, 0.85), 3, false);
  if (wr && wr.path && wr.path.length > 1)
    addRoutePolyline(wr.path, new Cesium.Color(0.46, 0.73, 0, 1.0),  5, true);
  if (dr && dr.path && dr.path.length > 0) {
    addMarker(dr.path[0],                  "A", Cesium.Color.fromCssColorString("#00d2ff"));
    addMarker(dr.path[dr.path.length - 1], "B", Cesium.Color.fromCssColorString("#76B900"));
  }
}

function addRoutePolyline(path, color, width, glow) {
  const positions = path.map(p => Cesium.Cartesian3.fromDegrees(p[0], p[1], p[2]));
  const n = positions.length;
  const mkPrim = (lineWidth, col) => {
    try {
      const prim = new Cesium.Primitive({
        geometryInstances: [new Cesium.GeometryInstance({
          geometry: new Cesium.PolylineGeometry({
            positions, colors: new Array(n).fill(col),
            width: lineWidth, colorsPerVertex: true, followSurface: false,
          }),
        })],
        appearance: new Cesium.PolylineColorAppearance({ translucent: true }),
        asynchronous: true,
      });
      viewer.scene.primitives.add(prim);
      routePrims.push(prim);
    } catch (e) { console.error("polyline error", e); }
  };
  mkPrim(width, color);
  if (glow) mkPrim(width * 4, new Cesium.Color(color.red, color.green, color.blue, 0.18));
}

function addMarker(coords, label, color) {
  const e = viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(coords[0], coords[1], coords[2]),
    point: { pixelSize: 14, color, outlineColor: Cesium.Color.WHITE, outlineWidth: 2 },
    label: {
      text: label, font: "bold 16px sans-serif",
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      outlineWidth: 3, outlineColor: Cesium.Color.BLACK,
      verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
      pixelOffset: new Cesium.Cartesian2(0, -18),
      fillColor: color,
    },
  });
  routeEntities.push(e);
}

function clearRoutes() {
  routePrims.forEach(p => viewer.scene.primitives.remove(p));
  routePrims.length = 0;
  routeEntities.forEach(e => viewer.entities.remove(e));
  routeEntities.length = 0;
}

function updateRouteUI() {
  if (!routeData) return;
  document.getElementById("route-results").classList.remove("hidden");

  const dr = routeData.distance_route || {};
  const wr = routeData.wind_route     || {};

  document.getElementById("dist-distance").textContent = Math.round(dr.distance_m || 0);
  document.getElementById("dist-time").textContent     = Math.round(dr.time_s     || 0);
  document.getElementById("dist-energy").textContent   = (dr.energy_wh || 0).toFixed(1);

  document.getElementById("wind-distance").textContent = Math.round(wr.distance_m || 0);
  document.getElementById("wind-time").textContent     = Math.round(wr.time_s     || 0);
  document.getElementById("wind-energy").textContent   = (wr.energy_wh || 0).toFixed(1);

  animateSavings(Math.max(0, routeData.energy_savings_pct || 0));
}

function animateSavings(target) {
  const el    = document.getElementById("savings-pct");
  const label = document.querySelector(".savings-label");
  if (target < 0.5) {
    el.textContent = "< 1";
    if (label) label.textContent = "% energy saving (similar routes)";
    return;
  }
  if (label) label.textContent = "% energy saving";
  let cur = 0;
  const step = Math.max(0.3, target / 40);
  const iv = setInterval(() => {
    cur = Math.min(cur + step, target);
    el.textContent = cur.toFixed(1);
    if (cur >= target) clearInterval(iv);
  }, 30);
}

// ── Streamline rendering ──────────────────────────────────────────────

function rebuildStreamlines() {
  clearStreamlines();
  if (!currentData || !currentData.streamlines || !currentData.streamlines.length) return;
  const opacity = parseInt(document.getElementById("sl-opacity").value) / 100;
  const width   = parseInt(document.getElementById("sl-width").value);
  addPolylines(currentData, opacity, width);
  if (document.getElementById("ck-glow").checked)
    addPolylines(currentData, opacity * 0.22, width * 3);
  if (document.getElementById("ck-particles").checked)
    initParticles(currentData);
}

function addPolylines(data, opacity, width) {
  const insts = [];
  for (const sl of data.streamlines) {
    const n = sl.num_points;
    if (n < 2) continue;
    const pos = [];
    const col = [];
    for (let i = 0; i < n; i++) {
      pos.push(Cesium.Cartesian3.fromDegrees(
        sl.positions[i*3], sl.positions[i*3+1], sl.positions[i*3+2]));
      col.push(new Cesium.Color(
        sl.colors[i*4]/255, sl.colors[i*4+1]/255,
        sl.colors[i*4+2]/255, (sl.colors[i*4+3]/255) * opacity));
    }
    try {
      insts.push(new Cesium.GeometryInstance({
        geometry: new Cesium.PolylineGeometry({
          positions: pos, colors: col, width,
          colorsPerVertex: true, followSurface: false }),
      }));
    } catch (_) {}
  }
  if (!insts.length) return;
  const prim = new Cesium.Primitive({
    geometryInstances: insts,
    appearance: new Cesium.PolylineColorAppearance({ translucent: true }),
    asynchronous: true,
  });
  viewer.scene.primitives.add(prim);
  primaries.push(prim);
}

function clearStreamlines() {
  primaries.forEach(p => viewer.scene.primitives.remove(p));
  primaries.length = 0;
  if (pointCollection) {
    viewer.scene.primitives.remove(pointCollection);
    pointCollection = null;
  }
  particles = [];
}

function clearAll() {
  clearStreamlines();
  clearRoutes();
  currentData = null;
  routeData   = null;
  document.getElementById("route-results").classList.add("hidden");
}

// ── Particles ─────────────────────────────────────────────────────────

function initParticles(data) {
  if (pointCollection) {
    viewer.scene.primitives.remove(pointCollection);
    pointCollection = null;
  }
  particles       = [];
  pointCollection = new Cesium.PointPrimitiveCollection();
  viewer.scene.primitives.add(pointCollection);

  const subset = data.streamlines.slice(0, 200);
  for (const sl of subset) {
    if (sl.num_points < 4) continue;
    for (let k = 0; k < 6; k++) {
      const pt = pointCollection.add({
        position: Cesium.Cartesian3.ZERO,
        pixelSize: 5,
        color: new Cesium.Color(1,1,1,0.8),
      });
      particles.push({ point: pt, sl, phase: k/6, speed: 0.07 + Math.random()*0.04 });
    }
  }
}

let _t = 0;
function tick() {
  if (!particles.length) return;
  _t += 0.016;
  for (const p of particles) {
    const t   = (p.phase + _t * p.speed) % 1.0;
    const n   = p.sl.num_points;
    const idx = Math.min(Math.floor(t * n), n-1);
    p.point.position = Cesium.Cartesian3.fromDegrees(
      p.sl.positions[idx*3], p.sl.positions[idx*3+1], p.sl.positions[idx*3+2]);
    p.point.color = new Cesium.Color(
      p.sl.colors[idx*4]/255, p.sl.colors[idx*4+1]/255,
      p.sl.colors[idx*4+2]/255, 0.85);
    p.point.pixelSize = 4 + (p.sl.colors[idx*4+3]/255) * 4;
  }
}

// ── Utilities ─────────────────────────────────────────────────────────

function sleep(ms)    { return new Promise(r => setTimeout(r, ms)); }
function setStatus(m) { document.getElementById("status").textContent = m; }

boot();
