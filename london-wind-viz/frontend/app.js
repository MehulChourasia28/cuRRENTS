/* NVIDIA Drone Delivery – Wind Analysis frontend */

let viewer, cfg, tileset;
let currentData = null;
let routeData = null;
const primaries = [];
let particles = [];
let pointCollection = null;
const routePrimitives = [];
const routeEntities = [];

// ═════════════════════════════════════════════════════════════════════
// Presets
// ═════════════════════════════════════════════════════════════════════

const PRESETS = [
  { name: "Custom", origin: null },
  { name: "Leadenhall → Heron Tower",
    origin: "51.5130, -0.0830", dest: "51.5158, -0.0805", oh: 100, dh: 110 },
  { name: "Royal Exchange → Gherkin",
    origin: "51.5128, -0.0850", dest: "51.5150, -0.0800", oh: 90, dh: 100 },
  { name: "Liverpool St → Bishopsgate",
    origin: "51.5180, -0.0835", dest: "51.5145, -0.0810", oh: 100, dh: 110 },
  { name: "Tower 42 → The Scalpel",
    origin: "51.5160, -0.0820", dest: "51.5130, -0.0815", oh: 95, dh: 100 },
];

// ═════════════════════════════════════════════════════════════════════
// Pipeline progress stepper
// ═════════════════════════════════════════════════════════════════════

const PIPELINE_STAGES = ["scanning", "cache", "wind", "nemotron", "lbm", "streamlines", "routing", "cuopt"];
const STAGE_TO_STEP = {
  scanning: 0, cache: 0, wind: 1, nemotron: 2, lbm: 3, streamlines: 4, routing: 5, cuopt: 5,
};

// ═════════════════════════════════════════════════════════════════════
// Bootstrap
// ═════════════════════════════════════════════════════════════════════

async function boot() {
  try {
    const r = await fetch("/api/config");
    cfg = await r.json();
  } catch {
    status("Cannot reach backend");
    return;
  }

  const hasToken = cfg.cesium_token && !cfg.cesium_token.startsWith("YOUR_");
  if (hasToken) Cesium.Ion.defaultAccessToken = cfg.cesium_token;

  viewer = new Cesium.Viewer("cesiumContainer", {
    globe: !hasToken ? undefined : false,
    skyAtmosphere: hasToken ? false : undefined,
    shadows: false, animation: false, timeline: false,
    baseLayerPicker: false, geocoder: false, homeButton: false,
    sceneModePicker: false, navigationHelpButton: false,
    requestRenderMode: false,
  });

  if (hasToken) {
    try {
      tileset = await Cesium.Cesium3DTileset.fromIonAssetId(2275207);
      viewer.scene.primitives.add(tileset);
    } catch (e) { console.error(e); }
  }

  flyTo(cfg.center_lat, cfg.center_lon);
  wireControls();
  document.getElementById("ck-particles").checked = true;
  viewer.scene.preRender.addEventListener(tick);
}

function flyTo(lat, lon) {
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(lon - 0.001, lat - 0.003, 550),
    orientation: {
      heading: Cesium.Math.toRadians(15),
      pitch: Cesium.Math.toRadians(-32),
      roll: 0,
    },
    duration: 1.5,
  });
}

// ═════════════════════════════════════════════════════════════════════
// Controls
// ═════════════════════════════════════════════════════════════════════

function wireControls() {
  document.getElementById("btn-analyze").onclick = analyzeRoute;
  document.getElementById("btn-scan").onclick = scanTileGeometry;
  document.getElementById("sl-opacity").oninput = rebuildStreamlines;
  document.getElementById("sl-width").oninput = rebuildStreamlines;
  document.getElementById("ck-glow").onchange = rebuildStreamlines;
  document.getElementById("ck-particles").onchange = rebuildStreamlines;
  document.getElementById("preset-route").onchange = applyPreset;
}

// ═════════════════════════════════════════════════════════════════════
// Preset handling
// ═════════════════════════════════════════════════════════════════════

function applyPreset() {
  const sel = document.getElementById("preset-route");
  const idx = sel.selectedIndex;
  const fields = document.getElementById("coord-inputs-section");

  if (idx === 0) {
    if (fields) fields.classList.remove("hidden");
    return;
  }

  const p = PRESETS[idx];
  if (!p || !p.origin) return;

  document.getElementById("inp-origin").value = p.origin;
  document.getElementById("inp-dest").value = p.dest;
  document.getElementById("inp-origin-h").value = String(p.oh ?? 30);
  document.getElementById("inp-dest-h").value = String(p.dh ?? 30);

  if (fields) fields.classList.add("hidden");
}

// ═════════════════════════════════════════════════════════════════════
// Pipeline progress UI
// ═════════════════════════════════════════════════════════════════════

function showPipelineProgress() {
  const div = document.getElementById("pipeline-progress");
  if (!div) return;
  div.classList.remove("hidden");
  const steps = div.querySelectorAll(".step");
  steps.forEach((s) => {
    s.classList.remove("done", "active");
  });
}

function setPipelineStep(stageKey) {
  const stepIdx = STAGE_TO_STEP[stageKey] ?? -1;
  const div = document.getElementById("pipeline-progress");
  if (!div) return;
  const steps = div.querySelectorAll(".step");
  steps.forEach((s, i) => {
    s.classList.remove("done", "active");
    if (i < stepIdx) s.classList.add("done");
    else if (i === stepIdx) s.classList.add("active");
  });
}

function hidePipelineProgress() {
  const div = document.getElementById("pipeline-progress");
  if (!div) return;
  const steps = div.querySelectorAll(".step");
  steps.forEach((s) => {
    s.classList.remove("active");
    s.classList.add("done");
  });
  setTimeout(() => { div.classList.add("hidden"); }, 1200);
}

// ═════════════════════════════════════════════════════════════════════
// Analyze Route (replaces old setDomain)
// ═════════════════════════════════════════════════════════════════════

async function analyzeRoute() {
  const parse = (id) => {
    const parts = document.getElementById(id).value.split(",").map(s => parseFloat(s.trim()));
    return { lat: parts[0], lon: parts[1] };
  };
  const origin = parse("inp-origin");
  const dest = parse("inp-dest");
  const originH = parseFloat(document.getElementById("inp-origin-h").value) || 30;
  const destH = parseFloat(document.getElementById("inp-dest-h").value) || 30;

  if ([origin.lat, origin.lon, dest.lat, dest.lon].some(isNaN)) {
    status("Invalid coordinates"); return;
  }

  status("Setting domain …");
  document.getElementById("btn-analyze").disabled = true;
  clearStreamlinePrimitives();
  clearRoutes();
  currentData = null;
  routeData = null;
  document.getElementById("route-results").classList.add("hidden");
  showPipelineProgress();

  try {
    const r = await fetch("/api/coords", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ origin, dest, origin_height: originH, dest_height: destH }),
    });
    const result = await r.json();
    if (result.error) { status("Error: " + result.error); return; }

    cfg.center_lat = result.center_lat;
    cfg.center_lon = result.center_lon;
    cfg.domain_half_x = result.half_x;
    cfg.domain_half_y = result.half_y;

    flyTo(result.center_lat, result.center_lon);

    // Show origin + destination markers immediately
    showInputMarkers(origin, dest, originH, destH);

    if (result.cached) {
      status("Cached geometry found. Pipeline starting …");
      pollPipeline();
    } else if (tileset) {
      status("Scanning tile geometry …");
      await scanTileGeometry();
    } else {
      status("No tileset loaded. Click Scan Tile Geometry manually.");
    }
  } catch (e) {
    console.error(e);
    status("Failed to set domain");
  } finally {
    document.getElementById("btn-analyze").disabled = false;
  }
}

function showInputMarkers(origin, dest, originH, destH) {
  clearRoutes();
  const geh = (cfg && cfg.ground_ellipsoid_height) || 58;
  addMarker([origin.lon, origin.lat, geh + originH], "A", Cesium.Color.fromCssColorString("#00d2ff"));
  addMarker([dest.lon, dest.lat, geh + destH], "B", Cesium.Color.fromCssColorString("#76B900"));
}

// ═════════════════════════════════════════════════════════════════════
// Heightmap scanning
// ═════════════════════════════════════════════════════════════════════

async function scanTileGeometry() {
  const btn = document.getElementById("btn-scan");
  btn.disabled = true;

  if (!tileset) { status("No 3D tileset loaded"); btn.disabled = false; return; }

  const sampleRes = cfg.heightmap_sample_res || 4.0;
  const halfX = cfg.domain_half_x;
  const halfY = cfg.domain_half_y;
  const nx = Math.floor(2 * halfX / sampleRes);
  const ny = Math.floor(2 * halfY / sampleRes);
  const total = nx * ny;

  const latPerM = 1.0 / 111320.0;
  const lonPerM = 1.0 / (111320.0 * Math.cos(cfg.center_lat * Math.PI / 180));

  status("Preparing " + total + " sample points …");
  await sleep(50);

  const positions = new Array(total);
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const localX = -halfX + (ix + 0.5) * sampleRes;
      const localY = -halfY + (iy + 0.5) * sampleRes;
      positions[iy * nx + ix] = Cesium.Cartographic.fromDegrees(
        cfg.center_lon + localX * lonPerM,
        cfg.center_lat + localY * latPerM);
    }
  }

  const BATCH = 500;
  const heights = new Float64Array(total);
  for (let i = 0; i < total; i += BATCH) {
    const batch = positions.slice(i, Math.min(i + BATCH, total));
    try {
      const results = await viewer.scene.sampleHeightMostDetailed(batch);
      for (let j = 0; j < results.length; j++) {
        const h = results[j].height;
        heights[i + j] = (h !== undefined && isFinite(h)) ? h : NaN;
      }
    } catch (e) {
      for (let j = 0; j < batch.length; j++) heights[i + j] = NaN;
    }
    status("Scanning: " + Math.min(i + BATCH, total) + " / " + total);
    await sleep(10);
  }

  status("Uploading heightmap …");
  try {
    const resp = await fetch("/api/heightmap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ heights: Array.from(heights), nx, ny, resolution: sampleRes }),
    });
    const result = await resp.json();
    if (result.error) { status("Server error: " + result.error); }
    else { status("Geometry captured (" + result.coverage + "% buildings). Pipeline running …"); }
  } catch (e) { status("Upload failed"); }

  btn.disabled = false;
  pollPipeline();
}

async function pollPipeline() {
  for (let i = 0; i < 600; i++) {
    await sleep(2000);
    try {
      const r = await fetch("/api/pipeline-status");
      const s = await r.json();
      const stageLabels = {
        scanning: "Scanning tile geometry",
        cache: "Loading cached geometry",
        wind: "Fetching wind data",
        nemotron: "Nemotron scenario gen",
        lbm: "GPU fluid simulation",
        streamlines: "Computing streamlines",
        routing: "Computing drone routes",
        cuopt: "cuOpt route optimisation",
      };
      const label = stageLabels[s.stage] || s.stage;
      status(label + " — " + s.detail);
      setPipelineStep(s.stage);
      if (s.stage === "done") {
        hidePipelineProgress();
        await loadCombinedStreamlines();
        await loadRoutes();
        return;
      }
      if (s.stage === "error") {
        hidePipelineProgress();
        status("Pipeline error: " + s.detail);
        return;
      }
    } catch { /* ignore */ }
  }
}

// ═════════════════════════════════════════════════════════════════════
// Streamline loading
// ═════════════════════════════════════════════════════════════════════

async function loadCombinedStreamlines() {
  try {
    const r = await fetch("/api/streamlines/combined");
    if (!r.ok) return;
    currentData = await r.json();
    rebuildStreamlines();
  } catch { /* not ready yet */ }
}

// ═════════════════════════════════════════════════════════════════════
// Route loading + rendering
// ═════════════════════════════════════════════════════════════════════

async function loadRoutes() {
  try {
    const r = await fetch("/api/routes");
    if (!r.ok) return;
    routeData = await r.json();
    renderRoutes();
    updateRouteUI();
  } catch { /* not ready yet */ }
}

function renderRoutes() {
  clearRoutes();
  if (!routeData) return;

  const geh = cfg.ground_ellipsoid_height || 58;

  if (routeData.distance_route && routeData.distance_route.path.length > 1) {
    addRoutePolyline(routeData.distance_route.path,
      new Cesium.Color(0, 0.82, 1.0, 0.85), 3, false);
  }

  if (routeData.wind_route && routeData.wind_route.path.length > 1) {
    addRoutePolyline(routeData.wind_route.path,
      new Cesium.Color(0.46, 0.73, 0, 1.0), 5, true);
  }

  // Origin / destination markers
  if (routeData.distance_route && routeData.distance_route.path.length > 0) {
    const op = routeData.distance_route.path[0];
    const dp = routeData.distance_route.path[routeData.distance_route.path.length - 1];
    addMarker(op, "A", Cesium.Color.fromCssColorString("#00d2ff"));
    addMarker(dp, "B", Cesium.Color.fromCssColorString("#76B900"));
  }
}

function addRoutePolyline(path, color, width, glow) {
  const positions = [];
  for (const p of path) {
    positions.push(Cesium.Cartesian3.fromDegrees(p[0], p[1], p[2]));
  }

  const n = positions.length;
  const colors = new Array(n).fill(color);

  try {
    const inst = new Cesium.GeometryInstance({
      geometry: new Cesium.PolylineGeometry({
        positions, colors, width, colorsPerVertex: true, followSurface: false,
      }),
    });
    const prim = new Cesium.Primitive({
      geometryInstances: [inst],
      appearance: new Cesium.PolylineColorAppearance({ translucent: true }),
      asynchronous: true,
    });
    viewer.scene.primitives.add(prim);
    routePrimitives.push(prim);

    if (glow) {
      const glowColor = new Cesium.Color(color.red, color.green, color.blue, 0.2);
      const glowColors = new Array(n).fill(glowColor);
      const gi = new Cesium.GeometryInstance({
        geometry: new Cesium.PolylineGeometry({
          positions, colors: glowColors, width: width * 4,
          colorsPerVertex: true, followSurface: false,
        }),
      });
      const gp = new Cesium.Primitive({
        geometryInstances: [gi],
        appearance: new Cesium.PolylineColorAppearance({ translucent: true }),
        asynchronous: true,
      });
      viewer.scene.primitives.add(gp);
      routePrimitives.push(gp);
    }
  } catch (e) { console.error("Route polyline error:", e); }
}

function addMarker(coords, label, color) {
  const e = viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(coords[0], coords[1], coords[2]),
    point: { pixelSize: 14, color: color, outlineColor: Cesium.Color.WHITE, outlineWidth: 2 },
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
  for (const p of routePrimitives) viewer.scene.primitives.remove(p);
  routePrimitives.length = 0;
  for (const e of routeEntities) viewer.entities.remove(e);
  routeEntities.length = 0;
}

function updateRouteUI() {
  if (!routeData) return;
  const panel = document.getElementById("route-results");
  panel.classList.remove("hidden");

  const dr = routeData.distance_route || {};
  const wr = routeData.wind_route || {};

  document.getElementById("dist-distance").textContent = Math.round(dr.distance_m || 0);
  document.getElementById("dist-time").textContent = Math.round(dr.time_s || 0);
  document.getElementById("dist-energy").textContent = (dr.energy_wh || 0).toFixed(1);

  document.getElementById("wind-distance").textContent = Math.round(wr.distance_m || 0);
  document.getElementById("wind-time").textContent = Math.round(wr.time_s || 0);
  document.getElementById("wind-energy").textContent = (wr.energy_wh || 0).toFixed(1);

  const savings = routeData.energy_savings_pct || 0;
  const badge = document.querySelector(".savings-badge");
  const label = document.getElementById("savings-label");

  if (savings >= 0) {
    badge.classList.remove("savings-negative");
    label.textContent = "energy saved";
    animateSavings(savings);
  } else {
    badge.classList.add("savings-negative");
    const windReduction = routeData.wind_reduction_pct || 0;
    if (windReduction > 0) {
      label.textContent = "wind exposure reduced";
      animateSavings(windReduction);
    } else {
      label.textContent = "safer route (less turbulence)";
      animateSavings(0);
    }
  }
}

function animateSavings(target) {
  const el = document.getElementById("savings-pct");
  let current = 0;
  const absTarget = Math.abs(target);
  const step = Math.max(0.5, absTarget / 40);
  const iv = setInterval(() => {
    current = Math.min(current + step, absTarget);
    el.textContent = current.toFixed(1);
    if (current >= absTarget) clearInterval(iv);
  }, 30);
}

// ═════════════════════════════════════════════════════════════════════
// Streamline rendering
// ═════════════════════════════════════════════════════════════════════

function rebuildStreamlines() {
  clearStreamlinePrimitives();
  if (!currentData || !currentData.streamlines || !currentData.streamlines.length) return;
  const opacity = parseInt(document.getElementById("sl-opacity").value) / 100;
  const width = parseInt(document.getElementById("sl-width").value);
  addPolylines(currentData, opacity, width);
  if (document.getElementById("ck-glow").checked) addPolylines(currentData, opacity * 0.25, width * 3);
  if (document.getElementById("ck-particles").checked) initParticles(currentData);
}

function addPolylines(data, opacity, width) {
  const insts = [];
  for (const sl of data.streamlines) {
    const n = sl.num_points;
    if (n < 2) continue;
    const pos = new Array(n);
    for (let i = 0; i < n; i++)
      pos[i] = Cesium.Cartesian3.fromDegrees(sl.positions[i*3], sl.positions[i*3+1], sl.positions[i*3+2]);
    const col = new Array(n);
    for (let i = 0; i < n; i++)
      col[i] = new Cesium.Color(sl.colors[i*4]/255, sl.colors[i*4+1]/255,
                                 sl.colors[i*4+2]/255, (sl.colors[i*4+3]/255)*opacity);
    try {
      insts.push(new Cesium.GeometryInstance({
        geometry: new Cesium.PolylineGeometry({
          positions: pos, colors: col, width, colorsPerVertex: true, followSurface: false }),
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

function clearStreamlinePrimitives() {
  for (const p of primaries) viewer.scene.primitives.remove(p);
  primaries.length = 0;
  if (pointCollection) { viewer.scene.primitives.remove(pointCollection); pointCollection = null; }
  particles = [];
}

// ═════════════════════════════════════════════════════════════════════
// Particles
// ═════════════════════════════════════════════════════════════════════

function initParticles(data) {
  if (pointCollection) { viewer.scene.primitives.remove(pointCollection); pointCollection = null; }
  particles = [];
  pointCollection = new Cesium.PointPrimitiveCollection();
  viewer.scene.primitives.add(pointCollection);
  const subset = data.streamlines.slice(0, 200);
  for (const sl of subset) {
    if (sl.num_points < 4) continue;
    for (let p = 0; p < 6; p++) {
      const pt = pointCollection.add({
        position: Cesium.Cartesian3.ZERO,
        pixelSize: 5,
        color: new Cesium.Color(1, 1, 1, 0.8),
      });
      particles.push({ point: pt, sl, phase: p / 6, speed: 0.08 + Math.random() * 0.04 });
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
    p.point.position = Cesium.Cartesian3.fromDegrees(
      p.sl.positions[idx*3], p.sl.positions[idx*3+1], p.sl.positions[idx*3+2]);
    p.point.color = new Cesium.Color(
      p.sl.colors[idx*4]/255, p.sl.colors[idx*4+1]/255, p.sl.colors[idx*4+2]/255, 0.8);
    p.point.pixelSize = (4 + (p.sl.colors[idx*4+3] / 255) * 4) * (0.85 + 0.15 * Math.sin(_t * 3 + p.phase * 6.28));
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function status(msg) { document.getElementById("status").textContent = msg; }
boot();
