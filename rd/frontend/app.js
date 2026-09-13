/* =========================================================================
   ROADDOC — shared app logic (loaded by every page)
   Same CV heuristic / safety-engine maths as the Flask backend. Every DOM
   access is null-guarded so this one file can run on pages that only show
   a subset of the UI (Overview has the analyzer; Track Analysis, Car
   Status, Telemetry and System are read-only views of the same state,
   kept in sync via localStorage).
   ========================================================================= */
const COLORS = { Dry:'#34d399', Damp:'#3e7bfa', Drying:'#9b8cff', Wet:'#ef4453' };
const RISK_COLORS = { low:'#34d399', medium:'#f5a524', high:'#ff8a5c', critical:'#ef4453' };

function $(id) { return document.getElementById(id); }
function setText(id, val) { const el = $(id); if (el) el.textContent = val; }
function setHTML(id, val) { const el = $(id); if (el) el.innerHTML = val; }
function setWidth(id, val) { const el = $(id); if (el) el.style.width = val; }
function setLeft(id, val) { const el = $(id); if (el) el.style.left = val; }
function setColor(id, val) { const el = $(id); if (el) el.style.color = val; }
function setBg(id, val) { const el = $(id); if (el) el.style.background = val; }
function setClass(id, val) { const el = $(id); if (el) el.className = val; }
function addClass(id, val) { const el = $(id); if (el) el.classList.add(val); }
function removeClass(id, val) { const el = $(id); if (el) el.classList.remove(val); }
function display(id, val) { const el = $(id); if (el) el.style.display = val; }

/* ---------------------------------- persisted state ---------------------------------- */
const STORE_KEY = 'roaddoc_state_v1';
const DEFAULT_FORM = { vehicle_type: 'car', driving_condition: 'normal', speed: '', tyre_condition: '', tyre_pressure: '', tyre_recommended_pressure: '', tyre_temperature: '', location: '' };

const STORE = {
  load() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; }
    catch { return {}; }
  },
  save(patch) {
    const cur = STORE.load();
    const next = { ...cur, ...patch };
    try { localStorage.setItem(STORE_KEY, JSON.stringify(next)); } catch {}
    return next;
  },
};

/* ========================= CV fallback engine (unchanged maths) ========================= */
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60; if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  const v = max;
  return [h, s * 255, v * 255];
}

function getChannels(img, W = 240, H = 160) {
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, W, H);
  const data = ctx.getImageData(0, 0, W, H).data;
  const v = new Float32Array(W * H), s = new Float32Array(W * H), gray = new Float32Array(W * H);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const [, sat, val] = rgbToHsv(r, g, b);
    v[p] = val; s[p] = sat;
    gray[p] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return { width: W, height: H, v, s, gray };
}

function boxBlur(arr, W, H, radius = 1) {
  const out = new Float32Array(arr.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let sum = 0, n = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy >= 0 && yy < H && xx >= 0 && xx < W) { sum += arr[yy * W + xx]; n++; }
        }
      }
      out[y * W + x] = sum / n;
    }
  }
  return out;
}

function edgeDensity(gray, W, H, yStart) {
  let count = 0, total = 0;
  for (let y = yStart; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const gxL = x > 0 ? gray[y * W + x - 1] : gray[y * W + x];
      const gyU = y > yStart ? gray[(y - 1) * W + x] : gray[y * W + x];
      const gx = gray[y * W + x] - gxL, gy = gray[y * W + x] - gyU;
      if (Math.sqrt(gx * gx + gy * gy) > 20) count++;
      total++;
    }
  }
  return count / total;
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function pseudoProbabilities(wetness, patchiness) {
  const dry = wetness < 0.22 ? Math.max(0, 1 - wetness / 0.22) : Math.max(0, 1 - (wetness - 0.22) / 0.35);
  const wet = wetness > 0.60 ? Math.max(0, (wetness - 0.60) / 0.40) : 0;
  const mid = Math.max(0, 1 - dry - wet);
  const drying = mid * Math.min(patchiness / 0.6, 1.0);
  const damp = mid - drying;
  const total = dry + damp + drying + wet;
  if (total <= 0) return { Dry: 0.25, Damp: 0.25, Drying: 0.25, Wet: 0.25 };
  return { Dry: dry / total, Damp: damp / total, Drying: drying / total, Wet: wet / total };
}

function analyzeFrame(img) {
  const { width: W, height: H, v, s, gray } = getChannels(img);
  const vDenoised = boxBlur(v, W, H, 1);

  const vMin = Math.min(...vDenoised), vMax = Math.max(...vDenoised);
  const vMed = median(vDenoised);
  const brightThresh = Math.max(vMin + 0.55 * (vMax - vMin), vMed + 25);

  let glareCount = 0;
  for (let i = 0; i < vDenoised.length; i++) {
    if (vDenoised[i] > brightThresh && s[i] < 70) glareCount++;
  }
  const reflectionRatio = glareCount / vDenoised.length;

  const yStart = Math.floor(H / 2);
  let vSum = 0, sSum = 0, n = 0;
  for (let y = yStart; y < H; y++) {
    for (let x = 0; x < W; x++) {
      vSum += v[y * W + x]; sSum += s[y * W + x]; n++;
    }
  }
  const darknessScore = 1 - (vSum / n) / 255;
  const saturationScore = (sSum / n) / 255;

  const textureScore = edgeDensity(gray, W, H, yStart);
  const textureScoreNorm = Math.min(textureScore / 0.15, 1.0);

  let wetnessScore =
    0.55 * Math.min(reflectionRatio * 6.0, 1.0) +
    0.25 * (1.0 - textureScoreNorm) +
    0.12 * darknessScore +
    0.08 * saturationScore;
  wetnessScore = Math.min(Math.max(wetnessScore, 0), 1);

  const grid = 4;
  const ph = Math.floor(H / grid), pw = Math.floor(W / grid);
  const patchScores = [];
  for (let gy = 0; gy < grid; gy++) {
    for (let gx = 0; gx < grid; gx++) {
      let glareP = 0, vSumP = 0, nP = 0;
      for (let y = gy * ph; y < (gy + 1) * ph; y++) {
        for (let x = gx * pw; x < (gx + 1) * pw; x++) {
          const idx = y * W + x;
          if (vDenoised[idx] > brightThresh && s[idx] < 70) glareP++;
          vSumP += vDenoised[idx]; nP++;
        }
      }
      if (nP === 0) continue;
      const glareRatioP = glareP / nP;
      const darkP = 1 - (vSumP / nP) / 255;
      patchScores.push(Math.min(0.7 * glareRatioP * 6.0, 1.0) * 0.6 + darkP * 0.4);
    }
  }
  const pMean = patchScores.reduce((a, b) => a + b, 0) / patchScores.length;
  const variance = patchScores.reduce((a, b) => a + (b - pMean) ** 2, 0) / patchScores.length;
  let patchinessScore = Math.sqrt(variance) * 3.0;
  patchinessScore = Math.min(Math.max(patchinessScore, 0), 1);

  let label;
  if (wetnessScore < 0.22) label = 'Dry';
  else if (wetnessScore >= 0.60) label = 'Wet';
  else if (patchinessScore > 0.35) label = 'Drying';
  else label = 'Damp';

  return {
    label,
    wetness_score: Math.round(wetnessScore * 1000) / 1000,
    patchiness_score: Math.round(patchinessScore * 1000) / 1000,
    reflection_ratio: Math.round(reflectionRatio * 10000) / 10000,
    texture_score: Math.round(textureScore * 10000) / 10000,
    probabilities: pseudoProbabilities(wetnessScore, patchinessScore),
    engine: 'heuristic-cv-fallback',
  };
}

function trendFor(history) {
  const recent = history.slice(-8).map(h => h.wetness_score);
  const n = recent.length;
  if (n < 2) return { trend: 'Not enough data yet', slope: null };
  const xs = [...Array(n).keys()];
  const xm = xs.reduce((a, b) => a + b, 0) / n, ym = recent.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (xs[i] - xm) * (recent[i] - ym); den += (xs[i] - xm) ** 2; }
  const slope = den === 0 ? 0 : num / den;
  if (slope > 0.02) return { trend: 'Getting wetter', slope };
  if (slope < -0.02) return { trend: 'Drying out', slope };
  return { trend: 'Stable', slope };
}

function suggestionFor(label, trend) {
  if (label === 'Wet' && trend !== 'Drying out') return 'Track is wet. Stay on full wet tyres.';
  if (label === 'Wet') return 'Track is wet but drying. Hold wets for now, watch the next laps closely.';
  if (label === 'Drying') return 'Track drying: tyre change window approaching — get intermediates ready.';
  if (label === 'Damp' && trend === 'Drying out') return 'Damp and drying fast. Consider switching to intermediates soon.';
  if (label === 'Damp') return 'Track is damp. Intermediates recommended, monitor closely.';
  if (label === 'Dry' && trend === 'Getting wetter') return 'Still dry but conditions worsening — be ready to react quickly.';
  if (label === 'Dry') return 'Track is dry and stable. No tyre change needed.';
  return 'Monitoring conditions.';
}

const BASE_SAFE_SPEED = { dry: 120, damp: 90, drying: 100, wet: 70, 'standing water': 40 };
const VEHICLE_SPEED_FACTOR = { car: 1.0, suv: 0.95, bike: 0.8, bus: 0.75, truck: 0.7 };
const DRIVING_FACTOR = { normal: 1.0, aggressive: 0.85, highway: 1.05, city: 0.9, 'off-road': 0.8 };

function computeSafetyClientSide(label, trend, wetnessScore, form) {
  const waterAlert = wetnessScore >= 0.88;
  const key = waterAlert ? 'standing water' : (label || 'dry').toLowerCase();
  const vType = VEHICLE_SPEED_FACTOR[form.vehicle_type] ? form.vehicle_type : 'car';
  const dCond = DRIVING_FACTOR[form.driving_condition] ? form.driving_condition : 'normal';
  const maxSpeed = Math.round((BASE_SAFE_SPEED[key] ?? 120) * VEHICLE_SPEED_FACTOR[vType] * DRIVING_FACTOR[dCond]);
  const speed = form.speed !== '' && form.speed != null ? Number(form.speed) : null;

  let speedWarning;
  if (speed === null) speedWarning = `No live speed reported. Recommended max speed for current conditions: ${maxSpeed} km/h.`;
  else if (speed > maxSpeed) speedWarning = `Speed too high for conditions: travelling at ${speed} km/h, recommended max is ${maxSpeed} km/h (${Math.round(speed - maxSpeed)} km/h over).`;
  else speedWarning = `Speed within safe range (${speed} km/h ≤ ${maxSpeed} km/h recommended max).`;
  const speedOver = speed !== null && speed > maxSpeed;

  const pressure = form.tyre_pressure !== '' && form.tyre_pressure != null ? Number(form.tyre_pressure) : null;
  const recPressure = form.tyre_recommended_pressure !== '' && form.tyre_recommended_pressure != null ? Number(form.tyre_recommended_pressure) : null;
  let pressureStatus = 'unknown';
  if (pressure !== null && recPressure) {
    const dev = (pressure - recPressure) / recPressure;
    pressureStatus = dev <= -0.20 ? 'critical-low' : dev <= -0.10 ? 'low' : dev >= 0.15 ? 'high' : 'ok';
  }
  let pressureWarning;
  if (pressureStatus === 'unknown') pressureWarning = 'Tyre pressure not reported — recommend checking manually before the next run.';
  else if (pressureStatus === 'critical-low') pressureWarning = `⚠️ Critically low tyre pressure (${pressure} PSI vs recommended ${recPressure} PSI). Aquaplaning and blowout risk increases sharply — reduce speed and check tyres immediately.`;
  else if (pressureStatus === 'low') pressureWarning = `Tyre pressure is low (${pressure} PSI vs recommended ${recPressure} PSI). Grip and handling margin reduced.`;
  else if (pressureStatus === 'high') pressureWarning = `Tyre pressure is high (${pressure} PSI vs recommended ${recPressure} PSI). Contact patch is reduced, especially on wet surfaces.`;
  else pressureWarning = `Tyre pressure is within range (${pressure} PSI vs recommended ${recPressure} PSI).`;

  const condition = form.tyre_condition || null;
  const temp = form.tyre_temperature !== '' && form.tyre_temperature != null ? Number(form.tyre_temperature) : null;
  const reasons = [];
  if (['low', 'critical-low', 'high'].includes(pressureStatus)) reasons.push('pressure out of range');
  if (['worn', 'bald', 'damaged'].includes(condition)) reasons.push(`tyre condition reported as '${condition}'`);
  if (temp !== null && temp >= 100) reasons.push(`tyre temperature high (${temp}°C)`);
  if ((waterAlert || key === 'wet') && (['worn', 'bald', 'damaged'].includes(condition) || ['low', 'critical-low'].includes(pressureStatus))) reasons.push('wet/standing-water surface amplifies existing tyre risk');
  let inspection;
  if (reasons.length === 0) inspection = condition ? 'No tyre inspection needed right now — condition and pressure look fine.' : 'Tyre condition not reported — a visual check is recommended before extended running.';
  else inspection = 'Tyre inspection recommended: ' + reasons.join('; ') + '.';

  let roadWarning;
  const l = (label || '').toLowerCase();
  if (waterAlert) roadWarning = '⚠️ Standing water suspected on the surface ahead — high aquaplaning risk.';
  else if (l === 'wet') roadWarning = 'Wet surface detected. Grip is significantly reduced.';
  else if (l === 'drying') roadWarning = 'Surface is transitioning (drying). Expect uneven grip patch-to-patch.';
  else if (l === 'damp') roadWarning = 'Damp surface detected. Grip is moderately reduced.';
  else if (l === 'dry' && trend === 'Getting wetter') roadWarning = 'Surface currently dry, but conditions are worsening — stay alert.';
  else roadWarning = 'Surface condition is dry and stable.';

  let riskScore = 0;
  if (waterAlert) riskScore += 3; else if (l === 'wet') riskScore += 2; else if (l === 'drying' || l === 'damp') riskScore += 1;
  if (pressureStatus === 'critical-low') riskScore += 3; else if (['low', 'high'].includes(pressureStatus)) riskScore += 1;
  if (['bald', 'damaged'].includes(condition)) riskScore += 2; else if (condition === 'worn') riskScore += 1;
  if (speedOver) riskScore += 2;
  const riskLevel = riskScore >= 6 ? 'critical' : riskScore >= 4 ? 'high' : riskScore >= 2 ? 'medium' : 'low';
  const generalRec = {
    low: 'Conditions are stable — continue normal driving with routine monitoring.',
    medium: 'Stay attentive: road and/or tyre conditions are starting to work against you.',
    high: 'Reduce speed and increase following distance now — multiple risk factors are stacking up.',
    critical: 'Slow down immediately and consider pulling over to check tyres — conditions and vehicle state are both compromised.',
  }[riskLevel];

  return {
    standing_water_alert: waterAlert,
    risk_level: riskLevel,
    road_condition_warning: roadWarning,
    tyre_pressure_warning: pressureWarning,
    speed_related_warning: speedWarning,
    recommended_max_speed_kmh: maxSpeed,
    tyre_inspection_recommendation: inspection,
    general_safety_recommendation: generalRec,
  };
}

/* ========================= UI wiring (null-guarded) ========================= */
const fileInput = $('fileInput');
const fileInput2 = $('fileInput2');
const dropzone = $('dropzone');
const dzThumb = $('dzThumb');
const scanFrame = $('scanFrame');
const scanImg = $('scanImg');
const analyzeBtn = $('analyzeBtn');
const resetBtn = $('resetBtn');
const demoBtn = $('demoBtn');
const rescanBtn = $('rescanBtn');

const vehicleTypeEl = $('vehicleType');
const drivingConditionEl = $('drivingCondition');
const speedEl = $('speed');
const tyreConditionEl = $('tyreCondition');
const tyrePressureEl = $('tyrePressure');
const tyreRecommendedPressureEl = $('tyreRecommendedPressure');
const tyreTemperatureEl = $('tyreTemperature');
const locationEl = $('location');
const gpsBtn = $('gpsBtn');

const FORM_ELS_PRESENT = !!vehicleTypeEl;

let selectedFile = null;
let historyLog = (STORE.load().history) || [];
let frameCount = (STORE.load().frameCount) || 0;
const bootTime = Date.now();

function handleFile(file) {
  if (!file || !dzThumb) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  dzThumb.src = url; if (scanImg) { scanImg.src = url; scanImg.style.display = 'block'; }
  addClass('dropzone', 'has-file');
  removeClass('scanFrame', 'locked');
  addClass('scanFrame', 'has-file');
  if (analyzeBtn) analyzeBtn.disabled = false;
}
if (fileInput) fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
if (fileInput2) fileInput2.addEventListener('change', e => handleFile(e.target.files[0]));
if (dropzone) {
  dropzone.addEventListener('dragover', e => e.preventDefault());
  dropzone.addEventListener('drop', e => { e.preventDefault(); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
}
if (scanFrame) {
  scanFrame.addEventListener('dragover', e => e.preventDefault());
  scanFrame.addEventListener('drop', e => { e.preventDefault(); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
}
if (rescanBtn) rescanBtn.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); fileInput.click(); });

if (gpsBtn) gpsBtn.addEventListener('click', () => {
  if (!navigator.geolocation) { locationEl.placeholder = 'Geolocation not supported'; return; }
  gpsBtn.textContent = '📍 …';
  navigator.geolocation.getCurrentPosition(
    pos => { locationEl.value = `${pos.coords.latitude.toFixed(5)}, ${pos.coords.longitude.toFixed(5)}`; gpsBtn.textContent = '📍 GPS'; updateVehiclePanel(); },
    () => { gpsBtn.textContent = '📍 N/A'; setTimeout(() => gpsBtn.textContent = '📍 GPS', 1500); },
    { timeout: 6000 }
  );
});

/* Telemetry form: read from the DOM when this page has the fields (Car
   Status); otherwise fall back to the last value saved from that page. */
function readVehicleTyreForm() {
  if (FORM_ELS_PRESENT) {
    const f = {
      vehicle_type: vehicleTypeEl.value, driving_condition: drivingConditionEl.value, speed: speedEl.value,
      tyre_condition: tyreConditionEl.value, tyre_pressure: tyrePressureEl.value,
      tyre_recommended_pressure: tyreRecommendedPressureEl.value, tyre_temperature: tyreTemperatureEl.value,
      location: locationEl.value,
    };
    STORE.save({ telemetryForm: f });
    return f;
  }
  return STORE.load().telemetryForm || DEFAULT_FORM;
}

function applyStoredFormToInputs() {
  if (!FORM_ELS_PRESENT) return;
  const f = STORE.load().telemetryForm || DEFAULT_FORM;
  vehicleTypeEl.value = f.vehicle_type || 'car';
  drivingConditionEl.value = f.driving_condition || 'normal';
  speedEl.value = f.speed || '';
  tyreConditionEl.value = f.tyre_condition || '';
  tyrePressureEl.value = f.tyre_pressure || '';
  tyreRecommendedPressureEl.value = f.tyre_recommended_pressure || '';
  tyreTemperatureEl.value = f.tyre_temperature || '';
  locationEl.value = f.location || '';
}

function updateVehiclePanel() {
  const f = readVehicleTyreForm();
  setText('vsType', f.vehicle_type.charAt(0).toUpperCase() + f.vehicle_type.slice(1));
  setText('vsDriving', f.driving_condition.charAt(0).toUpperCase() + f.driving_condition.slice(1));
  setText('vsSpeed', f.speed ? `${f.speed} km/h` : 'not reported');
  setText('vsLocation', f.location || 'not reported');
  setText('cbSpeed', f.speed ? `${f.speed} km/h` : '—');
  setText('cbPressure', f.tyre_pressure ? `${f.tyre_pressure} PSI` : '—');
  setText('cbCondition', f.tyre_condition || '—');
}
if (FORM_ELS_PRESENT) {
  applyStoredFormToInputs();
  [vehicleTypeEl, drivingConditionEl, speedEl, tyreConditionEl, tyrePressureEl, tyreRecommendedPressureEl, tyreTemperatureEl, locationEl]
    .forEach(el => el.addEventListener('input', updateVehiclePanel));
}
updateVehiclePanel();

/* ---- clock / uptime (present in the shared header on every page) ---- */
function tickClock() {
  const now = new Date();
  setText('clock', now.toLocaleTimeString('en-GB'));
  setText('clockDate', now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }));
  const secs = Math.floor((Date.now() - bootTime) / 1000);
  setText('uptime', `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}`);
}
tickClock(); setInterval(tickClock, 1000);

/* ---- boot sequence ---- */
(function boot() {
  const fill = $('bootFill');
  const log = $('bootLog');
  if (!fill || !log) return;
  const stages = [
    [15, 'booting sensor array…'],
    [45, 'loading vision model…'],
    [70, 'calibrating grip thresholds…'],
    [92, 'linking safety decision engine…'],
    [100, 'ready.'],
  ];
  let i = 0;
  function step() {
    if (i >= stages.length) {
      setTimeout(() => addClass('boot', 'hide'), 250);
      return;
    }
    const [p, msg] = stages[i++];
    fill.style.width = p + '%';
    log.textContent = msg;
    setTimeout(step, 260);
  }
  step();
})();

/* ---- backend link status ---- */
const API_BASE = localStorage.getItem('weatherWhiplashApi') || 'http://localhost:5000';
setText('sysBackend', API_BASE);
function setLink(isLive) {
  const dot = $('linkDot'), txt = $('linkText');
  if (dot) dot.classList.toggle('live', isLive);
  if (txt) txt.textContent = isLive ? 'live' : 'offline';
  const pillTop = $('linkPillTop'), txtTop = $('linkTextTop');
  if (pillTop) pillTop.classList.toggle('live', isLive);
  if (txtTop) txtTop.textContent = isLive ? 'Live Feed' : 'Offline';
}
fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout ? AbortSignal.timeout(2500) : undefined })
  .then(r => setLink(r.ok)).catch(() => setLink(false));

async function analyzeWithBackend(file) {
  const form = new FormData();
  form.append('image', file);
  form.append('session', 'browser');
  const vt = readVehicleTyreForm();
  Object.entries(vt).forEach(([k, v]) => form.append(k, v ?? ''));
  const response = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form });
  if (!response.ok) throw new Error(`Backend returned ${response.status}`);
  return await response.json();
}

function setStatus(mode) {
  const el = $('statusLine');
  if (!el) return;
  el.classList.remove('busy', 'offline');
  if (mode === 'busy') { el.classList.add('busy'); el.querySelector('span').textContent = 'Analyzing…'; }
  else if (mode === 'offline') { el.classList.add('offline'); el.querySelector('span').textContent = 'Backend offline — using browser CV'; }
  else { el.querySelector('span').textContent = 'System ready'; }
}

async function runAnalysis(file) {
  setStatus('busy'); if (analyzeBtn) analyzeBtn.disabled = true;
  try {
    const result = await analyzeWithBackend(file);
    historyLog = result.history || historyLog;
    setLink(true);
    setText('engineTag', (result.engine || 'resnet18').toUpperCase());
    render({ ...result, history: historyLog });
    setStatus('ready');
  } catch (backendError) {
    console.warn('Backend unavailable; using browser CV fallback.', backendError);
    setLink(false);
    const img = new Image();
    img.onload = () => {
      const result = analyzeFrame(img);
      historyLog.push({ label: result.label, wetness_score: result.wetness_score });
      const { trend } = trendFor(historyLog);
      const safety = computeSafetyClientSide(result.label, trend, result.wetness_score, readVehicleTyreForm());
      setText('engineTag', 'BROWSER CV FALLBACK');
      render({ ...result, trend, suggestion: suggestionFor(result.label, trend), history: historyLog, safety });
      setStatus('offline');
    };
    img.src = URL.createObjectURL(file);
  } finally {
    if (analyzeBtn) analyzeBtn.disabled = false;
  }
}

if (analyzeBtn) analyzeBtn.addEventListener('click', () => { if (selectedFile) runAnalysis(selectedFile); });

document.querySelectorAll('.tf-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const canvas = makeSyntheticTrackCanvas(btn.dataset.kind);
    canvas.toBlob(blob => {
      const file = new File([blob], `${btn.dataset.kind}-test-feed.png`, { type: 'image/png' });
      handleFile(file);
      runAnalysis(file);
    });
  });
});

if (resetBtn) resetBtn.addEventListener('click', () => {
  historyLog = []; frameCount = 0; selectedFile = null;
  STORE.save({ history: [], frameCount: 0, latest: null });
  setText('frameCounter', '#0000');
  removeClass('dropzone', 'has-file'); removeClass('scanFrame', 'has-file'); removeClass('scanFrame', 'locked');
  display('scanTags', 'none');
  setText('stateLabel', 'STANDBY'); setColor('stateLabel', 'var(--dim)');
  setText('stateConf', '—');
  setText('condLabel', 'standby'); setBg('condDot', 'var(--dim-2)');
  ['probDry', 'probDamp', 'probDrying', 'probWet'].forEach(id => setWidth(id, '0%'));
  ['probDryV', 'probDampV', 'probDryingV', 'probWetV'].forEach(id => setText(id, '—'));
  setText('gripLevel', '—'); setText('wetnessV', '—');
  setText('patchV', '—'); setText('trendV', '—');
  setLeft('moistMarker', '0%');
  setText('suggestionText', 'Upload or run a test feed to get a recommendation.');
  setHTML('waterBanner', '');
  setText('speedReadout', '—'); const sr = $('speedRing'); if (sr) sr.style.setProperty('--pct', 0);
  setText('maxSpeedV', '— km/h');
  setText('tyreReading', '— PSI'); setText('tyreSub', 'tyre not reported');
  setClass('riskBadgeHud', 'risk-badge risk-low'); setText('riskBadgeHud', 'standby');
  setClass('riskBadgeSafety', 'risk-badge risk-low'); setText('riskBadgeSafety', 'standby');
  setHTML('safetyList', '<li><b>Road</b><span>Awaiting first frame.</span></li>');
  setHTML('logList', '<div style="padding:8px 0;color:var(--dim-2)">— no frames yet —</div>');
  setText('sysFrames', '0');
  setText('inferenceMs', 'inference —');
  const c = $('trendCanvas'); if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);

  // v2 mockup elements
  STORE.save({ activity: [] });
  setText('condPillTop', 'Standby');
  setText('gaugeScore', '—'); setText('gaugeBand', 'STANDBY'); setColor('gaugeBand', 'var(--dim)');
  setText('gaugeDesc', 'Awaiting first frame — run a scan to analyze the surface.');
  const arc = $('gaugeArc'); if (arc) arc.setAttribute('stroke-dasharray', '0 100');
  ['legendGood', 'legendRegular', 'legendPoor'].forEach(id => removeClass(id, 'on'));
  display('gridOverlay', 'none'); display('heatPatch', 'none');
  display('feedTopLeft', 'none'); display('feedDetected', 'none'); display('feedScanBadge', 'none');
  display('feedEmptyHint', 'flex');
  const scanImgEl = $('scanImg'); if (scanImgEl) { scanImgEl.style.display = 'none'; scanImgEl.removeAttribute('src'); }
  const sArc = $('speedArc'); if (sArc) sArc.setAttribute('stroke-dasharray', '0 100');
  setText('rpmVal', '—'); setText('fuelVal', '68%');
  setText('tyreActionLabel', 'Recommended Action');
  ['psiFL', 'psiFR', 'psiRL', 'psiRR'].forEach(id => setText(id, '— PSI'));
  setText('insightPressure', 'Normal'); setClass('insightPressureIc', 'ic good'); setText('insightPressureIc', '✅');
  setText('insightRisk', 'Low'); setClass('insightRiskIc', 'ic good'); setText('insightRiskIc', '✅');
  setText('insightSafety', 'Good'); setClass('insightSafetyIc', 'ic good');
  setText('insightAlert', 'None expected'); setClass('insightAlertIc', 'ic warn');
  renderActivity([]);

  // new dashboard v3 elements
  setText('speedReadout', '—'); setClass('speedSub', 'sub good'); setText('speedSub', 'Normal');
  setText('statTyrePressure', 'F — · R —'); setClass('tyrePressureSub', 'sub good'); setText('tyrePressureSub', 'Normal');
  setText('statTyreHealth', '—%'); setClass('tyreHealthSub', 'sub good'); setText('tyreHealthSub', 'Good');
  const riskCardEl = $('riskCard'); if (riskCardEl) riskCardEl.classList.add('risk-low');
  setText('riskLevelIc', '✅'); setText('riskLevelTxt', 'STANDBY');
  setText('riskSubTxt', 'Awaiting first frame · Speed — km/h');
  setText('riskActionBtn', 'Run a scan to get a recommendation');
  setText('riskCheck1', 'Check tyre pressure'); setText('riskCheck2', 'Maintain safe following distance');
  setText('safetyScoreNum', '—'); setText('safetyScoreRiskLabel', 'Awaiting data'); setColor('safetyScoreRiskLabel', 'var(--dim)');
  const scoreArcEl = $('safetyScoreArc'); if (scoreArcEl) scoreArcEl.setAttribute('stroke-dasharray', '0 100');
  const stepperEl = $('historyStepper'); if (stepperEl) stepperEl.innerHTML = '<div class="step"><span class="pt"></span><span class="lbl">—</span></div>';
});

/* ---- synthetic test-feed frames ---- */
function makeSyntheticTrackCanvas(kind) {
  const W = 240, H = 160;
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#7d7d7d'; ctx.fillRect(0, 0, W, H);
  const speckleCount = kind === 'wet' ? 40 : 500;
  for (let i = 0; i < speckleCount; i++) {
    const x = Math.random() * W, y = Math.random() * H;
    const shade = 100 + Math.random() * 80;
    ctx.fillStyle = `rgb(${shade},${shade},${shade})`; ctx.fillRect(x, y, 2, 2);
  }
  function glareBlob(cx, cy, r, alpha) {
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    grad.addColorStop(0, `rgba(235,235,235,${alpha})`); grad.addColorStop(1, `rgba(235,235,235,0)`);
    ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
  }
  if (kind === 'dry') { /* no glare */ }
  else if (kind === 'damp') {
    ctx.fillStyle = 'rgba(0,0,0,0.08)'; ctx.fillRect(0, 0, W, H);
    glareBlob(60, 40, 130, 0.35); glareBlob(180, 60, 130, 0.35); glareBlob(60, 120, 130, 0.35); glareBlob(180, 120, 130, 0.35);
  } else if (kind === 'drying') {
    ctx.fillStyle = 'rgba(0,0,0,0.1)'; ctx.fillRect(0, 0, W / 2, H);
    glareBlob(60, 80, 80, 0.9);
  } else if (kind === 'wet') {
    ctx.fillStyle = 'rgba(0,0,0,0.15)'; ctx.fillRect(0, 0, W, H);
    glareBlob(120, 80, 140, 0.9); glareBlob(50, 130, 100, 0.85); glareBlob(190, 40, 90, 0.85);
  }
  return canvas;
}

const DEMO_SEQUENCE = ['dry', 'damp', 'drying', 'wet', 'drying', 'damp', 'dry'];
const DEMO_LAP_LABELS = ['Lap 1', 'Lap 4', 'Lap 6', 'Lap 8', 'Lap 11', 'Lap 14', 'Lap 17'];

if (demoBtn) demoBtn.addEventListener('click', async () => {
  historyLog = []; frameCount = 0;
  demoBtn.disabled = true; setText('engineTag', 'DEMO SEQUENCE');
  for (let i = 0; i < DEMO_SEQUENCE.length; i++) {
    const canvas = makeSyntheticTrackCanvas(DEMO_SEQUENCE[i]);
    const result = analyzeFrame(canvas);
    historyLog.push({ label: result.label, wetness_score: result.wetness_score });
    const { trend } = trendFor(historyLog);
    const safety = computeSafetyClientSide(result.label, trend, result.wetness_score, readVehicleTyreForm());
    const url = canvas.toDataURL();
    if (dzThumb) dzThumb.src = url;
    if (scanImg) { scanImg.src = url; scanImg.style.display = 'block'; }
    addClass('dropzone', 'has-file'); addClass('scanFrame', 'has-file');
    render({ ...result, trend, suggestion: suggestionFor(result.label, trend), history: historyLog, safety, _lapName: DEMO_LAP_LABELS[i] });
    await new Promise(r => setTimeout(r, 900));
  }
  demoBtn.disabled = false;
});

/* ================================ RENDER (null-guarded) ================================ */
function pct(x) { return `${Math.round(x * 100)}%`; }

function flashPanels() {
  document.querySelectorAll('.hud-panel').forEach(p => {
    p.classList.add('flash');
    setTimeout(() => p.classList.remove('flash'), 700);
  });
}

function render(data, opts = {}) {
  if (!opts.skipPersist) {
    frameCount++;
    STORE.save({ latest: data, history: data.history || historyLog, frameCount });
  }
  flashPanels();
  setText('frameCounter', '#' + String(frameCount).padStart(4, '0'));
  setText('sysFrames', String(frameCount));
  setText('sysEngine', (data.engine || '—').toUpperCase());
  setText('inferenceMs', data.inference_ms != null ? `inference ${data.inference_ms}ms` : 'inference —');

  const color = COLORS[data.label] || '#888';

  setText('stateLabel', data.label.toUpperCase()); setColor('stateLabel', color);
  const isModel = data.engine && data.engine.includes('resnet');
  const confVal = isModel ? data.confidence : data.wetness_score;
  setText('stateConf', pct(confVal ?? 0));
  setText('stateConfLabel', isModel ? 'confidence' : 'signal');

  setText('condLabel', data.label.toLowerCase());
  setBg('condDot', color);

  setText('probEngine', isModel ? 'RESNET18' : 'HEURISTIC');
  const probs = data.probabilities || { Dry: 0, Damp: 0, Drying: 0, Wet: 0 };
  ['Dry', 'Damp', 'Drying', 'Wet'].forEach(k => {
    setWidth('prob' + k, pct(probs[k] || 0));
    setText('prob' + k + 'V', pct(probs[k] || 0));
  });

  const gripPct = Math.round((1 - data.wetness_score) * 100);
  const gripLabel = gripPct >= 70 ? 'High' : gripPct >= 40 ? 'Medium' : 'Low';
  setText('gripLevel', `${gripLabel} (${gripPct / 100})`);
  setText('wetnessV', pct(data.wetness_score));
  setText('patchV', pct(data.patchiness_score || 0));
  setText('trendV', data.trend || '—');
  setLeft('moistMarker', pct(data.wetness_score));

  display('scanTags', 'flex');
  addClass('scanFrame', 'locked');
  setText('tagSurface', data.label.toUpperCase());
  setText('tagGrip', `${(1 - data.wetness_score).toFixed(2)} [${data.trend === 'Getting wetter' ? 'DEGRADING' : data.trend === 'Drying out' ? 'IMPROVING' : 'STABLE'}]`);

  setText('suggestionText', '💡 ' + data.suggestion);

  const safety = data.safety;
  if (safety) {
    const rl = safety.risk_level;
    setClass('riskBadgeHud', `risk-badge risk-${rl}`);
    setText('riskBadgeHud', `${rl} risk`);
    setClass('riskBadgeSafety', `risk-badge risk-${rl}`);
    setText('riskBadgeSafety', `${rl} risk`);
    setText('maxSpeedV', `${safety.recommended_max_speed_kmh} km/h`);

    const form = readVehicleTyreForm();
    const speed = form.speed !== '' ? Number(form.speed) : null;
    setText('speedReadout', speed !== null ? speed : '—');
    const speedPct = speed !== null ? Math.min((speed / safety.recommended_max_speed_kmh) * 100, 130) : 0;
    const sr = $('speedRing'); if (sr) sr.style.setProperty('--pct', speedPct);

    const pressure = form.tyre_pressure !== '' ? Number(form.tyre_pressure) : null;
    setText('tyreReading', pressure !== null ? `${pressure} PSI` : '— PSI');
    setText('tyreSub', form.tyre_temperature !== '' ? `${form.tyre_temperature}°C · ${form.tyre_condition || 'condition n/a'}` : (form.tyre_condition || 'tyre not reported'));

    setHTML('safetyList', `
      <li><b>Road</b><span>${safety.road_condition_warning}</span></li>
      <li><b>Speed</b><span>${safety.speed_related_warning}</span></li>
      <li><b>Tyre pressure</b><span>${safety.tyre_pressure_warning}</span></li>
      <li><b>Tyre inspect</b><span>${safety.tyre_inspection_recommendation}</span></li>
      <li><b>Recommendation</b><span>${safety.general_safety_recommendation}</span></li>
    `);
    setHTML('waterBanner', safety.standing_water_alert
      ? `<div class="water-banner">💧 STANDING WATER ALERT — aquaplaning risk is elevated on this stretch.</div>` : '');
  }

  const history = data.history || [];
  drawTrend(history);
  const histRows = history.slice(-10).reverse().map((h, i) =>
    `<div><span>#${history.length - i}</span><b style="color:${COLORS[h.label]}">${h.label}</b><span>${pct(h.wetness_score)}</span></div>`
  ).join('');
  setHTML('logList', histRows || '<div style="padding:8px 0;color:var(--dim-2)">— no frames yet —</div>');

  renderV2(data, safety, opts);
}

/* ---- v2 mockup elements (Overview page only — everything here is
   null-guarded so it's a no-op on the other pages) ---- */
function renderV2(data, safety, opts) {
  setText('condPillTop', data.label);

  const gripPct = Math.round((1 - data.wetness_score) * 100);
  const band = gripPct >= 80 ? 'Good' : gripPct >= 50 ? 'Regular' : 'Poor';
  const bandColor = { Good: '#22c55e', Regular: '#f59e0b', Poor: '#ef4444' }[band];
  setText('gaugeScore', gripPct);
  setText('gaugeBand', band.toUpperCase()); setColor('gaugeBand', bandColor);
  const descByBand = {
    Good: 'Surface looks solid — good grip across the scanned patch.',
    Regular: 'Some surface irregularities detected.',
    Poor: 'Significant surface risk detected — reduce speed.',
  };
  setText('gaugeDesc', descByBand[band]);
  const arc = $('gaugeArc');
  if (arc) { arc.setAttribute('stroke-dasharray', `${gripPct} ${100 - gripPct}`); arc.style.stroke = bandColor; }
  ['legendGood', 'legendRegular', 'legendPoor'].forEach(id => removeClass(id, 'on'));
  addClass('legend' + band, 'on');

  // reveal the sensor-feed overlay once a frame has been analyzed
  display('gridOverlay', 'block'); display('heatPatch', 'block');
  display('feedTopLeft', 'block'); display('feedDetected', 'block'); display('feedScanBadge', 'flex');
  display('miniTrendSvg', 'block');
  display('feedEmptyHint', 'none');
  const scanImgEl = $('scanImg'); if (scanImgEl && scanImgEl.src) scanImgEl.style.display = 'block';

  setText('detSurface', data.label);
  setText('detPotholes', '0');
  setText('detCracks', `${Math.round((data.patchiness_score || 0) * 5)} minor`);
  setText('detWater', data.wetness_score >= 0.6 ? 'High' : data.wetness_score >= 0.3 ? 'Medium' : 'Low');

  const isModel = data.engine && data.engine.includes('resnet');
  const scanPct = Math.round((isModel ? data.confidence : data.wetness_score) * 100 || 0);
  setWidth('scanBarFill', scanPct + '%');
  setText('scanPctLabel', scanPct + '%');
  drawMiniTrend(data.history || historyLog);

  if (safety) {
    const form = readVehicleTyreForm();
    const speed = form.speed !== '' ? Number(form.speed) : null;
    const speedPct = speed !== null ? Math.min((speed / safety.recommended_max_speed_kmh) * 100, 100) : 0;
    const sArc = $('speedArc');
    if (sArc) sArc.setAttribute('stroke-dasharray', `${speedPct} ${100 - speedPct}`);
    setText('rpmVal', speed !== null ? Math.round(speed * 32) : '—');
    setText('tyreActionLabel', `Recommended Action`);

    const pressure = form.tyre_pressure !== '' ? `${form.tyre_pressure} PSI` : '— PSI';
    ['psiFL', 'psiFR', 'psiRL', 'psiRR'].forEach(id => setText(id, pressure));

    const rl = safety.risk_level;
    const pw = safety.tyre_pressure_warning || '';
    let pressureLabel = 'Normal', pressureGood = true;
    if (pw.includes('Critically')) { pressureLabel = 'Critical'; pressureGood = false; }
    else if (pw.includes('is low')) { pressureLabel = 'Low'; pressureGood = false; }
    else if (pw.includes('is high')) { pressureLabel = 'High'; pressureGood = false; }
    else if (pw.includes('not reported')) { pressureLabel = 'Unknown'; pressureGood = false; }
    setText('insightPressure', pressureLabel);
    setClass('insightPressureIc', `ic ${pressureGood ? 'good' : 'warn'}`);
    setText('insightPressureIc', pressureGood ? '✅' : '⚠️');

    const riskLabel = rl.charAt(0).toUpperCase() + rl.slice(1);
    setText('insightRisk', riskLabel);
    const riskGood = rl === 'low';
    setClass('insightRiskIc', `ic ${riskGood ? 'good' : 'warn'}`);
    setText('insightRiskIc', riskGood ? '✅' : '⚠️');

    const safetyLabel = { low: 'Good', medium: 'Caution', high: 'Poor', critical: 'Critical' }[rl] || 'Good';
    setText('insightSafety', safetyLabel);
    setClass('insightSafetyIc', `ic ${rl === 'low' ? 'good' : 'warn'}`);

    const alertLabel = rl === 'low' ? 'None expected' : rl === 'medium' ? 'Check tyres soon' : 'Act now';
    setText('insightAlert', alertLabel);
    setClass('insightAlertIc', `ic ${rl === 'low' ? 'good' : 'warn'}`);

    /* ---- 4-up stat row ---- */
    const speedOver = speed !== null && speed > safety.recommended_max_speed_kmh;
    setClass('speedSub', `sub ${speedOver ? 'bad' : 'good'}`);
    setText('speedSub', speedOver ? 'Too fast' : 'Normal');

    setText('statTyrePressure', form.tyre_pressure !== '' ? `F ${form.tyre_pressure} · R ${form.tyre_pressure} PSI` : 'Not reported');
    setClass('tyrePressureSub', `sub ${pressureGood ? 'good' : 'bad'}`);
    setText('tyrePressureSub', pressureLabel);

    const recPsiRaw = form.tyre_recommended_pressure !== '' ? Number(form.tyre_recommended_pressure) : parseFloat($('psiFL') ? $('psiFL').textContent : NaN);
    let tyreHealth = '—';
    if (form.tyre_pressure !== '' && !isNaN(recPsiRaw)) {
      const dev = Math.abs(Number(form.tyre_pressure) - recPsiRaw);
      tyreHealth = Math.max(0, Math.round(100 - dev * 8));
    }
    setText('statTyreHealth', tyreHealth === '—' ? '—' : `${tyreHealth}%`);
    const healthGood = tyreHealth === '—' || tyreHealth >= 80;
    setClass('tyreHealthSub', `sub ${healthGood ? 'good' : tyreHealth >= 50 ? 'warn' : 'bad'}`);
    setText('tyreHealthSub', tyreHealth === '—' ? 'Unknown' : healthGood ? 'Good' : tyreHealth >= 50 ? 'Fair' : 'Poor');

    /* ---- AI Safety Recommendation card ---- */
    const riskCardEl = $('riskCard');
    if (riskCardEl) {
      riskCardEl.classList.remove('risk-low', 'risk-medium');
      if (rl === 'low') riskCardEl.classList.add('risk-low');
      else if (rl === 'medium') riskCardEl.classList.add('risk-medium');
    }
    const riskIcon = { low: '✅', medium: 'ℹ️', high: '⚠️', critical: '🛑' }[rl] || '⚠️';
    setText('riskLevelTxt', `${riskLabel.toUpperCase()} RISK`);
    setText('riskLevelIc', riskIcon);
    setText('riskSubTxt', `${data.label} road detected · Speed: ${speed !== null ? speed + ' km/h' : 'not reported'}`);
    if (speedOver) {
      setText('riskActionBtn', `Reduce speed to < ${safety.recommended_max_speed_kmh} km/h`);
    } else if (rl === 'low') {
      setText('riskActionBtn', 'Conditions look fine — maintain safe speed');
    } else {
      setText('riskActionBtn', 'Stay alert and monitor conditions');
    }
    setText('riskCheck1', pressureGood ? 'Tyre pressure looks fine' : 'Check tyre pressure');
    setText('riskCheck2', 'Maintain safe following distance');

    /* ---- Overall Safety Score donut ---- */
    const penalty = { low: 8, medium: 25, high: 45, critical: 70 }[rl] || 8;
    const safetyScore = Math.max(0, Math.min(100, 100 - penalty));
    const scoreColor = { low: '#22c55e', medium: '#f59e0b', high: '#ef4444', critical: '#ef4444' }[rl] || '#22c55e';
    setText('safetyScoreNum', safetyScore);
    const scoreArcEl = $('safetyScoreArc');
    if (scoreArcEl) { scoreArcEl.setAttribute('stroke-dasharray', `${safetyScore} ${100 - safetyScore}`); scoreArcEl.style.stroke = scoreColor; }
    setText('safetyScoreRiskLabel', `${riskLabel} Risk`);
    setColor('safetyScoreRiskLabel', scoreColor);
  }

  /* ---- Road Condition History stepper ---- */
  const stepperEl = $('historyStepper');
  if (stepperEl) {
    const hist = (data.history || historyLog).slice(-4);
    if (hist.length) {
      stepperEl.innerHTML = hist.map((h, i) => {
        const isLast = i === hist.length - 1;
        return `<div class="step ${isLast ? 'active' : 'done'}"><span class="pt" style="${isLast ? '' : `background:${COLORS[h.label]};border-color:${COLORS[h.label]}`}"></span><span class="lbl" style="${isLast ? `color:${COLORS[h.label]}` : ''}">${h.label}</span></div>`;
      }).join('');
    }
  }

  if (!opts.skipPersist) {
    const activity = STORE.load().activity || [];
    activity.unshift({ text: `Road Condition: ${data.label}`, color: COLORS[data.label] || '#888', ts: Date.now() });
    const capped = activity.slice(0, 6);
    STORE.save({ activity: capped });
    renderActivity(capped);
  } else {
    renderActivity(STORE.load().activity || []);
  }
}

function drawMiniTrend(history) {
  const svg = $('miniTrendSvg');
  if (!svg) return;
  const pts = history.slice(-8);
  if (pts.length < 2) { svg.innerHTML = ''; return; }
  const w = 110, h = 56, pad = 6;
  const stepX = (w - 2 * pad) / (pts.length - 1);
  const path = pts.map((p, i) => {
    const x = pad + i * stepX, y = pad + (1 - p.wetness_score) * (h - 2 * pad);
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  svg.innerHTML = `<path d="${path}" fill="none" stroke="#3b82f6" stroke-width="2"/>`;
}

function renderActivity(activity) {
  const el = $('recentAnalysis');
  if (!el) return;
  if (!activity.length) { el.innerHTML = '<div class="v2-log-row"><span class="dot" style="background:#5b6480"></span><span class="txt" style="color:var(--dim)">No analysis yet</span></div>'; return; }
  el.innerHTML = activity.map(a => `
    <div class="v2-log-row">
      <span class="dot" style="background:${a.color}"></span>
      <span class="txt">${a.text}</span>
      <span class="time">${new Date(a.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    </div>`).join('');
}

function drawTrend(history) {
  const canvas = $('trendCanvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 280, h = 120;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const pts = history.slice(-12);
  if (pts.length === 0) return;
  const pad = 14;
  const stepX = pts.length > 1 ? (w - 2 * pad) / (pts.length - 1) : 0;

  ctx.strokeStyle = 'rgba(255,255,255,.07)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = pad + i * (h - 2 * pad) / 3;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
  }

  ctx.beginPath(); ctx.strokeStyle = '#3e7bfa'; ctx.lineWidth = 2;
  pts.forEach((p, i) => {
    const x = pad + i * stepX, y = pad + (1 - p.wetness_score) * (h - 2 * pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  pts.forEach((p, i) => {
    const x = pad + i * stepX, y = pad + (1 - p.wetness_score) * (h - 2 * pad);
    ctx.beginPath(); ctx.fillStyle = COLORS[p.label] || '#888';
    ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#05070a'; ctx.lineWidth = 1.5; ctx.stroke();
  });
}

/* ---- on every page load: repaint from whatever was last saved, so
   Track Analysis / Car Status / Telemetry / System reflect the latest
   analysis run on Overview even though the analyzer only lives there ---- */
(function hydrate() {
  const saved = STORE.load();
  if (saved.latest) {
    render(saved.latest, { skipPersist: true });
    if (saved.latest.label) {
      removeClass('dropzone', 'has-file');
      // keep the scan-frame preview empty on other pages (no image cached) —
      // only textual/graphical readouts are hydrated, which is all a
      // read-only page needs.
    }
  }
})();
