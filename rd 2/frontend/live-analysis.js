/* =========================================================================
   ROADDOC — Live Analysis page (video upload + live camera).
   Uses the same API_BASE / $ / setText helpers and client-side heuristic
   classifier already defined in app.js (loaded before this file).
   ========================================================================= */

/* ---------------------------------- video upload mode ---------------------------------- */
let selectedVideoFile = null;

const videoInput = $('videoInput');
const videoDropzone = $('videoDropzone');
const videoPreview = $('videoPreview');
const analyzeVideoBtn = $('analyzeVideoBtn');

function handleVideoFile(file) {
  if (!file) return;
  selectedVideoFile = file;
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
  videoPreview.style.display = 'block';
  addClass('videoDropzone', 'has-file');
  setText('videoFileName', file.name);
  setText('videoFileMeta', `${(file.size / (1024 * 1024)).toFixed(1)} MB`);
  analyzeVideoBtn.disabled = false;
  setVideoStatus('ready', 'Ready to analyze');
}
if (videoInput) videoInput.addEventListener('change', e => handleVideoFile(e.target.files[0]));
if (videoDropzone) {
  videoDropzone.addEventListener('dragover', e => e.preventDefault());
  videoDropzone.addEventListener('drop', e => {
    e.preventDefault();
    if (e.dataTransfer.files.length) handleVideoFile(e.dataTransfer.files[0]);
  });
}

function setVideoStatus(mode, text) {
  const el = $('videoStatusLine');
  if (!el) return;
  el.classList.remove('busy', 'offline');
  if (mode === 'busy') el.classList.add('busy');
  if (mode === 'error') el.classList.add('offline');
  el.querySelector('span').textContent = text;
}

if (analyzeVideoBtn) analyzeVideoBtn.addEventListener('click', async () => {
  if (!selectedVideoFile) return;
  analyzeVideoBtn.disabled = true;
  setVideoStatus('busy', 'Uploading and analyzing — this runs real computer vision server-side, give it a few seconds…');

  const form = new FormData();
  form.append('video', selectedVideoFile);
  form.append('vehicle_type', $('videoVehicleType').value);
  form.append('driving_condition', $('videoDrivingCondition').value);

  try {
    const res = await fetch(`${API_BASE}/api/analyze-video`, { method: 'POST', body: form });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.error || `Backend returned ${res.status}`);
    }
    const result = await res.json();
    renderVideoResult(result);
    setVideoStatus('ready', `Done in ${(result.processing_ms / 1000).toFixed(1)}s — ${result.frames_analyzed} segments analyzed automatically`);
  } catch (e) {
    setVideoStatus('error', `Video analysis needs the Flask backend running (${API_BASE}) — ${e.message}`);
  } finally {
    analyzeVideoBtn.disabled = false;
  }
});

function renderVideoResult(result) {
  display('videoSummary', 'block');
  display('videoEmptyState', 'none');
  display('videoTimelineWrap', 'block');

  setText('sumDuration', `${result.duration_s}s`);
  setText('sumFrames', result.frames_analyzed);

  const speeds = result.timeline.map(s => s.speed_kmh).filter(v => v != null);
  const avgSpeed = speeds.length ? Math.round(speeds.reduce((a, b) => a + b, 0) / speeds.length) : null;
  setText('sumSpeed', avgSpeed != null ? `${avgSpeed} km/h` : 'n/a');

  const totals = result.defect_totals || { pothole: 0, crack: 0 };
  setText('sumDefects', `${totals.pothole || 0} / ${totals.crack || 0}`);

  const timelineEl = $('videoTimeline');
  timelineEl.innerHTML = result.timeline.map((seg, i) => {
    const color = COLORS[seg.road_condition] || '#888';
    const spd = seg.speed_kmh != null ? `${seg.speed_kmh.toFixed(0)} km/h` : '—';
    const spdTag = seg.speed_kmh != null ? (seg.speed_calibrated ? '' : ' ~') : '';
    const defCount = seg.defects.length;
    const rl = seg.safety.risk_level;
    return `
      <div class="timeline-row" data-t="${seg.t}">
        <span class="t">${seg.t.toFixed(1)}s</span>
        <span class="cond" style="color:${color}">${seg.road_condition}</span>
        <span class="defects ${defCount ? '' : 'none'}">${defCount ? `${defCount} detection(s)` : 'no defects seen'}</span>
        <span class="spd">${spd}${spdTag}</span>
        <span>${seg.recommended_tyre_psi} PSI</span>
        <span class="risk-badge risk-${rl}" style="font-size:.6rem;padding:3px 8px">${rl}</span>
      </div>`;
  }).join('');

  timelineEl.querySelectorAll('.timeline-row').forEach(row => {
    row.addEventListener('click', () => {
      videoPreview.currentTime = parseFloat(row.dataset.t);
      videoPreview.play();
    });
  });

  drawVideoSpeedChart(result.timeline);
}

function drawVideoSpeedChart(timeline) {
  const canvas = $('videoSpeedChart');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300, h = 90;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const pts = timeline.filter(s => s.speed_kmh != null);
  if (pts.length < 2) return;
  const pad = 10;
  const maxSpeed = Math.max(...pts.map(p => p.speed_kmh), 20);
  const stepX = (w - 2 * pad) / (pts.length - 1);

  ctx.strokeStyle = 'rgba(255,255,255,.08)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 2; i++) {
    const y = pad + i * (h - 2 * pad) / 2;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
  }

  ctx.beginPath(); ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 2;
  pts.forEach((p, i) => {
    const x = pad + i * stepX, y = pad + (1 - p.speed_kmh / maxSpeed) * (h - 2 * pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = 'var(--v2-dim)';
  ctx.font = '10px Inter';
  ctx.fillText(`speed (km/h), auto-estimated — peak ${Math.round(maxSpeed)}`, pad, h - 2);
}

/* ---------------------------------- live camera mode ---------------------------------- */
const RECOMMEND_BASE = { car: 32, suv: 35, bike: 28, bus: 65, truck: 80 };

function recommendTyrePressureClient(vehicleType, speedKmh, waterAlert) {
  const base = RECOMMEND_BASE[vehicleType] || 32;
  let speedBump = 0;
  if (speedKmh != null) {
    if (speedKmh > 100) speedBump = 2;
    else if (speedKmh > 80) speedBump = 1;
  }
  const waterBump = waterAlert ? 1 : 0;
  return base + speedBump + waterBump;
}

let liveStream = null;
let liveTimer = null;
let liveWatchId = null;
let liveGpsSpeedKmh = null;
let liveHistory = [];

const startLiveBtn = $('startLiveBtn');
const stopLiveBtn = $('stopLiveBtn');
const liveVideo = $('liveVideo');
const liveCanvas = $('liveCanvas');

function setLiveStatus(mode, text) {
  const el = $('liveStatusLine');
  if (!el) return;
  el.classList.remove('busy', 'offline');
  if (mode === 'busy') el.classList.add('busy');
  if (mode === 'offline') el.classList.add('offline');
  el.querySelector('span').textContent = text;
}

async function startLiveCamera() {
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' } }, audio: false,
    });
  } catch (e) {
    setLiveStatus('offline', `Camera access failed: ${e.message}`);
    return;
  }
  liveVideo.srcObject = liveStream;
  startLiveBtn.style.display = 'none';
  stopLiveBtn.style.display = '';
  display('liveVideoPlaceholder', 'none');
  display('liveRecBadge', 'flex');
  setLiveStatus('busy', 'Camera live — analyzing road condition every ~1.5s');

  if (navigator.geolocation) {
    liveWatchId = navigator.geolocation.watchPosition(
      pos => { liveGpsSpeedKmh = pos.coords.speed != null ? Math.max(0, pos.coords.speed * 3.6) : null; },
      () => { liveGpsSpeedKmh = null; },
      { enableHighAccuracy: true, maximumAge: 2000 }
    );
  } else {
    setText('liveSpeedVal', 'GPS not supported');
  }

  liveHistory = [];
  liveTimer = setInterval(runLiveTick, 1500);
}

function stopLiveCamera() {
  if (liveTimer) clearInterval(liveTimer);
  if (liveStream) liveStream.getTracks().forEach(t => t.stop());
  if (liveWatchId != null && navigator.geolocation) navigator.geolocation.clearWatch(liveWatchId);
  liveStream = null; liveTimer = null; liveWatchId = null;
  startLiveBtn.style.display = '';
  stopLiveBtn.style.display = 'none';
  display('liveVideoPlaceholder', 'flex');
  display('liveRecBadge', 'none');
  setLiveStatus('ready', 'Camera stopped');
}

function runLiveTick() {
  if (!liveVideo.videoWidth) return;
  const result = analyzeFrame(liveVideo);
  liveHistory.push({ label: result.label, wetness_score: result.wetness_score });
  const { trend } = trendFor(liveHistory);
  const waterAlert = result.wetness_score >= 0.88;
  const speed = liveGpsSpeedKmh;
  const recPsi = recommendTyrePressureClient('car', speed, waterAlert);

  const form = {
    vehicle_type: 'car', driving_condition: 'normal',
    speed: speed != null ? speed : '', tyre_condition: '', tyre_pressure: '',
    tyre_recommended_pressure: recPsi, tyre_temperature: '', location: '',
  };
  const safety = computeSafetyClientSide(result.label, trend, result.wetness_score, form);

  setText('liveCondVal', result.label);
  setColor('liveCondVal', COLORS[result.label] || '#fff');
  setText('liveSpeedVal', speed != null ? `${speed.toFixed(0)} km/h` : 'no GPS fix');
  setText('livePsiVal', `${recPsi} PSI`);
  setClass('liveRiskVal', `val risk-badge risk-${safety.risk_level}`);
  setText('liveRiskVal', `${safety.risk_level} risk`);
  setLiveStatus('busy', `Live — last read ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`);
}

if (startLiveBtn) startLiveBtn.addEventListener('click', startLiveCamera);
if (stopLiveBtn) stopLiveBtn.addEventListener('click', stopLiveCamera);
