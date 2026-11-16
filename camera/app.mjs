import { PoseMetrics } from './geometry.mjs';

const MODULE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/vision_bundle.mjs';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const FRAME_INTERVAL_MS = 1000 / 15;
const MAX_DIMENSION = 960;
const $ = (id) => document.getElementById(id);
const video = $('source');
const canvas = $('preview');
const context = canvas.getContext('2d', { alpha: false });
const detectorCanvas = document.createElement('canvas');
const detectorContext = detectorCanvas.getContext('2d', { alpha: false });
const metricCells = new Map();
let config = null;
let configSignature = '';
let generation = 0;
let active = false;
let running = false;
let activeStream = null;
let landmarker = null;
let tracker = null;
let animationFrame = 0;
let lastInference = -Infinity;
let lastVideoTime = -1;
let analysisOrigin = null;
let lastAnalysisTime = -1;
let lastHeight = 0;
let renderDisabled = false;

// This bridge carries layout/readiness only, never camera or measurement data.
function bridge(type, payload = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...payload }, '*');
}

function setFrameHeight() {
  const height = Math.ceil($('camera-app').getBoundingClientRect().height);
  if (height !== lastHeight) {
    lastHeight = height;
    bridge('streamlit:setFrameHeight', { height });
  }
}

function status(message, level = 'info') {
  if ($('status').textContent !== message) $('status').textContent = message;
  $('status').dataset.level = level;
}

function controls() {
  $('start').disabled = active || !config || !$('consent').checked || renderDisabled;
  $('stop').disabled = !active;
}

function clearMeasurements() {
  $('measured').textContent = '—';
  $('elapsed').textContent = '—';
  $('count').textContent = '—';
  for (const cell of metricCells.values()) cell.textContent = '—';
  $('calibration').textContent = 'The first 10 reliable frames calibrate torso scale. Keep your whole body visible and the camera fixed and level.';
}

function resetAnalysis() {
  tracker = config ? new PoseMetrics(config) : null;
  analysisOrigin = null;
  lastAnalysisTime = -1;
  clearMeasurements();
}

function releaseModel(model) {
  if (!model) return '';
  try {
    model.close();
    return '';
  } catch (error) {
    return ` Detector cleanup reported an error: ${error.message || String(error)}. Reload before restarting.`;
  }
}

function stop(message = 'Camera stopped. Press Start for a fresh analysis.', level = 'info') {
  ++generation;
  active = false;
  running = false;
  cancelAnimationFrame(animationFrame);
  animationFrame = 0;
  if (activeStream) {
    for (const track of activeStream.getTracks()) track.stop();
    activeStream = null;
  }
  video.pause();
  video.srcObject = null;
  const cleanupError = releaseModel(landmarker);
  landmarker = null;
  tracker = null;
  canvas.hidden = true;
  context.clearRect(0, 0, canvas.width, canvas.height);
  detectorContext.clearRect(0, 0, detectorCanvas.width, detectorCanvas.height);
  clearMeasurements();
  $('reset-notice').hidden = true;
  controls();
  status(message + cleanupError, cleanupError ? 'error' : level);
}

function cameraError(error) {
  switch (error.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera permission was denied or blocked. Use HTTPS, allow camera access in browser site settings, and ensure the embedding page permits camera access. Then press Start again.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No usable camera was found. Connect or enable a camera, then press Start again; video upload remains available.';
    case 'NotReadableError':
    case 'AbortError':
      return 'The camera could not be opened. Close other apps using it, check your device permissions, and press Start again.';
    default:
      return `Camera startup failed: ${error.message || String(error)}. Check camera permission and HTTPS, then press Start again.`;
  }
}

async function start() {
  if (active || !config || !$('consent').checked || renderDisabled) return;
  if (document.hidden) {
    status('Bring this page into view before starting the camera.');
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    status('Camera access requires HTTPS (or localhost) and a browser with camera support. Open the HTTPS app directly or use video upload.', 'error');
    return;
  }
  const session = ++generation;
  const current = () => session === generation && active;
  active = true;
  controls();
  resetAnalysis();
  $('reset-notice').hidden = true;
  status('Waiting for camera permission. Allow video access in the browser prompt; Stop cancels this session.');
  let stage = 'camera';
  try {
    // Called directly from the Start click; never from render or a timer.
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, frameRate: { ideal: 15, max: 15 } },
      audio: false,
    });
    if (!current()) {
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    activeStream = stream;
    for (const track of stream.getVideoTracks()) {
      track.addEventListener('ended', () => {
        if (current()) stop('Camera access ended or was withdrawn. Check permissions and press Start to begin again.');
      }, { once: true });
    }
    video.srcObject = stream;
    await video.play();
    if (!current()) return;
    stage = 'model';
    status('Camera connected. Downloading the official MediaPipe browser runtime…');
    const { FilesetResolver, PoseLandmarker } = await import(MODULE_URL);
    if (!current()) return;
    status('Preparing MediaPipe WebAssembly…');
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    if (!current()) return;
    status('Downloading and initializing the official pose model. This may take a moment on the first start…');
    const model = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.3,
      minPosePresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
      outputSegmentationMasks: false,
    });
    if (!current()) {
      releaseModel(model);
      return;
    }
    landmarker = model;
    resetAnalysis();
    lastInference = -Infinity;
    lastVideoTime = -1;
    running = true;
    status('Camera running. Keep your whole body visible while torso scale calibrates.');
    animationFrame = requestAnimationFrame(inferFrame);
  } catch (error) {
    if (!current()) return;
    const message = stage === 'camera' ? cameraError(error)
      : `The official pose runtime or model could not load: ${error.message || String(error)}. Check connectivity and content blockers for cdn.jsdelivr.net and storage.googleapis.com, then press Start again. Video upload remains available.`;
    stop(message, 'error');
  }
}

function drawOverlay(points, row) {
  if (!points) return;
  const radius = Math.max(3, canvas.width / 180);
  context.strokeStyle = '#44edb4';
  context.fillStyle = '#44edb4';
  context.lineWidth = Math.max(2, canvas.width / 240);
  context.beginPath();
  for (const [a, b] of [['wrist', 'elbow'], ['elbow', 'shoulder'], ['shoulder', 'hip'], ['hip', 'knee'], ['knee', 'ankle']]) {
    context.moveTo(...points[a]);
    context.lineTo(...points[b]);
  }
  context.stroke();
  for (const point of Object.values(points)) {
    context.beginPath();
    context.arc(point[0], point[1], radius, 0, 2 * Math.PI);
    context.fill();
  }
  context.save();
  context.strokeStyle = '#f9d65c';
  context.setLineDash([7, 6]);
  context.beginPath();
  context.moveTo(points.wrist[0], 0);
  context.lineTo(points.wrist[0], canvas.height);
  context.stroke();
  context.restore();
  const fontSize = Math.max(12, canvas.width / 45);
  context.font = `600 ${fontSize}px system-ui, sans-serif`;
  context.textBaseline = 'top';
  for (const [joint, key] of [['elbow', 'elbow_deg'], ['shoulder', 'shoulder_deg'], ['hip', 'hip_deg'], ['knee', 'knee_deg']]) {
    if (!Number.isFinite(row[key])) continue;
    const text = `${row[key].toFixed(0)}°`;
    const textWidth = context.measureText(text).width;
    const x = Math.max(4, Math.min(canvas.width - textWidth - 8, points[joint][0] + radius + 4));
    const y = Math.max(4, Math.min(canvas.height - fontSize - 8, points[joint][1] + radius + 4));
    context.fillStyle = 'rgba(0, 0, 0, .75)';
    context.fillRect(x - 3, y - 2, textWidth + 6, fontSize + 6);
    context.fillStyle = '#ffffff';
    context.fillText(text, x, y);
  }
}

function showMeasurements({ row, summary }) {
  $('measured').textContent = `${(summary.measured_fraction * 100).toFixed(0)}%`;
  $('elapsed').textContent = `${summary.duration_s.toFixed(1)} s`;
  $('count').textContent = config.skill === 'Handstand push-up'
    ? String(summary.full_reps) : `${summary.longest_tracked_posture_s.toFixed(1)} s`;
  for (const [key, cell] of metricCells) {
    cell.textContent = Number.isFinite(row[key]) ? row[key].toFixed(2) : '—';
  }
  $('calibration').textContent = Number.isFinite(summary.torso_reference_px)
    ? `Torso reference: ${summary.torso_reference_px.toFixed(1)} pixels. Camera resolution changes recalibrate scale. Green segments are tracked joints; the yellow line passes through the wrist.`
    : 'Calibrating: first 10 reliable frames establish torso scale. Keep the torso visible and side-on.';
  if (row.status === 'Measured') {
    status('Measured — reliable landmarks match the selected posture. Inspect the overlay before interpreting measurements.');
  } else if (row.status === 'Insufficient visibility') {
    status('Insufficient visibility / low confidence — keep wrists through ankles visible, improve lighting, and check the selected side and detector orientation. Measurements are unavailable.');
  } else if (row.status === 'Outside selected posture') {
    status('Outside selected posture — landmarks are visible but do not match the chosen skill. Posture time and full repetitions do not advance.');
  } else {
    status(`${row.status} — keep your body visible and the camera fixed.`);
  }
}

function inferFrame(timestamp) {
  if (!running) return;
  // Synchronous inference, with at most one scheduled frame: no growing queue.
  if (timestamp - lastInference >= FRAME_INTERVAL_MS && video.readyState >= 2
      && video.videoWidth > 0 && video.videoHeight > 0 && video.currentTime !== lastVideoTime) {
    try {
      lastInference = timestamp;
      lastVideoTime = video.currentTime;
      const scale = Math.min(1, MAX_DIMENSION / Math.max(video.videoWidth, video.videoHeight));
      const width = Math.max(2, Math.round(video.videoWidth * scale));
      const height = Math.max(2, Math.round(video.videoHeight * scale));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = detectorCanvas.width = width;
        canvas.height = detectorCanvas.height = height;
      }
      canvas.hidden = false;
      context.drawImage(video, 0, 0, width, height);
      detectorContext.save();
      if (config.detector_inverted) {
        detectorContext.translate(width, height);
        detectorContext.rotate(Math.PI);
      }
      detectorContext.drawImage(canvas, 0, 0);
      detectorContext.restore();
      const result = landmarker.detectForVideo(detectorCanvas, timestamp);
      if (analysisOrigin === null) analysisOrigin = timestamp;
      const timeSeconds = Math.max((timestamp - analysisOrigin) / 1000, lastAnalysisTime + 0.000001);
      lastAnalysisTime = timeSeconds;
      const measurement = tracker.update(result.landmarks?.[0] ?? null, width, height, timeSeconds);
      drawOverlay(measurement.points, measurement.row);
      showMeasurements(measurement);
    } catch (error) {
      stop(`Live analysis failed: ${error.message || String(error)}. Camera stopped; press Start to retry or use video upload.`, 'error');
      return;
    }
  }
  if (running) animationFrame = requestAnimationFrame(inferFrame);
}

function applyTheme(theme) {
  if (!theme) return;
  const style = document.documentElement.style;
  for (const [variable, key] of [['--background', 'backgroundColor'], ['--secondary', 'secondaryBackgroundColor'], ['--text', 'textColor'], ['--primary', 'primaryColor']]) {
    if (typeof theme[key] === 'string') style.setProperty(variable, theme[key]);
  }
  style.colorScheme = theme.base === 'dark' ? 'dark' : 'light';
  if (typeof theme.font === 'string') document.body.style.fontFamily = theme.font;
}

window.addEventListener('message', (event) => {
  if (event.source !== window.parent || event.data?.type !== 'streamlit:render') return;
  const { args, theme, disabled } = event.data;
  applyTheme(theme);
  renderDisabled = Boolean(disabled);
  if (args?.config) {
    const signature = JSON.stringify(args.config);
    const changed = signature !== configSignature;
    config = { ...args.config };
    configSignature = signature;
    if (args.metrics) {
      $('metrics').replaceChildren();
      metricCells.clear();
      for (const [key, label] of Object.entries(args.metrics)) {
        const row = document.createElement('tr');
        const name = document.createElement('th');
        name.scope = 'row';
        name.textContent = label;
        const value = document.createElement('td');
        value.textContent = '—';
        row.append(name, value);
        $('metrics').append(row);
        metricCells.set(key, value);
      }
    }
    $('count-label').textContent = config.skill === 'Handstand push-up' ? 'Full repetitions' : 'Longest tracked posture';
    $('variation').hidden = config.skill !== 'Planche';
    $('variation').textContent = `Selected planche variation: ${config.variant}. Tuck shapes are not penalized for bent hips or knees. Side-view video cannot measure straddle width; variation is your selection, not an automatic classification.`;
    if (changed) {
      resetAnalysis();
      if (running) {
        canvas.hidden = true;
        $('reset-notice').textContent = 'Analysis settings changed. Calibration, elapsed time, posture duration, and repetitions were reset; the camera remains on.';
        $('reset-notice').hidden = false;
        status('Settings changed: calibration, elapsed time, and counts reset. Camera remains on; analyzing with the new settings.');
      } else if (!active) {
        status('Camera is off. Consent above and press Start to begin on-device analysis.');
      }
    }
  }
  controls();
  setFrameHeight();
});

$('start').addEventListener('click', start);
$('stop').addEventListener('click', () => stop());
$('consent').addEventListener('change', () => {
  if (!$('consent').checked) stop('Consent withdrawn. Camera stopped; no automatic restart.');
  else controls();
});
video.addEventListener('error', () => {
  if (active) stop('The browser could not read the camera video. Check your camera and permissions, then press Start again.', 'error');
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && active) stop('Camera stopped because the page became hidden. Press Start to begin a new session.');
});
window.addEventListener('pagehide', () => stop('Camera stopped because the page was closed or left.'));
window.addEventListener('beforeunload', () => stop('Camera stopped.'));
new ResizeObserver(setFrameHeight).observe($('camera-app'));
bridge('streamlit:componentReady', { apiVersion: 1 });
setFrameHeight();
