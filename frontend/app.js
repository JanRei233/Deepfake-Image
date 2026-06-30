// ============================================================
//  DEEPFAKE DETECTOR — app.js
// ============================================================

const API_URL = "https://deepfake-image-2w5s.onrender.com/predict";

// ── DOM refs ──────────────────────────────────────────────
const dropzone       = document.getElementById("dropzone");
const fileInput      = document.getElementById("fileInput");
const previewPanel   = document.getElementById("previewPanel");
const previewImg     = document.getElementById("preview-img");
const previewFrame   = document.getElementById("previewFrame");
const scanLine       = document.getElementById("scanLine");
const fileNameEl     = document.getElementById("fileName");
const fileSizeEl     = document.getElementById("fileSize");
const detectBtn      = document.getElementById("detectBtn");
const resultPanel    = document.getElementById("resultPanel");
const verdictBadge   = document.getElementById("verdictBadge");
const verdictIcon    = document.getElementById("verdictIcon");
const verdictTitle   = document.getElementById("verdictTitle");
const verdictSubtitle= document.getElementById("verdictSubtitle");
const meterFill      = document.getElementById("meterFill");
const confidenceVal  = document.getElementById("confidenceVal");
const detailStatus   = document.getElementById("detailStatus");
const detailModel    = document.getElementById("detailModel");
const detailTime     = document.getElementById("detailTime");
const detailFileName = document.getElementById("detailFileName");
const toast          = document.getElementById("toast");

let selectedFile = null;
let analysisStartTime = null;

// ── Drag & Drop wiring ────────────────────────────────────
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dz-active");
});

dropzone.addEventListener("dragleave", (e) => {
  // Only fire when leaving the dropzone itself (not a child)
  if (!dropzone.contains(e.relatedTarget)) {
    dropzone.classList.remove("dz-active");
  }
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dz-active");
  const files = e.dataTransfer.files;
  if (files.length > 0 && files[0].type.startsWith("image/")) {
    handleFile(files[0]);
  } else {
    showToast("⚠️  Please drop a valid image file.");
  }
});

// ── File input change ─────────────────────────────────────
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

// ── Change image button ───────────────────────────────────
document.getElementById("changeBtn").addEventListener("click", () => {
  fileInput.value = "";
  selectedFile = null;
  previewPanel.classList.remove("visible");
  resultPanel.classList.remove("visible");
  meterFill.style.width = "0";
});

// ── Detect button ─────────────────────────────────────────
detectBtn.addEventListener("click", detectDeepfake);

// ── Handle selected file ──────────────────────────────────
function handleFile(file) {
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewPanel.classList.add("visible");
    resultPanel.classList.remove("visible");
    meterFill.style.width = "0";
    meterFill.className = "meter-fill";
    confidenceVal.className = "confidence-value";
  };
  reader.readAsDataURL(file);

  // Meta
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = formatSize(file.size);
}

// ── Core detection logic ──────────────────────────────────
async function detectDeepfake() {
  if (!selectedFile) {
    showToast("📂  Please upload an image first.");
    return;
  }

  // Loading state
  setLoadingState(true);
  analysisStartTime = performance.now();

  // Scan animation
  previewFrame.classList.add("scanning");

  // Hide old result
  resultPanel.classList.remove("visible");
  meterFill.style.width = "0";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error || JSON.stringify(data.detail));
    }

    const elapsed = ((performance.now() - analysisStartTime) / 1000).toFixed(2);
    renderResult(data, elapsed);

  } catch (err) {
    renderError(err.message);
  } finally {
    setLoadingState(false);
    previewFrame.classList.remove("scanning");
  }
}

// ── Render verdict ────────────────────────────────────────
function renderResult(data, elapsed) {
  const confidence = data.confidence; // 0–1
  const pct = (confidence * 100).toFixed(1);
  const isDeepfake = data.is_deepfake;

  // Verdict badge
  verdictBadge.className = "verdict-badge " + (isDeepfake ? "deepfake" : "real");
  verdictIcon.textContent = isDeepfake ? "🚨" : "✅";
  verdictTitle.textContent = isDeepfake ? "Deepfake Detected" : "Authentic Image";
  verdictSubtitle.textContent = isDeepfake
    ? "This image shows signs of AI manipulation."
    : "No manipulation artifacts detected.";

  // Confidence meter
  const cls = isDeepfake ? "deepfake" : "real";
  meterFill.className = "meter-fill " + cls;
  confidenceVal.className = "confidence-value " + cls;

  // Animate after short delay (allows CSS transition to fire)
  requestAnimationFrame(() => {
    setTimeout(() => {
      meterFill.style.width = pct + "%";
    }, 80);
  });

  // Counter-up animation for percentage
  animateCounter(confidenceVal, 0, parseFloat(pct), 1100);

  // Detail grid
  detailStatus.textContent   = isDeepfake ? "MANIPULATED" : "AUTHENTIC";
  detailStatus.style.color   = isDeepfake ? "var(--rose-400)" : "var(--emerald-400)";
  detailModel.textContent    = "EfficientNet-B4";
  detailTime.textContent     = elapsed + "s";
  detailFileName.textContent = truncate(selectedFile.name, 20);

  // Show panel
  resultPanel.classList.add("visible");
  resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderError(message) {
  verdictBadge.className = "verdict-badge error";
  verdictIcon.textContent = "⚠️";
  verdictTitle.textContent = "Analysis Failed";
  verdictTitle.style.color = "var(--rose-400)";
  verdictSubtitle.textContent = message || "Could not connect to the backend.";

  meterFill.style.width = "0";
  detailStatus.textContent = "ERROR";
  detailStatus.style.color = "var(--rose-400)";
  detailModel.textContent    = "—";
  detailTime.textContent     = "—";
  detailFileName.textContent = selectedFile ? truncate(selectedFile.name, 20) : "—";

  resultPanel.classList.add("visible");
}

// ── Loading state ─────────────────────────────────────────
function setLoadingState(isLoading) {
  detectBtn.disabled = isLoading;
  detectBtn.classList.toggle("loading", isLoading);
}

// ── Toast ─────────────────────────────────────────────────
function showToast(message, duration = 3000) {
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), duration);
}

// ── Utilities ─────────────────────────────────────────────
function formatSize(bytes) {
  if (bytes < 1024)        return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

function truncate(str, max) {
  return str.length > max ? str.slice(0, max - 1) + "…" : str;
}

function animateCounter(el, from, to, duration) {
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = (from + (to - from) * eased).toFixed(1) + "%";
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
