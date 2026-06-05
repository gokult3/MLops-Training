/* ═══════════════════════════════════════════════════════
   YOLO Vision — app.js
═══════════════════════════════════════════════════════ */

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone       = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const browseBtn      = document.getElementById("browseBtn");
const previewWrap    = document.getElementById("previewWrap");
const previewImg     = document.getElementById("previewImg");
const clearBtn       = document.getElementById("clearBtn");
const confSlider     = document.getElementById("confSlider");
const confVal        = document.getElementById("confVal");
const detectBtn      = document.getElementById("detectBtn");
const loaderOverlay  = document.getElementById("loaderOverlay");
const loaderMsg      = document.getElementById("loaderMsg");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultImgWrap  = document.getElementById("resultImgWrap");
const resultImg      = document.getElementById("resultImg");
const resultStats    = document.getElementById("resultStats");
const detectionsWrap = document.getElementById("detectionsWrap");
const detList        = document.getElementById("detList");
const detCount       = document.getElementById("detCount");
const healthBadge    = document.getElementById("healthBadge");
const toast          = document.getElementById("toast");

let currentFile = null;

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch("/health");
    const d = await r.json();
    if (d.status === "ok") {
      healthBadge.textContent = "";
      healthBadge.innerHTML = `<span class="dot"></span> ${d.model} · ${d.classes} classes`;
      healthBadge.classList.add("ready");
    }
  } catch {
    healthBadge.textContent = "⚠ model unavailable";
  }
}
checkHealth();

// ── Slider ────────────────────────────────────────────────────────────────────
confSlider.addEventListener("input", () => {
  const v = confSlider.value;
  confVal.textContent = v + "%";
  confSlider.style.setProperty("--val", v + "%");
});
confSlider.style.setProperty("--val", confSlider.value + "%");

// ── File handling ─────────────────────────────────────────────────────────────
browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) handleFile(f);
  else showToast("Please drop an image file.", "error");
});

function handleFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    dropZone.classList.add("hidden");
    previewWrap.classList.remove("hidden");
    detectBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener("click", resetInput);
function resetInput() {
  currentFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewWrap.classList.add("hidden");
  dropZone.classList.remove("hidden");
  detectBtn.disabled = true;
  resetResults();
}

// ── Detection ─────────────────────────────────────────────────────────────────
detectBtn.addEventListener("click", runDetection);

async function runDetection() {
  if (!currentFile) return;

  loaderMsg.textContent = "Running YOLO inference…";
  loaderOverlay.classList.remove("hidden");
  detectBtn.disabled = true;

  const fd = new FormData();
  fd.append("image", currentFile);
  fd.append("conf", (parseInt(confSlider.value) / 100).toFixed(2));

  try {
    const res  = await fetch("/detect", { method: "POST", body: fd });
    const data = await res.json();

    if (!res.ok || data.error) {
      showToast(data.error || "Detection failed.", "error");
      return;
    }

    renderResults(data);
  } catch (err) {
    showToast("Network error: " + err.message, "error");
  } finally {
    loaderOverlay.classList.add("hidden");
    detectBtn.disabled = false;
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  resultPlaceholder.classList.add("hidden");
  resultImgWrap.classList.remove("hidden");
  detectionsWrap.classList.remove("hidden");

  // Annotated image (cache-bust)
  resultImg.src = "/static/" + data.result_image + "?t=" + Date.now();

  // Stats bar
  resultStats.innerHTML = `
    <span>⬡ <b>${data.total}</b> detection${data.total !== 1 ? "s" : ""}</span>
    <span>⏱ <b>${data.inference_ms}ms</b></span>
    <span>📐 <b>${data.image_size.width}×${data.image_size.height}</b></span>
    <span>🧠 <b>${data.model}</b></span>
  `;

  // Detection count badge
  detCount.textContent = data.total;

  // Detection list
  detList.innerHTML = "";
  if (data.total === 0) {
    detList.innerHTML = `<div class="det-item" style="color:var(--muted);font-family:var(--mono);font-size:.78rem;">
      No objects detected above the confidence threshold.
    </div>`;
  } else {
    const palette = buildPalette(data.detections.map(d => d.class_id));
    data.detections.forEach((det, i) => {
      const color  = palette[det.class_id % palette.length];
      const confCls = det.confidence >= 70 ? "high" : det.confidence < 40 ? "low" : "";
      const [x1, y1, x2, y2] = det.bbox;
      const item = document.createElement("div");
      item.className = "det-item";
      item.style.animationDelay = (i * 0.05) + "s";
      item.innerHTML = `
        <div class="det-swatch" style="background:${color}"></div>
        <span class="det-label">${det.label}</span>
        <span class="det-conf ${confCls}">${det.confidence}%</span>
        <span class="det-bbox">[${x1},${y1} → ${x2},${y2}]</span>
      `;
      detList.appendChild(item);
    });
  }
}

function resetResults() {
  resultPlaceholder.classList.remove("hidden");
  resultImgWrap.classList.add("hidden");
  detectionsWrap.classList.add("hidden");
  detList.innerHTML = "";
}

// ── Palette ───────────────────────────────────────────────────────────────────
function buildPalette(classIds) {
  const colors = [
    "#00d2ff","#ff4d6d","#ffe033","#a0ff80",
    "#ff9a3c","#b388ff","#40e0d0","#ff6eb4",
    "#6ee7f7","#ffd166","#06d6a0","#ef476f",
  ];
  return colors;
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, type = "") {
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.className = "toast" + (type ? " " + type : "");
  toast.classList.remove("hidden");
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 3500);
}
