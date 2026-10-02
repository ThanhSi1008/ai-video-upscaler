import os
import sys
import time
import threading
import tempfile
import subprocess
import urllib.parse
import urllib.request
from queue import Queue
import importlib
import torch
import gradio as gr

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

try:
    gr.close_all()
except Exception:
    pass

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(repo_root)

import upscale
importlib.reload(upscale)
from upscale import upscale_video

for arg in sys.argv:
    if arg.startswith("--alias="):
        os.environ["TINYURL_ALIAS"] = arg.split("=", 1)[1]

device_obj, device_type, device_desc = upscale.get_best_device()
encoder_flags, encoder_desc = upscale.get_hevc_encoder_flags(device_type)

if device_type == 'mps':
    device_badge = f"🍎 Apple Silicon M3 Pro (MPS Metal Acceleration) | MÃ HÓA: {encoder_desc}"
elif device_type == 'cuda':
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_badge = f"🔥 {num_gpus}x NVIDIA CUDA (Multi-Processing) | MÃ HÓA: {encoder_desc}"
    else:
        device_badge = f"🚀 Single NVIDIA GPU (CUDA) | MÃ HÓA: {encoder_desc}"
else:
    device_badge = f"💻 CPU Software Mode | MÃ HÓA: {encoder_desc}"

MODEL_MAP = {
    "⚡ NGUỒN B: AnimeJaNai V3 UltraCompact (WEB-DL Gốc - Tốc Độ Nhanh ~6–8 FPS trên T4)": "animejanai_v3_ultracompact",
    "⚡ NGUỒN A: AnimeJaNai V3 Sharp UltraCompact (BDRip 10-bit - Tốc Độ Nhanh ~6–8 FPS trên T4)": "animejanai_v3_sharp_ultracompact",
    "🎯 NGUỒN B: AnimeJaNai V3 Compact (Khuyên dùng WEB-DL Gốc: SubsPlease/Erai - Master Quality ~3.8 FPS)": "animejanai_v3_compact",
    "🎯 NGUỒN A: AnimeJaNai V3 Sharp (Khuyên dùng BDRip 10-bit: Hi10P/Main 10 - Nét đanh giữ grain ~3.8 FPS)": "animejanai_v3_sharp",
}

CUSTOM_CSS = """
.container {
    max-width: 1360px;
    margin: 0 auto;
    padding: 20px;
}
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    color: #ffffff;
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-box p {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid #0284c7;
    color: #38bdf8;
    font-family: monospace;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 9999px;
    margin-top: 14px;
}
.panel-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
}

/* THANH ĐIỀU KHIỂN SO SÁNH & ĐỒNG BỘ THỜI GIAN */
.comp-toolbar-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
}
.comp-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(51, 65, 85, 0.7);
}
.comp-title {
    color: #f1f5f9;
    font-size: 0.96rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
}
.comp-btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    margin-bottom: 8px;
}
.btn-group {
    display: flex;
    gap: 6px;
    background: rgba(15, 23, 42, 0.6);
    padding: 4px;
    border-radius: 10px;
    border: 1px solid #334155;
}
.comp-btn {
    background: #1e293b;
    border: 1px solid #475569;
    color: #cbd5e1;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 7px 14px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
}
.comp-btn:hover {
    background: #334155;
    color: #ffffff;
    border-color: #38bdf8;
    transform: translateY(-1px);
}
.comp-btn.fs-hero-btn {
    background: linear-gradient(135deg, #0284c7 0%, #3b82f6 50%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: 1px solid #38bdf8 !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 8px 18px !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.35) !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.comp-btn.fs-hero-btn:hover {
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 50%, #4f46e5 100%) !important;
    box-shadow: 0 0 22px rgba(56, 189, 248, 0.6) !important;
    transform: translateY(-1px);
}
.comp-btn.accent-btn {
    background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
    color: #ffffff;
    border-color: #f97316;
}
.comp-btn.accent-btn:hover {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
    box-shadow: 0 0 12px rgba(249, 115, 22, 0.4);
}
.sync-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #059669;
    color: #34d399;
    font-family: monospace;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
}
.sync-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #10b981;
    animation: sync-pulse 1.5s infinite;
}
@keyframes sync-pulse {
    0% { opacity: 0.4; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.15); }
    100% { opacity: 0.4; transform: scale(0.9); }
}
.comp-tip {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.4;
    padding-top: 4px;
}
#comparison_row {
    position: relative;
    transition: all 0.3s ease;
}
#video_orig, #video_upscaled {
    transition: all 0.25s ease-in-out;
}

/* CHẾ ĐỘ TOÀN MÀN HÌNH (FULLSCREEN COMPARISON THEATER) */
#comparison_row:fullscreen,
#comparison_row.is-fullscreen {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    max-width: 100vw !important;
    max-height: 100vh !important;
    background: #000000 !important;
    z-index: 999999 !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
}

#comparison_row:fullscreen #video_orig,
#comparison_row:fullscreen #video_upscaled,
#comparison_row.is-fullscreen #video_orig,
#comparison_row.is-fullscreen #video_upscaled {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    max-width: 100vw !important;
    max-height: 100vh !important;
    background: #000000 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    border-radius: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

#comparison_row:fullscreen .wrap,
#comparison_row:fullscreen .video-container,
#comparison_row.is-fullscreen .wrap,
#comparison_row.is-fullscreen .video-container {
    width: 100% !important;
    height: 100% !important;
    max-height: 100vh !important;
    background: #000000 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: none !important;
}

#comparison_row:fullscreen video,
#comparison_row.is-fullscreen video {
    width: 100vw !important;
    height: 100vh !important;
    max-height: 100vh !important;
    max-width: 100vw !important;
    object-fit: contain !important;
    background: #000000 !important;
}

/* HEADS-UP DISPLAY (HUD) KHI CHUYỂN TẬP TRONG FULL SCREEN */
.fs-hud {
    display: none;
    position: absolute;
    top: 36px;
    left: 50%;
    transform: translateX(-50%) translateY(-25px);
    background: rgba(15, 23, 42, 0.92);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(56, 189, 248, 0.6);
    border-radius: 9999px;
    padding: 10px 24px;
    align-items: center;
    gap: 14px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.8), 0 0 24px rgba(56, 189, 248, 0.35);
    z-index: 1000000;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s cubic-bezier(0.16, 1, 0.3, 1), transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.fs-hud.visible {
    opacity: 1 !important;
    transform: translateX(-50%) translateY(0) !important;
}
.fs-hud-icon {
    font-size: 1.6rem;
    line-height: 1;
}
.fs-hud-text {
    display: flex;
    flex-direction: column;
}
.fs-hud-title {
    color: #f8fafc;
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.3px;
}
.fs-hud-sub {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 500;
}

/* THANH ĐIỀU KHIỂN NỔI KHI RÊ CHUỘT TRONG FULL SCREEN */
.fs-controls {
    display: none;
    position: absolute;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%) translateY(20px);
    width: 92%;
    max-width: 1050px;
    background: rgba(15, 23, 42, 0.94);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 16px;
    padding: 12px 20px;
    flex-direction: column;
    gap: 10px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.9), 0 0 25px rgba(56, 189, 248, 0.2);
    z-index: 1000000;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s cubic-bezier(0.16, 1, 0.3, 1), transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.fs-controls.visible {
    opacity: 1 !important;
    transform: translateX(-50%) translateY(0) !important;
    pointer-events: auto !important;
}

/* THANH TUA TIMELINE */
.fs-timeline-row {
    display: flex;
    align-items: center;
    gap: 14px;
    width: 100%;
}
.fs-time-text {
    font-family: monospace;
    font-size: 0.92rem;
    font-weight: 700;
    color: #e2e8f0;
    min-width: 52px;
    text-align: center;
}
.fs-timeline-slider {
    -webkit-appearance: none;
    appearance: none;
    flex: 1;
    width: 100%;
    height: 8px;
    background: rgba(51, 65, 85, 0.8);
    border-radius: 9999px;
    outline: none;
    cursor: pointer;
    transition: height 0.15s ease;
}
.fs-timeline-slider:hover {
    height: 10px;
}
.fs-timeline-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #38bdf8;
    border: 2px solid #ffffff;
    cursor: pointer;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.8);
    transition: transform 0.15s ease, background 0.15s ease;
}
.fs-timeline-slider::-webkit-slider-thumb:hover {
    transform: scale(1.25);
    background: #60a5fa;
}
.fs-timeline-slider::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #38bdf8;
    border: 2px solid #ffffff;
    cursor: pointer;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.8);
}

/* HÀNG CÁC NÚT ĐIỀU KHIỂN */
.fs-btn-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    flex-wrap: wrap;
    gap: 10px;
}
.fs-btn-group-left,
.fs-btn-group-center,
.fs-btn-group-right {
    display: flex;
    align-items: center;
    gap: 8px;
}
.fs-ep-btn.active-ep {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    border-color: #38bdf8 !important;
    color: #ffffff !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.5) !important;
}

.fs-ctrl-btn {
    background: #1e293b;
    border: 1px solid #475569;
    color: #e2e8f0;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
}
.fs-ctrl-btn:hover {
    background: #334155;
    color: #ffffff;
    border-color: #38bdf8;
    transform: translateY(-1px);
}
.fs-ctrl-play {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
    border-color: #38bdf8;
    color: #ffffff;
}
.fs-ctrl-exit {
    background: rgba(239, 68, 68, 0.2);
    border-color: #ef4444;
    color: #fca5a5;
}
.fs-ctrl-exit:hover {
    background: #ef4444;
    color: #ffffff;
}

#comparison_row:fullscreen .fs-hud,
#comparison_row:fullscreen .fs-controls,
#comparison_row.is-fullscreen .fs-hud,
#comparison_row.is-fullscreen .fs-controls {
    display: flex !important;
}
"""

COMPARISON_TOOLBAR_HTML = """
<div class="comp-toolbar-container">
  <div class="comp-header">
    <div class="comp-title">
      <span>🔍</span> <b>BỘ SO SÁNH & ĐỒNG BỘ THỜI GIAN HAI TẬP PHIM</b>
    </div>
    <div class="sync-badge">
      <span class="sync-dot"></span> <span id="sync-status-text">Đồng Bộ Khóa Lockstep: SẴN SÀNG</span>
    </div>
  </div>

  <div class="comp-btn-row">
    <!-- NÚT BẬT TOÀN MÀN HÌNH SO SÁNH (TẬP TRUNG CHÍNH) -->
    <button type="button" id="btn-enter-fs" class="comp-btn fs-hero-btn" onclick="window.enterComparisonFullscreen()">
      ⛶ BẬT TOÀN MÀN HÌNH SO SÁNH (Nhấn ← / → để chuyển tập)
    </button>

    <!-- Nhóm điều khiển đồng bộ -->
    <div class="btn-group">
      <button type="button" id="btn-sync-play" class="comp-btn accent-btn" onclick="window.toggleSyncPlay()">
        ⏯️ Phát / Tạm Dừng Cả Hai (Space)
      </button>
      <button type="button" class="comp-btn" onclick="window.seekSyncBoth(-5)">
        ⏪ Tua Lùi 5s
      </button>
      <button type="button" class="comp-btn" onclick="window.seekSyncBoth(5)">
        ⏩ Tua Tới 5s
      </button>
      <button type="button" id="btn-toggle-audio" class="comp-btn" onclick="window.toggleAudioSource()">
        🔊 Âm thanh: Video 4K (Bấm để đổi)
      </button>
    </div>
  </div>
  
  <div class="comp-tip">
    💡 <b>Trải Nghiệm Toàn Màn Hình:</b> Bấm <b>"⛶ BẬT TOÀN MÀN HÌNH SO SÁNH"</b>. Khi đang xem full screen, chỉ cần ấn phím <b>← (Mũi tên trái)</b> hoặc <b>→ (Mũi tên phải)</b> để chuyển đổi tức thì giữa Video Gốc (1080p) và Video 4K UHD như chuyển tập phim mới, giữ nguyên 100% thời gian đang chiếu!
  </div>
</div>
"""

HEAD_SCRIPTS = """
<script>
(function() {
  window.fsActiveEpisode = 'upscaled';
  window.activeAudio = 'upscaled';

  function getVideos() {
    const v1 = document.querySelector('#video_orig video');
    const v2 = document.querySelector('#video_upscaled video');
    return { v1, v2 };
  }

  function isFullscreenActive() {
    return !!(
      document.fullscreenElement ||
      document.webkitFullscreenElement ||
      document.mozFullScreenElement ||
      document.msFullscreenElement
    );
  }

  let hudTimer = null;
  window.showFsHud = function(icon, title, sub) {
    window.ensureFsElements();
    const hud = document.getElementById('fs-hud');
    const hudIcon = document.getElementById('fs-hud-icon');
    const hudTitle = document.getElementById('fs-hud-title');
    const hudSub = document.getElementById('fs-hud-sub');
    if (!hud) return;

    if (hudIcon) hudIcon.innerText = icon;
    if (hudTitle) hudTitle.innerText = title;
    if (hudSub) hudSub.innerText = sub;

    hud.classList.add('visible');
    clearTimeout(hudTimer);
    hudTimer = setTimeout(() => {
      hud.classList.remove('visible');
    }, 2400);
  };

  let fsHideTimer = null;
  function onMouseMoveFS() {
    if (!isFullscreenActive()) return;
    window.ensureFsElements();
    const ctrl = document.getElementById('fs-controls');
    if (ctrl) {
      ctrl.classList.add('visible');
      clearTimeout(fsHideTimer);
      fsHideTimer = setTimeout(() => {
        ctrl.classList.remove('visible');
      }, 3000);
    }
  }

  function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return '00:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
      return `${h}:${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
    }
    return `${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
  }

  window.ensureFsElements = function() {
    const compRow = document.getElementById('comparison_row');
    if (!compRow) return;

    if (!document.getElementById('fs-hud')) {
      const hud = document.createElement('div');
      hud.id = 'fs-hud';
      hud.className = 'fs-hud';
      hud.innerHTML = `
        <span id="fs-hud-icon" class="fs-hud-icon">✨</span>
        <div class="fs-hud-text">
          <div id="fs-hud-title" class="fs-hud-title">TẬP 2: VIDEO 4K UHD</div>
          <div id="fs-hud-sub" class="fs-hud-sub">Dùng phím ← / → để chuyển tập phim</div>
        </div>
      `;
      compRow.appendChild(hud);
    }

    if (!document.getElementById('fs-controls')) {
      const ctrl = document.createElement('div');
      ctrl.id = 'fs-controls';
      ctrl.className = 'fs-controls';
      ctrl.innerHTML = `
        <!-- DÒNG 1: THANH TUA TIMELINE ĐỒNG BỘ -->
        <div class="fs-timeline-row">
          <span id="fs-time-current" class="fs-time-text">00:00</span>
          <input type="range" id="fs-timeline-slider" class="fs-timeline-slider" min="0" max="100" step="0.05" value="0">
          <span id="fs-time-total" class="fs-time-text">00:00</span>
        </div>

        <!-- DÒNG 2: CÁC NÚT ĐIỀU KHIỂN & CHUYỂN TẬP -->
        <div class="fs-btn-row">
          <div class="fs-btn-group-left">
            <button type="button" id="fs-btn-play" class="fs-ctrl-btn fs-ctrl-play" onclick="window.toggleSyncPlay()">
              ⏯️ Phát / Tạm Dừng (Space)
            </button>
            <button type="button" class="fs-ctrl-btn" onclick="window.seekSyncBoth(-10)">
              ⏪ -10s
            </button>
            <button type="button" class="fs-ctrl-btn" onclick="window.seekSyncBoth(10)">
              ⏩ +10s
            </button>
          </div>

          <div class="fs-btn-group-center">
            <button type="button" id="fs-btn-ep-orig" class="fs-ctrl-btn fs-ep-btn" onclick="window.switchEpisode('orig')">
              ⏮ Tập Video Gốc (Phím ←)
            </button>
            <button type="button" id="fs-btn-ep-up" class="fs-ctrl-btn fs-ep-btn active-ep" onclick="window.switchEpisode('upscaled')">
              Tập Video 4K (Phím →) ⏭
            </button>
          </div>

          <div class="fs-btn-group-right">
            <button type="button" id="fs-btn-audio" class="fs-ctrl-btn" onclick="window.toggleAudioSource()">
              🔊 Âm Thanh: 4K
            </button>
            <button type="button" class="fs-ctrl-btn fs-ctrl-exit" onclick="window.exitComparisonFullscreen()">
              ✕ Thoát Toàn Màn Hình (Esc)
            </button>
          </div>
        </div>
      `;
      compRow.appendChild(ctrl);

      const slider = ctrl.querySelector('#fs-timeline-slider');
      slider.addEventListener('input', function() {
        window.isScrubbing = true;
        const { v1, v2 } = getVideos();
        const refVid = (window.fsActiveEpisode === 'orig' ? (v1 || v2) : (v2 || v1));
        if (!refVid || !refVid.duration) return;
        const targetTime = (parseFloat(slider.value) / 100) * refVid.duration;
        if (v1) v1.currentTime = targetTime;
        if (v2) v2.currentTime = targetTime;
        const curEl = document.getElementById('fs-time-current');
        if (curEl) curEl.innerText = formatTime(targetTime);
      });

      slider.addEventListener('change', function() {
        window.isScrubbing = false;
        const { v1, v2 } = getVideos();
        const refVid = (window.fsActiveEpisode === 'orig' ? (v1 || v2) : (v2 || v1));
        if (!refVid || !refVid.duration) return;
        const targetTime = (parseFloat(slider.value) / 100) * refVid.duration;
        if (v1) v1.currentTime = targetTime;
        if (v2) v2.currentTime = targetTime;
        onMouseMoveFS();
      });

      ctrl.addEventListener('mouseenter', () => {
        clearTimeout(fsHideTimer);
        ctrl.classList.add('visible');
      });
      ctrl.addEventListener('mouseleave', () => {
        onMouseMoveFS();
      });

      compRow.addEventListener('mousemove', onMouseMoveFS);
    }
  };

  window.enterComparisonFullscreen = function() {
    const compRow = document.getElementById('comparison_row');
    if (!compRow) return;

    window.ensureFsElements();

    if (compRow.requestFullscreen) {
      compRow.requestFullscreen().catch(()=>{});
    } else if (compRow.webkitRequestFullscreen) {
      compRow.webkitRequestFullscreen();
    } else if (compRow.mozRequestFullScreen) {
      compRow.mozRequestFullScreen();
    }
  };

  window.exitComparisonFullscreen = function() {
    if (document.exitFullscreen) {
      document.exitFullscreen().catch(()=>{});
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.mozCancelFullScreen) {
      document.mozCancelFullScreen();
    }
  };

  window.switchEpisode = function(target) {
    const { v1, v2 } = getVideos();
    if (!v1 && !v2) return;

    const curTime = (window.fsActiveEpisode === 'orig' ? (v1 ? v1.currentTime : 0) : (v2 ? v2.currentTime : 0));
    const isPlaying = (window.fsActiveEpisode === 'orig' ? (v1 && !v1.paused) : (v2 && !v2.paused));

    window.fsActiveEpisode = target;

    const fsElem = document.fullscreenElement || document.webkitFullscreenElement;
    if (fsElem === v1 && target === 'upscaled' && v2) {
      v2.currentTime = curTime;
      if (isPlaying) v2.play().catch(()=>{});
      v2.requestFullscreen().catch(()=>{});
      return;
    } else if (fsElem === v2 && target === 'orig' && v1) {
      v1.currentTime = curTime;
      if (isPlaying) v1.play().catch(()=>{});
      v1.requestFullscreen().catch(()=>{});
      return;
    }

    const cOrig = document.getElementById('video_orig');
    const cUp = document.getElementById('video_upscaled');

    if (target === 'orig') {
      if (v1) {
        v1.currentTime = curTime;
        if (isPlaying && v1.paused) v1.play().catch(()=>{});
        if (!isPlaying && !v1.paused) v1.pause();
        v1.muted = false;
      }
      if (v2) {
        v2.currentTime = curTime;
        v2.muted = true;
      }

      if (cOrig) {
        cOrig.style.setProperty('display', 'flex', 'important');
        cOrig.style.setProperty('opacity', '1', 'important');
        cOrig.style.setProperty('z-index', '20', 'important');
        cOrig.style.setProperty('pointer-events', 'auto', 'important');
      }
      if (cUp) {
        cUp.style.setProperty('display', 'none', 'important');
        cUp.style.setProperty('opacity', '0', 'important');
        cUp.style.setProperty('z-index', '10', 'important');
        cUp.style.setProperty('pointer-events', 'none', 'important');
      }

      window.showFsHud('📺', 'TẬP 1: VIDEO GỐC (1080p)', 'Dùng phím → để chuyển sang Tập Video 4K UHD');
    } else {
      if (v2) {
        v2.currentTime = curTime;
        if (isPlaying && v2.paused) v2.play().catch(()=>{});
        if (!isPlaying && !v2.paused) v2.pause();
        v2.muted = false;
      }
      if (v1) {
        v1.currentTime = curTime;
        v1.muted = true;
      }

      if (cUp) {
        cUp.style.setProperty('display', 'flex', 'important');
        cUp.style.setProperty('opacity', '1', 'important');
        cUp.style.setProperty('z-index', '20', 'important');
        cUp.style.setProperty('pointer-events', 'auto', 'important');
      }
      if (cOrig) {
        cOrig.style.setProperty('display', 'none', 'important');
        cOrig.style.setProperty('opacity', '0', 'important');
        cOrig.style.setProperty('z-index', '10', 'important');
        cOrig.style.setProperty('pointer-events', 'none', 'important');
      }

      window.showFsHud('✨', 'TẬP 2: VIDEO 4K UHD (Native 2x)', 'Dùng phím ← để chuyển về Tập Video Gốc (1080p)');
    }

    const btnEpOrig = document.getElementById('fs-btn-ep-orig');
    const btnEpUp = document.getElementById('fs-btn-ep-up');
    if (target === 'orig') {
      if (btnEpOrig) btnEpOrig.classList.add('active-ep');
      if (btnEpUp) btnEpUp.classList.remove('active-ep');
    } else {
      if (btnEpOrig) btnEpOrig.classList.remove('active-ep');
      if (btnEpUp) btnEpUp.classList.add('active-ep');
    }
  };

  window.toggleSyncPlay = function() {
    const { v1, v2 } = getVideos();
    if (!v1 && !v2) return;
    const isPaused = (v2 ? v2.paused : (v1 ? v1.paused : true));
    const btnPlay = document.getElementById('fs-btn-play');
    if (isPaused) {
      if (v1 && v2) v1.currentTime = v2.currentTime;
      if (v1) v1.play().catch(()=>{});
      if (v2) v2.play().catch(()=>{});
      if (btnPlay) btnPlay.innerText = '⏸️ Tạm Dừng (Space)';
    } else {
      if (v1) v1.pause();
      if (v2) v2.pause();
      if (btnPlay) btnPlay.innerText = '▶️ Tiếp Tục Phát (Space)';
    }
  };

  window.seekSyncBoth = function(seconds) {
    const { v1, v2 } = getVideos();
    const cur = v2 ? v2.currentTime : (v1 ? v1.currentTime : 0);
    const target = Math.max(0, cur + seconds);
    if (v1) v1.currentTime = target;
    if (v2) v2.currentTime = target;
  };

  window.toggleAudioSource = function() {
    const { v1, v2 } = getVideos();
    const btn = document.getElementById('btn-toggle-audio');
    if (window.activeAudio === 'upscaled') {
      window.activeAudio = 'orig';
      if (v1) v1.muted = false;
      if (v2) v2.muted = true;
      if (btn) btn.innerText = '🔊 Âm thanh: Video Gốc';
      if (isFullscreenActive()) window.showFsHud('🔊', 'ÂM THANH: VIDEO GỐC', '');
    } else {
      window.activeAudio = 'upscaled';
      if (v1) v1.muted = true;
      if (v2) v2.muted = false;
      if (btn) btn.innerText = '🔊 Âm thanh: Video 4K';
      if (isFullscreenActive()) window.showFsHud('🔊', 'ÂM THANH: VIDEO 4K UHD', '');
    }
  };

  /* XỬ LÝ PHÍM BẤM: PHÍM ← VÀ → CHUYỂN TẬP KHI Ở FULL SCREEN */
  window.addEventListener('keydown', function(e) {
    const tag = document.activeElement ? document.activeElement.tagName : '';
    if (['INPUT', 'TEXTAREA'].includes(tag)) return;

    const isFS = isFullscreenActive();

    if (isFS) {
      // 1. KHI Ở CHẾ ĐỘ FULL SCREEN: Phím ← và → chuyển đổi giữa 2 tập phim
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        e.stopPropagation();
        window.switchEpisode('orig');
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        e.stopPropagation();
        window.switchEpisode('upscaled');
      } else if (e.code === 'Space') {
        e.preventDefault();
        e.stopPropagation();
        window.toggleSyncPlay();
      }
    } else {
      // 2. KHI KHÔNG Ở FULL SCREEN: Phím mũi tên dùng để tua 5s giữ 2 video đồng bộ
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        window.seekSyncBoth(-5);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        window.seekSyncBoth(5);
      } else if (e.code === 'Space') {
        const compRow = document.getElementById('comparison_row');
        if (compRow && (compRow.contains(document.activeElement) || document.activeElement === document.body)) {
          e.preventDefault();
          window.toggleSyncPlay();
        }
      }
    }
  });

  function handleFsChange() {
    const compRow = document.getElementById('comparison_row');
    if (!compRow) return;

    const isFS = isFullscreenActive();
    const fsElem = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement;

    if (isFS && (fsElem === compRow || compRow.contains(fsElem))) {
      compRow.classList.add('is-fullscreen');
      window.ensureFsElements();
      window.switchEpisode(window.fsActiveEpisode || 'upscaled');
      onMouseMoveFS();
    } else if (!isFS) {
      compRow.classList.remove('is-fullscreen');
      const cOrig = document.getElementById('video_orig');
      const cUp = document.getElementById('video_upscaled');
      if (cOrig) {
        cOrig.style.removeProperty('display');
        cOrig.style.removeProperty('opacity');
        cOrig.style.removeProperty('z-index');
        cOrig.style.removeProperty('pointer-events');
        cOrig.style.display = 'block';
        cOrig.style.width = '50%';
        cOrig.style.maxWidth = '50%';
        cOrig.style.flex = '1 1 50%';
      }
      if (cUp) {
        cUp.style.removeProperty('display');
        cUp.style.removeProperty('opacity');
        cUp.style.removeProperty('z-index');
        cUp.style.removeProperty('pointer-events');
        cUp.style.display = 'block';
        cUp.style.width = '50%';
        cUp.style.maxWidth = '50%';
        cUp.style.flex = '1 1 50%';
      }
      const hud = document.getElementById('fs-hud');
      if (hud) hud.classList.remove('visible');
      const ctrl = document.getElementById('fs-controls');
      if (ctrl) ctrl.classList.remove('visible');
    }
  }

  document.addEventListener('fullscreenchange', handleFsChange);
  document.addEventListener('webkitfullscreenchange', handleFsChange);
  document.addEventListener('mozfullscreenchange', handleFsChange);

  /* KHÓA ĐỒNG BỘ THỜI GIAN HAI VIDEO LIÊN TỤC (LOCKSTEP SYNC) */
  function attachTimeSync() {
    const { v1, v2 } = getVideos();
    if (!v1 || !v2) return;

    if (v1._hasSyncAttached && v2._hasSyncAttached) return;
    v1._hasSyncAttached = true;
    v2._hasSyncAttached = true;

    if (window.activeAudio === 'upscaled') {
      v1.muted = true;
      v2.muted = false;
    } else {
      v1.muted = false;
      v2.muted = true;
    }

    let isSyncing = false;
    function sync(from, to) {
      if (isSyncing) return;
      isSyncing = true;
      try {
        if (Math.abs(from.currentTime - to.currentTime) > 0.05) {
          to.currentTime = from.currentTime;
        }
        if (from.paused && !to.paused) {
          to.pause();
        } else if (!from.paused && to.paused) {
          to.play().catch(()=>{});
        }
        if (to.playbackRate !== from.playbackRate) {
          to.playbackRate = from.playbackRate;
        }
      } finally {
        setTimeout(() => { isSyncing = false; }, 30);
      }
    }

    v1.addEventListener('play', () => sync(v1, v2));
    v1.addEventListener('pause', () => sync(v1, v2));
    v1.addEventListener('seeking', () => sync(v1, v2));
    v1.addEventListener('seeked', () => sync(v1, v2));
    v1.addEventListener('timeupdate', () => {
      if (!v1.paused && !isSyncing && Math.abs(v1.currentTime - v2.currentTime) > 0.15) {
        v2.currentTime = v1.currentTime;
      }
      updateTimelineUI();
    });
    v1.addEventListener('durationchange', updateTimelineUI);

    v2.addEventListener('play', () => sync(v2, v1));
    v2.addEventListener('pause', () => sync(v2, v1));
    v2.addEventListener('seeking', () => sync(v2, v1));
    v2.addEventListener('seeked', () => sync(v2, v1));
    v2.addEventListener('timeupdate', () => {
      if (!v2.paused && !isSyncing && Math.abs(v2.currentTime - v1.currentTime) > 0.15) {
        v1.currentTime = v2.currentTime;
      }
      updateTimelineUI();
    });
    v2.addEventListener('durationchange', updateTimelineUI);

    function updateTimelineUI() {
      if (window.isScrubbing) return;
      const refVid = (window.fsActiveEpisode === 'orig' ? (v1 || v2) : (v2 || v1));
      if (!refVid || isNaN(refVid.duration) || refVid.duration <= 0) return;

      const slider = document.getElementById('fs-timeline-slider');
      const curEl = document.getElementById('fs-time-current');
      const totEl = document.getElementById('fs-time-total');

      if (slider) slider.value = (refVid.currentTime / refVid.duration) * 100;
      if (curEl) curEl.innerText = formatTime(refVid.currentTime);
      if (totEl) totEl.innerText = formatTime(refVid.duration);
    }

    const statusText = document.getElementById('sync-status-text');
    if (statusText) statusText.innerText = 'Đồng Bộ Khóa Lockstep: HOẠT ĐỘNG';
  }

  setInterval(attachTimeSync, 500);
})();
</script>
"""

def create_tinyurl(target_url, custom_alias=None, api_token=None):
    custom_alias = custom_alias or os.environ.get("TINYURL_ALIAS")
    api_token = api_token or os.environ.get("TINYURL_API_TOKEN")

    if custom_alias:
        try:
            api_url = f"https://tinyurl.com/api-create.php?url={urllib.parse.quote(target_url)}&alias={urllib.parse.quote(custom_alias)}"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_url = resp.read().decode('utf-8').strip()
                if "tinyurl.com" in res_url:
                    return res_url
        except Exception:
            pass

    try:
        api_url = f"https://tinyurl.com/api-create.php?url={urllib.parse.quote(target_url)}"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode('utf-8').strip()
    except Exception:
        return None

def ensure_web_friendly_video(video_path, output_dir=None):
    """
    Tạo nhanh bản preview MP4 (H.264 yuv420p + faststart) tương thích 100% với trình duyệt web.
    Sử dụng GPU (NVENC / VideoToolbox) để hoàn tất chỉ trong 1-2 giây,
    ngăn chặn Gradio tự động re-encode bằng CPU gây chậm trễ 5-10 phút.
    """
    if not video_path or not os.path.exists(video_path):
        return video_path

    try:
        from gradio import processing_utils
        if processing_utils.video_is_playable(video_path):
            return video_path
    except Exception:
        pass

    if output_dir is None:
        if os.path.exists('/content'):
            output_dir = '/content/temp_web'
        elif os.path.exists('/kaggle/working'):
            output_dir = '/kaggle/working/temp_web'
        else:
            output_dir = os.path.join(tempfile.gettempdir(), 'ai_upscale_web')
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_preview = os.path.join(output_dir, f"_web_{base}.mp4")

    # Nếu đã tạo rồi và kích thước hợp lệ
    if os.path.exists(out_preview) and os.path.getsize(out_preview) > 1000:
        return out_preview

    encoder = "libx264"
    extra_flags = ["-preset", "ultrafast", "-crf", "23"]

    if torch.cuda.is_available() and sys.platform.startswith("linux"):
        encoder = "h264_nvenc"
        extra_flags = ["-preset", "p1", "-cq", "24"]
    elif sys.platform == "darwin":
        encoder = "h264_videotoolbox"
        extra_flags = ["-b:v", "8M"]

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", encoder,
        *extra_flags,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out_preview
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(out_preview) and os.path.getsize(out_preview) > 1000:
            return out_preview
    except Exception as e:
        print(f"⚠️ Encode {encoder} không thành công ({e}), fallback sang libx264 ultrafast...")
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "24",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            out_preview
        ]
        try:
            subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(out_preview) and os.path.getsize(out_preview) > 1000:
                return out_preview
        except Exception as e_fb:
            print(f"⚠️ Fallback cũng lỗi ({e_fb}), trả về file gốc.")
            return video_path
    return video_path

def update_sub_model_choices(provider):
    if "DeepSeek" in provider:
        return gr.update(choices=["deepseek-chat", "deepseek-coder"], value="deepseek-chat")
    elif "OpenAI" in provider:
        return gr.update(choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], value="gpt-4o-mini")
    else:
        return gr.update(choices=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"], value="gemini-2.5-flash")

def process_ui(
    file_upload,
    local_path_or_link,
    output_dir_custom,
    model_choice,
    enable_dedup=True,
    dedup_thresh=0.003,
    translate_sub=False,
    sub_provider="Google Gemini (Khuyên dùng - Nhanh & Miễn phí)",
    sub_model="gemini-2.5-flash",
    sub_api_key="",
    progress=gr.Progress(track_tqdm=True)
):
    target_input = None
    if file_upload is not None:
        if isinstance(file_upload, str):
            target_input = file_upload
        elif hasattr(file_upload, 'name'):
            target_input = file_upload.name
    elif local_path_or_link and local_path_or_link.strip():
        target_input = os.path.expanduser(local_path_or_link.strip())

    if not target_input:
        raise gr.Error("❌ Vui lòng kéo thả file video từ Finder hoặc nhập đường dẫn file trên máy Mac!")

    # Nếu là file cục bộ, kiểm tra xem có tồn tại không
    is_gdrive = isinstance(target_input, str) and ("drive.google.com" in target_input or "drive.usercontent.google.com" in target_input)
    if not is_gdrive and not os.path.exists(target_input):
        raise gr.Error(f"❌ Không tìm thấy file: '{target_input}'. Vui lòng kiểm tra lại đường dẫn!")

    out_dir_clean = os.path.expanduser(output_dir_custom.strip()) if output_dir_custom and output_dir_custom.strip() else os.path.expanduser('~/Movies/Upscaled')
    os.makedirs(out_dir_clean, exist_ok=True)

    api_type = "gemini"
    if "DeepSeek" in sub_provider:
        api_type = "deepseek"
    elif "OpenAI" in sub_provider:
        api_type = "openai"

    api_key_clean = sub_api_key.strip() if sub_api_key else ""
    if translate_sub and not api_key_clean:
        if api_type == "gemini":
            api_key_clean = os.environ.get("GEMINI_API_KEY", "")
        elif api_type == "deepseek":
            api_key_clean = os.environ.get("DEEPSEEK_API_KEY", "")
        elif api_type == "openai":
            api_key_clean = os.environ.get("OPENAI_API_KEY", "")

    if translate_sub and not api_key_clean:
        raise gr.Error("❌ Bạn đã bật tính năng dịch phụ đề TriSub AI nhưng chưa nhập API Key! Vui lòng nhập API Key (lấy Gemini API Key miễn phí tại https://aistudio.google.com) hoặc cài biến môi trường GEMINI_API_KEY.")

    model_name = MODEL_MAP.get(model_choice, "animejanai_v3_compact")
    progress_queue = Queue()

    def progress_cb(pct, desc=""):
        progress_queue.put((pct, desc))
        if pct is not None:
            progress(pct, desc=desc)

    web_orig = None
    if os.path.exists(target_input):
        try:
            web_orig = ensure_web_friendly_video(target_input)
        except Exception as e_orig:
            print(f"⚠️ Chuẩn bị web preview nguồn: {e_orig}")
            web_orig = target_input

    yield web_orig, None, gr.update(visible=False), gr.update(visible=False), f"⏳ Đang khởi tạo luồng giải mã Native 2x (4K HEVC 10-bit Metal & Smart Dedup)..."

    output_result = [None]
    error_result = [None]

    def worker():
        try:
            res = upscale.upscale_video(
                video_input=target_input,
                output_dir=out_dir_clean,
                model_name=model_name,
                progress_callback=progress_cb,
                translate_sub=translate_sub,
                sub_api_key=api_key_clean if translate_sub else None,
                sub_api_type=api_type,
                sub_model=sub_model.strip() if sub_model else None,
                enable_dedup=enable_dedup,
                dedup_threshold=float(dedup_thresh)
            )
            output_result[0] = res
        except Exception as e:
            error_result[0] = e
        finally:
            progress_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        try:
            item = progress_queue.get(timeout=0.2)
            if item is None:
                break
            pct, desc = item
            if desc:
                yield gr.update(), gr.update(), gr.update(), gr.update(), desc
        except Exception:
            if not thread.is_alive() and progress_queue.empty():
                break

    thread.join()

    if error_result[0]:
        raise gr.Error(f"❌ Lỗi xử lý: {str(error_result[0])}")

    output_path = output_result[0]
    yield gr.update(), gr.update(), gr.update(), gr.update(), "⚡ Đang chuẩn bị web preview xem trực tiếp..."
    web_output = ensure_web_friendly_video(output_path)

    if not web_orig and os.path.exists(target_input):
        web_orig = ensure_web_friendly_video(target_input)

    sub_download_update = gr.update(visible=False)
    has_sub = False
    if output_path:
        out_dir = os.path.dirname(output_path)
        base_name = os.path.basename(os.path.splitext(output_path)[0]).replace("_4K", "")
        possible_srt = os.path.join(out_dir, f"{base_name}_Vietsub.srt")
        if os.path.exists(possible_srt) and os.path.getsize(possible_srt) > 20:
            sub_download_update = gr.update(value=possible_srt, visible=True)
            has_sub = True

    status_msg = f"✨ Nâng cấp thành công! Tập phim 4K đã lưu tại: {output_path}."
    if translate_sub and has_sub:
        status_msg += " Đã tự động dịch và nhúng phụ đề Tiếng Việt (TriSub AI) làm track mặc định!"
    status_msg += " Đã khóa đồng bộ thời gian hai video (dùng phím ← / → để so sánh)."

    yield web_orig, web_output, gr.update(value=output_path, visible=True), sub_download_update, status_msg

def process_quick_sub_ui(
    sub_file_upload,
    sub_path_input,
    sub_out_dir_input,
    sub_provider,
    sub_model,
    sub_api_key,
    progress=gr.Progress(track_tqdm=True)
):
    target_input = None
    if sub_file_upload is not None:
        target_input = sub_file_upload if isinstance(sub_file_upload, str) else sub_file_upload.name
    elif sub_path_input and sub_path_input.strip():
        target_input = os.path.expanduser(sub_path_input.strip())

    if not target_input:
        raise gr.Error("❌ Vui lòng chọn/kéo thả file từ Finder hoặc nhập đường dẫn file video / phụ đề!")

    is_gdrive = isinstance(target_input, str) and ("drive.google.com" in target_input or "drive.usercontent.google.com" in target_input)
    if not is_gdrive and not os.path.exists(target_input):
        raise gr.Error(f"❌ Không tìm thấy file: '{target_input}'!")

    out_dir = os.path.expanduser(sub_out_dir_input.strip()) if sub_out_dir_input and sub_out_dir_input.strip() else os.path.expanduser('~/Movies/Upscaled')
    os.makedirs(out_dir, exist_ok=True)

    api_type = "gemini"
    if "DeepSeek" in sub_provider:
        api_type = "deepseek"
    elif "OpenAI" in sub_provider:
        api_type = "openai"

    api_key_clean = sub_api_key.strip() if sub_api_key else ""
    if not api_key_clean:
        if api_type == "gemini":
            api_key_clean = os.environ.get("GEMINI_API_KEY", "")
        elif api_type == "deepseek":
            api_key_clean = os.environ.get("DEEPSEEK_API_KEY", "")
        elif api_type == "openai":
            api_key_clean = os.environ.get("OPENAI_API_KEY", "")

    if not api_key_clean:
        raise gr.Error("❌ Thiếu API Key! Vui lòng nhập API Key (lấy Gemini API Key miễn phí tại https://aistudio.google.com).")

    yield gr.update(visible=False), gr.update(visible=False), "☁️ Đang kết nối và chuẩn bị file nguồn..."

    if is_gdrive:
        def dl_cb(pct, desc=""):
            progress(pct * 0.2, desc=desc)
        local_file = upscale.download_gdrive(target_input, output_dir=os.path.join(out_dir, "input"), progress_cb=dl_cb)
    else:
        local_file = target_input

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from trisub_ai.tri_sub_translate import translate_subtitles_for_video

    base_name = os.path.basename(os.path.splitext(local_file)[0])
    out_srt = os.path.join(out_dir, f"{base_name}_Vietsub.srt")

    progress_queue = Queue()
    def sub_cb(pct, desc=""):
        progress_queue.put((pct, desc))
        if pct is not None:
            progress(pct, desc=desc)

    output_result = [None]
    error_result = [None]

    def sub_worker():
        try:
            res = translate_subtitles_for_video(
                video_path=local_file,
                output_srt=out_srt,
                api_key=api_key_clean,
                api_type=api_type,
                model_name=sub_model.strip() if sub_model else None,
                progress_callback=sub_cb
            )
            output_result[0] = res
        except Exception as e:
            error_result[0] = e
        finally:
            progress_queue.put(None)

    sub_thread = threading.Thread(target=sub_worker, daemon=True)
    sub_thread.start()

    while True:
        try:
            item = progress_queue.get(timeout=0.2)
            if item is None:
                break
            pct, desc = item
            if desc:
                yield gr.update(visible=False), gr.update(visible=False), desc
        except Exception:
            if not sub_thread.is_alive() and progress_queue.empty():
                break

    sub_thread.join()

    if error_result[0]:
        raise gr.Error(f"❌ Lỗi dịch phụ đề: {str(error_result[0])}")

    out_path = output_result[0]

    # Đọc 25 câu đầu tiên làm preview
    preview_lines = []
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            preview_lines = lines[:75]
    preview_text = "".join(preview_lines)

    yield (
        gr.update(value=out_path, visible=True),
        gr.update(value=preview_text, visible=True),
        f"✅ Dịch thành công chỉ trong vài chục giây! File phụ đề lưu tại: {out_path}"
    )

with gr.Blocks(title="AI Video Upscaler 4K - Apple Silicon Native Studio", theme=gr.themes.Default(), css=CUSTOM_CSS) as app:
    with gr.Column(elem_classes=["container"]):
        with gr.Group(elem_classes=["header-box"]):
            gr.Markdown(f"""
            # 🎬 AI Video Upscaler 4K - Apple Silicon Native Studio
            Hệ thống chuyên dụng nâng cấp Anime 1080p lên **4K Ultra-HD (3840x2160 Native 2x)** tối ưu 100% cho chip **Apple Silicon (M-Series MPS Metal & VideoToolbox 10-bit)**.
            Tích hợp **Smart Anime Deduplication** tăng tốc gấp đôi và dịch phụ đề Anime Tiếng Việt **TriSub AI**.
            
            <div class="badge">THIẾT BỊ: {device_badge}</div>
            """)

        with gr.Tabs():
            with gr.Tab("🎬 Nâng Cấp 4K & Dịch Phụ Đề (All-in-One)"):
                with gr.Accordion("📖 Hướng dẫn sử dụng nhanh trên máy Mac", open=False):
                    gr.Markdown("""
                    ### 🍎 Hướng Dẫn Sử Dụng Trên Máy Mac
                    1. **Chọn Video Nguồn**: Kéo thả file video (.mkv, .mp4) trực tiếp từ Finder hoặc dán đường dẫn file trên máy Mac (ví dụ: `~/Movies/anime.mkv`).
                    2. **Mô Hình AI**:
                       - **⚡ UltraCompact (Khuyên dùng tối ưu M3 Pro ~7.0+ FPS)**: Tốc độ cao nhất, tiết kiệm pin, thời gian render nhanh gấp đôi.
                         - **NGUỒN B UltraCompact**: Tối ưu cho WEB-DL Gốc (SubsPlease, Erai-raws, Crunchyroll/Netflix).
                         - **NGUỒN A Sharp UltraCompact**: Tối ưu cho BDRip 10-bit (Hi10P / Main 10) đã deband sạch.
                       - **🎯 Compact (Master Quality)**: Chi tiết tối đa cho từng nét vẽ nghệ thuật.
                    3. **Smart Anime Deduplication**: Bật để tự động phát hiện và bỏ qua các khung hình tĩnh/trùng lặp của Anime, đẩy tốc độ render lên gấp ~2 lần mà chất lượng vẫn bảo tồn 100%.
                    4. **Dịch Phụ Đề Tiếng Việt (TriSub AI)**: Mở mục *Dịch Phụ Đề Tiếng Việt*, tích chọn bật dịch và dán API Key (Gemini miễn phí tại https://aistudio.google.com).
                    5. **Bắt Đầu**: Bấm **"🚀 Nâng Cấp Video 4K"**. Tập phim 4K Ultra-HD sẽ được mã hóa và xuất thẳng về thư mục `~/Movies/Upscaled`!
                    """)

                # 1. Ô CHỌN FILE HOẶC NHẬP ĐƯỜNG DẪN CỤC BỘ TRÊN MAC
                with gr.Row():
                    local_path_input = gr.Textbox(
                        value="",
                        label="📁 Đường Dẫn File Video Trên Máy Mac (hoặc Link Google Drive)",
                        placeholder="Ví dụ: ~/Movies/Mushoku_Tensei_S02E14.mkv hoặc kéo thả file sang ô bên phải ➔",
                        lines=2,
                        scale=7
                    )
                    file_uploader = gr.File(
                        label="📂 Hoặc Kéo Thả / Chọn File Từ Finder",
                        file_types=[".mkv", ".mp4", ".mov", ".avi", ".webm"],
                        scale=5
                    )

                output_folder_input = gr.Textbox(
                    value="~/Movies/Upscaled",
                    label="💾 Thư Mục Lưu Video 4K Kết Quả (Mặc định: ~/Movies/Upscaled)",
                    placeholder="Mặc định: ~/Movies/Upscaled",
                    lines=1
                )

                file_uploader.change(
                    fn=lambda f: f if isinstance(f, str) else (f.name if f else ""),
                    inputs=[file_uploader],
                    outputs=[local_path_input]
                )

                # 2. BẢNG ĐIỀU KHIỂN SO SÁNH & ĐỒNG BỘ THỜI GIAN
                comparison_toolbar = gr.HTML(COMPARISON_TOOLBAR_HTML)

                # 3. KHUNG VIDEO GỐC VÀ KẾT QUẢ 4K (CHỈ DÙNG ĐỂ PHÁT & SO SÁNH ĐỒNG BỘ)
                with gr.Row(equal_height=True, elem_id="comparison_row"):
                    orig_preview = gr.Video(
                        label="📺 Video Gốc (Nguồn Ban Đầu)",
                        elem_id="video_orig",
                        interactive=False,
                        scale=1
                    )
                    output_preview = gr.Video(
                        label="✨ Video 4K Kết Quả (Native 2x UHD)",
                        elem_id="video_upscaled",
                        interactive=False,
                        scale=1
                    )

                # 4. THANH TIẾN ĐỘ THỜI GIAN THỰC
                status_box = gr.Textbox(
                    label="📊 Tiến Độ & Trạng Thái Thời Gian Thực (Live Progress)",
                    value="Chờ chọn file hoặc nhập đường dẫn video...",
                    interactive=False
                )

                # 5. BẢNG CẤU HÌNH & NÚT BẮT ĐẦU / TẢI VỀ
                with gr.Row():
                    with gr.Column(scale=7):
                        with gr.Group(elem_classes=["panel-box"]):
                            model_dropdown = gr.Dropdown(
                                choices=list(MODEL_MAP.keys()),
                                value="⚡ NGUỒN B: AnimeJaNai V3 UltraCompact (WEB-DL Gốc - Tốc Độ Nhanh ~6–8 FPS trên T4)",
                                label="🤖 Mô Hình AI (Super-Resolution Native 2x UHD)",
                                info="Mô hình AI siêu phân giải chuyên dụng cho Anime, xử lý Native 4K UHD với tốc độ vượt trội và giữ nguyên 100% chi tiết gốc."
                            )

                            with gr.Group(elem_classes=["panel-box"]):
                                gr.Markdown("#### ⚡ Tăng Tốc Apple Silicon (Smart Frame Deduplication)")
                                with gr.Row():
                                    enable_dedup_cb = gr.Checkbox(
                                        label="Kích hoạt Smart Anime Deduplication (Tăng tốc gấp đôi ~7.2 FPS)",
                                        value=True,
                                        info="Tự động nhận diện và tái sử dụng frame tĩnh/trùng lặp đặc trưng của Anime, bỏ qua tính toán AI dư thừa."
                                    )
                                    dedup_thresh_slider = gr.Slider(
                                        minimum=0.001,
                                        maximum=0.008,
                                        value=0.003,
                                        step=0.001,
                                        label="Độ nhạy bỏ qua (Threshold)",
                                        info="0.003 là ngưỡng chuẩn tối ưu cho Anime (bảo toàn 100% nét vẽ các cảnh chuyển động)."
                                    )

                            with gr.Accordion("🌐 Dịch Phụ Đề Tiếng Việt (TriSub AI Vietsub)", open=False):
                                translate_sub_cb = gr.Checkbox(
                                    label="Bật tự động dịch phụ đề Anime sang Tiếng Việt",
                                    value=False,
                                    info="Trích xuất phụ đề mềm từ video, đối chiếu 3 thứ tiếng (Anh + Nhật + Trung) để dịch chuẩn phong cách Anime."
                                )
                                with gr.Row():
                                    sub_provider_radio = gr.Radio(
                                        choices=["Google Gemini (Khuyên dùng - Nhanh & Miễn phí)", "DeepSeek", "OpenAI"],
                                        value="Google Gemini (Khuyên dùng - Nhanh & Miễn phí)",
                                        label="Nhà cung cấp AI"
                                    )
                                    sub_model_dropdown = gr.Dropdown(
                                        choices=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"],
                                        value="gemini-2.5-flash",
                                        label="Mô hình AI"
                                    )
                                sub_api_key_box = gr.Textbox(
                                    label="API Key (Gemini / DeepSeek / OpenAI)",
                                    placeholder="Dán API Key (Gemini lấy miễn phí tại https://aistudio.google.com)",
                                    type="password",
                                    value=os.environ.get("GEMINI_API_KEY", "")
                                )
                                gr.Markdown("💡 **TriSub AI** sẽ tự động trích xuất phụ đề gốc trong video, dịch sang Tiếng Việt và nhúng làm track phụ đề mặc định trong file MKV 4K kết quả (giữ nguyên phụ đề gốc và font chữ).")

                                sub_provider_radio.change(
                                    fn=update_sub_model_choices,
                                    inputs=[sub_provider_radio],
                                    outputs=[sub_model_dropdown]
                                )

                    with gr.Column(scale=5):
                        submit_btn = gr.Button("🚀 Nâng Cấp Video 4K (Native 2x UHD)", variant="primary", size="lg")
                        download_file = gr.File(
                            label="📥 Tệp 4K kết quả (.mkv đầy đủ Sub & Audio)",
                            visible=False
                        )
                        download_sub_file = gr.File(
                            label="📝 File phụ đề Tiếng Việt riêng (.srt)",
                            visible=False
                        )

                submit_btn.click(
                    fn=process_ui,
                    inputs=[
                        file_uploader,
                        local_path_input,
                        output_folder_input,
                        model_dropdown,
                        enable_dedup_cb,
                        dedup_thresh_slider,
                        translate_sub_cb,
                        sub_provider_radio,
                        sub_model_dropdown,
                        sub_api_key_box
                    ],
                    outputs=[orig_preview, output_preview, download_file, download_sub_file, status_box]
                )

            with gr.Tab("⚡ Dịch Phụ Đề Nhanh (TriSub AI Standalone - Chỉ 30s)"):
                with gr.Group(elem_classes=["panel-box"]):
                    gr.Markdown("""
                    ### 🌐 Dịch Phụ Đề Siêu Tốc Bằng TriSub AI Trên Máy Mac
                    Nếu bạn đã có video hoặc file phụ đề và **chỉ muốn dịch sang Tiếng Việt** mà không cần nâng cấp video 4K:
                    - **⚡ Tốc độ cao**: Chỉ mất ~30 đến 60 giây cho cả tập phim 24 phút (~400 câu thoại).
                    - **🎬 Định dạng hỗ trợ**: Video MKV, MP4 (chứa phụ đề mềm) hoặc file phụ đề trực tiếp (.srt, .ass, .vtt).
                    - **🎯 Chất lượng dịch**: Đối chiếu 3 ngôn ngữ (Anh + Nhật + Trung) để xưng hô chuẩn phong cách Anime.
                    """)
                    with gr.Row():
                        quick_sub_path = gr.Textbox(
                            value="",
                            label="📁 Đường Dẫn File Video Hoặc Phụ Đề Trên Máy Mac",
                            placeholder="Ví dụ: ~/Movies/ReZero_19.ass hoặc .mkv, hoặc kéo thả file sang ô bên phải ➔",
                            lines=2,
                            scale=7
                        )
                        quick_sub_file_upload = gr.File(
                            label="📂 Hoặc Kéo Thả File Từ Finder",
                            file_types=[".mkv", ".mp4", ".mov", ".ass", ".srt", ".vtt"],
                            scale=5
                        )

                    quick_out_dir = gr.Textbox(
                        value="~/Movies/Upscaled",
                        label="💾 Thư Mục Lưu File Phụ Đề Tiếng Việt (.srt)",
                        lines=1
                    )

                    quick_sub_file_upload.change(
                        fn=lambda f: f if isinstance(f, str) else (f.name if f else ""),
                        inputs=[quick_sub_file_upload],
                        outputs=[quick_sub_path]
                    )

                    with gr.Row():
                        quick_provider_radio = gr.Radio(
                            choices=["Google Gemini (Khuyên dùng - Nhanh & Miễn phí)", "DeepSeek", "OpenAI"],
                            value="Google Gemini (Khuyên dùng - Nhanh & Miễn phí)",
                            label="Nhà cung cấp AI"
                        )
                        quick_model_dropdown = gr.Dropdown(
                            choices=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"],
                            value="gemini-2.5-flash",
                            label="Mô hình AI"
                        )
                    quick_api_key_box = gr.Textbox(
                        label="API Key (Gemini / DeepSeek / OpenAI)",
                        placeholder="Dán API Key (Gemini lấy miễn phí tại https://aistudio.google.com)",
                        type="password",
                        value=os.environ.get("GEMINI_API_KEY", "")
                    )
                    quick_provider_radio.change(
                        fn=update_sub_model_choices,
                        inputs=[quick_provider_radio],
                        outputs=[quick_model_dropdown]
                    )

                    quick_sub_btn = gr.Button("🌐 Bắt Đầu Dịch Phụ Đề Tiếng Việt (~30 giây)", variant="primary", size="lg")
                    quick_sub_status = gr.Textbox(
                        label="📊 Tiến độ dịch",
                        value="Chờ nhấn bắt đầu...",
                        interactive=False
                    )
                    quick_sub_file = gr.File(
                        label="📥 Tải file phụ đề Tiếng Việt kết quả (.srt)",
                        visible=False
                    )
                    quick_sub_preview = gr.Textbox(
                        label="📝 Xem trước nội dung phụ đề Tiếng Việt vừa dịch",
                        lines=12,
                        interactive=False,
                        visible=False
                    )

                    quick_sub_btn.click(
                        fn=process_quick_sub_ui,
                        inputs=[quick_sub_file_upload, quick_sub_path, quick_out_dir, quick_provider_radio, quick_model_dropdown, quick_api_key_box],
                        outputs=[quick_sub_file, quick_sub_preview, quick_sub_status]
                    )

if __name__ == '__main__':
    share_mode = True if ("--share" in sys.argv or "--public" in sys.argv or os.environ.get("GRADIO_SHARE") == "True") else False
    allowed_dirs = [
        "/tmp",
        tempfile.gettempdir(),
        os.getcwd(),
        os.path.expanduser('~'),
        os.path.expanduser('~/Movies'),
        os.path.expanduser('~/Downloads'),
        os.path.expanduser('~/Desktop'),
        "/kaggle/working",
        "/content"
    ]

    # Trên Linux/Colab: tự động giải phóng port 7860 nếu có tiến trình zombie cũ chiếm giữ
    if sys.platform.startswith("linux"):
        try:
            subprocess.run(["fuser", "-k", "7860/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        except Exception:
            pass

    server_port = 7860
    launch_res = None
    try:
        launch_res = app.queue().launch(
            server_name="0.0.0.0",
            server_port=server_port,
            share=share_mode,
            allowed_paths=allowed_dirs,
            prevent_thread_lock=True,
            head=HEAD_SCRIPTS
        )
    except Exception as e_port:
        print(f"⚠️ Port {server_port} không khả dụng ({e_port}), chuyển sang tự động chọn port trống...")
        launch_res = app.queue().launch(
            server_name="0.0.0.0",
            share=share_mode,
            allowed_paths=allowed_dirs,
            prevent_thread_lock=True,
            head=HEAD_SCRIPTS
        )

    share_url = None
    if isinstance(launch_res, tuple) and len(launch_res) == 3:
        share_url = launch_res[2]
    if not share_url:
        share_url = getattr(app, "share_url", None)

    if share_url:
        print("\n" + "="*68, flush=True)
        print(f"🌐 GRADIO PUBLIC URL:  {share_url}", flush=True)
        tiny_url = create_tinyurl(share_url)
        if tiny_url:
            print(f"🔗 TINYURL SHORTLINK:   {tiny_url}", flush=True)
            print(f"💡 Dùng ngay link TinyURL trên để mở WebUI!", flush=True)
        print("="*68 + "\n", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Ứng dụng đã dừng.")
