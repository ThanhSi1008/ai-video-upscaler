// -*- coding: utf-8 -*-
/**
 * web/static/js/app.js
 * Main Real-Time Web Studio Orchestrator:
 * - Connects to 1.0 Hz Server-Sent Events (/api/events) with automatic REST fallback.
 * - Drives Tab 1 Live Monitor (6-stage stepper, KPIs, 4 High-DPI charts, Chunk Heatmap,
 *   .temp_* checkpoints, Graceful Pause / Force Stop, filterable Live Log Console).
 * - Coordinates Tabs 2..5 controllers, Theme Switcher, and accessible focusgroup navigation.
 */

import { TelemetryChartsController } from "./charts.js";
import { initCompareStage } from "./compare_stage.js";
import {
  initModelsPanel,
  initQueuePanel,
  initSubtransAndAuditsPanel,
} from "./panels.js";
import {
  apiFetch,
  clearChildren,
  el,
  formatDuration,
  initTabs,
  initThemeSwitcher,
  showToast,
} from "./ui.js";

const STEP_ORDER = [
  "probing",
  "preflight_dedup",
  "upscaling",
  "muxing",
  "auditing",
  "completed",
];

function parseElapsedSeconds(startedAtIso) {
  if (!startedAtIso) return null;
  const ts = Date.parse(startedAtIso);
  if (Number.isNaN(ts)) return null;
  return Math.max(0, Math.round((Date.now() - ts) / 1000));
}

function bootstrapStudio() {
  const charts = new TelemetryChartsController();

  // Header status pills
  const sseBadge = document.getElementById("sse-badge");
  const sseBadgeText = document.getElementById("sse-badge-text");
  const sourceBadge = document.getElementById("source-badge");
  const sourceBadgeText = document.getElementById("source-badge-text");
  const gpuBadge = document.getElementById("gpu-badge");
  const gpuBadgeText = document.getElementById("gpu-badge-text");
  const footerClock = document.getElementById("footer-clock");

  // Live Monitor DOM refs
  const episodeTitle = document.getElementById("live-episode-title");
  const btnGoQueue = document.getElementById("btn-go-queue");
  const btnPauseGraceful = document.getElementById("btn-pause-graceful");
  const btnClearPause = document.getElementById("btn-clear-pause");
  const btnAbortJob = document.getElementById("btn-abort-job");
  const stepperEl = document.getElementById("pipeline-stepper");
  const progressEl = document.getElementById("episode-progress");
  const progressLabel = document.getElementById("episode-progress-label");
  const stageDetailEl = document.getElementById("live-stage-detail");

  const kpiModel = document.getElementById("kpi-model");
  const kpiFrames = document.getElementById("kpi-frames");
  const kpiChunks = document.getElementById("kpi-chunks");
  const kpiFps = document.getElementById("kpi-fps");
  const kpiEta = document.getElementById("kpi-eta");
  const kpiDisk = document.getElementById("kpi-disk");
  const hwLiveSummary = document.getElementById("hw-live-summary");

  const chunkSummary = document.getElementById("chunk-legend-summary");
  const chunkHeatmap = document.getElementById("chunk-heatmap");
  const checkpointList = document.getElementById("checkpoint-list");

  const logAutoscroll = document.getElementById("log-autoscroll");
  const btnCopyLogs = document.getElementById("btn-copy-logs");
  const logFilterInput = document.getElementById("log-filter-input");
  const logLevelSelect = document.getElementById("log-level-select");
  const logPre = document.getElementById("live-log-pre");
  const logCode = document.getElementById("live-log-code");

  let logRing = [];
  const maxLogLines = 450;
  let lastLogCursor = { path: "", sig: "", offset: 0 };
  let lastEffectiveStage = null;
  let lastActivePipelineStep = null;
  let lastHeatmapSignature = "";
  let lastCheckpointSignature = null;
  let gpuStaticInfo = { gpu_name: "NVIDIA RTX GPU", vram_total_mib: 12288 };

  let compareCtrl = null;
  let queueCtrl = null;
  let modelsCtrl = null;
  let subtransCtrl = null;

  const tabsCtrl = initTabs((activeTabId) => {
    if (activeTabId === "tab-monitor") {
      charts.redrawAll();
      if (logPre && logAutoscroll instanceof HTMLInputElement && logAutoscroll.checked) {
        window.requestAnimationFrame(() => {
          logPre.scrollTop = logPre.scrollHeight;
        });
      }
    } else if (activeTabId === "tab-queue") {
      queueCtrl?.onTabActivated();
    } else if (activeTabId === "tab-compare") {
      compareCtrl?.redraw();
    } else if (activeTabId === "tab-subtrans") {
      subtransCtrl?.refreshAll();
    }
  });

  initThemeSwitcher(() => {
    charts.redrawAll();
    compareCtrl?.redraw();
  });

  compareCtrl = initCompareStage({
    selectTabById: tabsCtrl.selectTabById,
  });

  queueCtrl = initQueuePanel({
    selectTabById: tabsCtrl.selectTabById,
    onJobLaunched: () => {
      fetchSnapshotFallback();
    },
  });

  modelsCtrl = initModelsPanel({
    onShowsUpdated: () => {
      queueCtrl?.refreshSchedule();
    },
  });

  subtransCtrl = initSubtransAndAuditsPanel({
    selectTabById: tabsCtrl.selectTabById,
    compareCtrl,
    onJobLaunched: () => {
      fetchSnapshotFallback();
    },
  });

  // Footer live clock
  const updateClock = () => {
    if (footerClock) {
      footerClock.textContent = new Date().toLocaleString("vi-VN", { hour12: false });
    }
  };
  updateClock();
  window.setInterval(updateClock, 1000);

  // Live Log filtering & rendering
  const renderLogs = () => {
    if (!logCode || !logPre) return;
    const kw = (logFilterInput?.value || "").trim().toLowerCase();
    const mode = logLevelSelect?.value || "all";

    const filtered = logRing.filter((line) => {
      const low = line.toLowerCase();
      if (mode === "chunk" && !(low.includes("chunk") || low.includes("fps") || low.includes("đoạn"))) {
        return false;
      }
      if (
        mode === "warn" &&
        !(low.includes("[warn]") || low.includes("[error]") || low.includes("⚠") || low.includes("❌"))
      ) {
        return false;
      }
      if (
        mode === "subtrans" &&
        !(
          low.includes("sub") ||
          low.includes("gemini") ||
          low.includes("whisper") ||
          low.includes("dtw")
        )
      ) {
        return false;
      }
      if (kw && !low.includes(kw)) {
        return false;
      }
      return true;
    });

    logCode.textContent =
      filtered.length > 0
        ? filtered.join("\n")
        : logRing.length > 0
          ? "Không có dòng log nào khớp bộ lọc hiện tại."
          : "Chưa có nhật ký tiến trình.";

    if (logAutoscroll instanceof HTMLInputElement && logAutoscroll.checked) {
      window.requestAnimationFrame(() => {
        logPre.scrollTop = logPre.scrollHeight;
      });
    }
  };

  logFilterInput?.addEventListener("input", renderLogs);
  logLevelSelect?.addEventListener("change", renderLogs);

  btnCopyLogs?.addEventListener("click", async () => {
    const text = logCode?.textContent || "";
    try {
      await navigator.clipboard.writeText(text);
      showToast("Đã sao chép nhật ký vào bộ nhớ tạm.", "success", 2500);
    } catch {
      showToast("Trình duyệt không cho phép sao chép tự động.", "warn");
    }
  });

  // Hero Control Buttons
  btnGoQueue?.addEventListener("click", () => {
    tabsCtrl.selectTabById("tab-queue", { focusTab: false });
  });

  btnPauseGraceful?.addEventListener("click", async () => {
    try {
      await apiFetch("/api/jobs/pause-graceful", { method: "POST" });
      showToast(
        "Đã đặt cờ Dừng êm: Pipeline sẽ hoàn tất nốt chunk 8s hiện tại, bảo toàn checkpoint và tạm dừng sạch.",
        "info"
      );
      fetchSnapshotFallback();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  btnClearPause?.addEventListener("click", async () => {
    try {
      await apiFetch("/api/jobs/clear-pause", { method: "POST" });
      showToast("Đã hủy lệnh dừng êm — Pipeline tiếp tục chạy bình thường.", "success", 3000);
      fetchSnapshotFallback();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  btnAbortJob?.addEventListener("click", async () => {
    try {
      const res = await apiFetch("/api/jobs/abort", { method: "POST" });
      if (res.aborted_pid) {
        showToast(res.message || `Đã buộc dừng tiến trình Web Job (PID=${res.aborted_pid}).`, "warn");
      } else if (res.status === "external_lock") {
        showToast(
          res.message || `Tiến trình GPU bên ngoài (PID=${res.external_lock_pid || "?"}) vẫn đang chạy.`,
          "warn",
          6000
        );
      } else {
        showToast(res.message || "Không có tiến trình Web Job nào đang chạy.", "info");
      }
      fetchSnapshotFallback();
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  // Render Stepper + Chunk Heatmap + Checkpoints + KPIs
  const applyPipelineAndTelemetry = (telemetry, pipeline) => {
    if (!pipeline) return;
    const live = pipeline.live || {};
    const stage = pipeline.effective_stage || "idle";
    const activeSource = pipeline.active_source || "idle";
    const job = pipeline.job || {};
    const activeWebJob = job.active_job || null;
    const pausePending = Boolean(pipeline.graceful_pause_pending);

    // Detect completion transition only when an active job genuinely finishes (never on initial page load)
    const activeTransitionStages = new Set([
      "initializing",
      "probing",
      "preflight_dedup",
      "upscaling",
      "muxing",
      "auditing",
      "subtrans",
    ]);
    if (activeTransitionStages.has(lastEffectiveStage) && stage === "completed") {
      const toastMsg =
        lastEffectiveStage === "subtrans"
          ? "✅ Hoàn tất dịch & xử lý phụ đề thành công!"
          : "🎉 Hoàn tất xuất xưởng 4K & Mux Audit thành công!";
      showToast(toastMsg, "success", 6500);
      compareCtrl?.refreshCandidates();
      subtransCtrl?.refreshAll({ force: true });
      modelsCtrl?.refresh();
    }
    lastEffectiveStage = stage;
    if (stage === "idle") {
      lastActivePipelineStep = null;
    } else if (stage === "initializing") {
      lastActivePipelineStep = "probing";
    } else if (stage === "subtrans") {
      lastActivePipelineStep = "completed";
    } else if (STEP_ORDER.includes(stage) && stage !== "completed") {
      lastActivePipelineStep = stage;
    }

    // Header Source Badge
    if (sourceBadge && sourceBadgeText) {
      if (activeSource === "web") {
        sourceBadge.dataset.tone = "running";
        sourceBadgeText.textContent = `Đang chạy Web Job (${activeWebJob?.job_type || "upscale"} • PID ${activeWebJob?.pid || live.pid || "—"})`;
      } else if (activeSource === "cli") {
        sourceBadge.dataset.tone = "running";
        sourceBadgeText.textContent = `Đang đồng bộ CLI ngoài (PID ${pipeline.gpu_lock?.pid || live.pid || "—"})`;
      } else if (stage === "paused") {
        sourceBadge.dataset.tone = "warn";
        sourceBadgeText.textContent = "Đã tạm dừng (Có Checkpoint)";
      } else {
        sourceBadge.dataset.tone = "idle";
        sourceBadgeText.textContent = "Hệ thống rảnh";
      }
    }

    // Header GPU Badge
    if (gpuBadge && gpuBadgeText && telemetry) {
      const shortGpu = (gpuStaticInfo.gpu_name || "RTX GPU").replace("NVIDIA GeForce ", "");
      gpuBadgeText.textContent = `${shortGpu}: ${telemetry.gpu_util_pct.toFixed(0)}% • ${telemetry.temp_c.toFixed(0)}°C • ${telemetry.power_w.toFixed(0)}W`;
      gpuBadge.dataset.tone = telemetry.temp_c >= 83 ? "warn" : "info";
    }

    // Hardware summary line
    if (hwLiveSummary && telemetry) {
      hwLiveSummary.textContent =
        `CPU: ${telemetry.cpu_pct.toFixed(1)}% | P-State: ${telemetry.pstate} | ` +
        `Xung GPU: ${Math.round(telemetry.core_clock_mhz)} MHz | ` +
        `VRAM: ${Math.round(telemetry.vram_used_mib)} / ${Math.round(telemetry.vram_total_mib)} MiB`;
    }

    // Hero Episode Banner & Action Buttons
    const isRunning = activeSource === "web" || activeSource === "cli";
    const isSubtransWebJob =
      Boolean(activeWebJob) &&
      (activeWebJob.job_type === "subtrans" || activeWebJob.job_type === "subs_pending");

    if (episodeTitle) {
      if (isSubtransWebJob && isRunning) {
        const subTarget =
          activeWebJob.job_type === "subs_pending"
            ? "pending_subs.json"
            : activeWebJob.queue_items?.[0] || live.video_name || "";
        episodeTitle.textContent = `💬 Đang dịch phụ đề Anime: ${subTarget}`;
      } else if (live.video_name) {
        const prefix = isRunning
          ? "🎬 Đang xử lý: "
          : stage === "completed"
            ? "✅ Gần nhất: "
            : stage === "error"
              ? "❌ Lỗi phiên gần nhất: "
              : "⏸ Phiên gần nhất: ";
        episodeTitle.textContent = `${prefix}${live.video_name}`;
      } else {
        episodeTitle.textContent = "Chưa có tiến trình nào đang chạy — Sẵn sàng nhận tập phim mới";
      }
    }

    if (btnPauseGraceful instanceof HTMLButtonElement) {
      const subtransOnlyNoGpu = isSubtransWebJob && !pipeline.gpu_lock?.locked;
      btnPauseGraceful.disabled = !isRunning || pausePending || subtransOnlyNoGpu;
    }
    if (btnClearPause instanceof HTMLButtonElement) {
      btnClearPause.hidden = !pausePending;
    }
    if (btnAbortJob instanceof HTMLButtonElement) {
      btnAbortJob.disabled = activeSource !== "web";
    }

    // 6-Stage Stepper
    if (stepperEl) {
      const normalizeStep = (st) => {
        if (st === "initializing") return "probing";
        if (st === "subtrans") return "completed";
        return st;
      };
      const mappedStage = normalizeStep(stage);
      const curIdx = STEP_ORDER.indexOf(mappedStage);
      const hasCheckpoints = Array.isArray(pipeline.checkpoints) && pipeline.checkpoints.length > 0;
      const fallbackAnchor =
        live.failed_stage ||
        live.last_stage ||
        lastActivePipelineStep ||
        (hasCheckpoints ? "upscaling" : "probing");
      const anchorIdx = Math.max(0, STEP_ORDER.indexOf(normalizeStep(fallbackAnchor)));

      for (const li of stepperEl.querySelectorAll(".step-item")) {
        const stepKey = li.dataset.step;
        const idx = STEP_ORDER.indexOf(stepKey);
        delete li.dataset.state;
        if (stage === "completed") {
          li.dataset.state = "done";
        } else if (stage === "error") {
          if (idx < anchorIdx) li.dataset.state = "done";
          else if (idx === anchorIdx) li.dataset.state = "error";
        } else if (stage === "paused") {
          if (idx < anchorIdx) li.dataset.state = "done";
          else if (idx === anchorIdx) li.dataset.state = "paused";
        } else if (curIdx >= 0) {
          if (idx < curIdx) li.dataset.state = "done";
          else if (idx === curIdx) li.dataset.state = isRunning ? "active" : "done";
        }
      }
    }

    // Progress calculation
    const totChunks = Number(live.total_chunks || 0);
    const doneChunks = Number(live.completed_chunks || 0);
    let pct = 0;
    if (stage === "completed") {
      pct = 100;
    } else if (totChunks > 0) {
      pct = Math.min(99.5, (doneChunks / totChunks) * 100);
    }
    if (progressEl instanceof HTMLProgressElement) {
      progressEl.value = Number(pct.toFixed(1));
      progressEl.style.setProperty("--progress-pct", pct.toFixed(1));
    }
    if (progressLabel) {
      progressLabel.textContent = `Tiến độ phân đoạn: ${pct.toFixed(1)}%`;
    }
    if (stageDetailEl) {
      const pauseNote = pausePending ? " [⏸ Sẽ dừng êm sau chunk hiện tại]" : "";
      stageDetailEl.textContent = `${live.stage_detail || "Đang chờ tác vụ…"}${pauseNote}`;
    }

    // 6 KPI Boxes
    if (kpiModel) {
      const mId = live.model_id || "—";
      const modeTag = live.model_id ? (live.zero_disk ? "Zero-Disk" : "PNG Disk") : "";
      kpiModel.textContent = modeTag ? `${mId} (${modeTag})` : mId;
    }
    if (kpiFrames) {
      const totF = Number(live.total_frames || 0);
      const uniqF = Number(live.unique_frames || 0);
      if (totF > 0 && totChunks > 0) {
        const estDoneF =
          stage === "completed" ? totF : Math.min(totF, Math.round((doneChunks / totChunks) * totF));
        const dedupNote = live.dedup_ratio_percent ? ` (-${Number(live.dedup_ratio_percent).toFixed(1)}%)` : "";
        kpiFrames.textContent = `${estDoneF.toLocaleString("vi-VN")} / ${totF.toLocaleString("vi-VN")}${dedupNote}`;
      } else if (totF > 0) {
        kpiFrames.textContent = `${totF.toLocaleString("vi-VN")} (Unique: ${uniqF.toLocaleString("vi-VN")})`;
      } else {
        kpiFrames.textContent = "0 / 0";
      }
    }
    if (kpiChunks) {
      kpiChunks.textContent = `${doneChunks} / ${totChunks}`;
    }
    if (kpiFps) {
      const aiFps = Number(live.steady_fps || live.latest_fps || 0).toFixed(2);
      const effFps = Number(live.effective_fps || 0).toFixed(2);
      kpiFps.textContent = `${aiFps} / ${effFps} fps`;
    }
    if (kpiEta) {
      const elapsedSec = isRunning
        ? parseElapsedSeconds(activeWebJob?.started_at || live.started_at)
        : null;
      const elapsedStr = elapsedSec !== null ? formatDuration(elapsedSec) : "00:00";
      const etaStr =
        live.eta_seconds !== null && live.eta_seconds !== undefined
          ? formatDuration(live.eta_seconds)
          : "--:--";
      kpiEta.textContent = `${elapsedStr} / ${etaStr}`;
    }
    if (kpiDisk && telemetry) {
      kpiDisk.textContent = `${telemetry.disk_free_gb.toFixed(1)} GB trống`;
    }

    // Chunk Heatmap (only rebuild DOM when chunk state actually changes)
    if (chunkSummary) {
      chunkSummary.textContent =
        totChunks > 0
          ? `${doneChunks} / ${totChunks} phân đoạn 8s hoàn tất`
          : "Chưa có phân đoạn hoạt động";
    }
    if (chunkHeatmap) {
      const doneSet = new Set(live.resumed_chunks || []);
      const chunkPerfMap = new Map();
      for (const c of live.chunks || []) {
        doneSet.add(c.chunk_idx);
        chunkPerfMap.set(c.chunk_idx, c);
      }
      const activeChunk =
        isRunning && typeof live.active_chunk === "number" ? live.active_chunk : -1;
      const sig = `${totChunks}|${doneSet.size}|${activeChunk}|${stage}`;

      if (sig !== lastHeatmapSignature) {
        lastHeatmapSignature = sig;
        clearChildren(chunkHeatmap);
        if (totChunks <= 0) {
          chunkHeatmap.append(
            el("p", {
              className: "empty-hint",
              textContent:
                "Bản đồ tiến độ từng chunk 8s sẽ hiển thị ngay khi tập phim bắt đầu chạy.",
            })
          );
        } else {
          const activeLabel = activeChunk >= 0 ? `, đang xử lý phân đoạn #${activeChunk + 1}` : "";
          const pendingCount = Math.max(0, totChunks - doneSet.size - (activeChunk >= 0 ? 1 : 0));
          chunkHeatmap.append(
            el("p", {
              className: "visually-hidden",
              textContent: `Tổng ${totChunks} phân đoạn 8s: ${doneSet.size} hoàn tất${activeLabel}, ${pendingCount} chờ xử lý.`,
            })
          );
          for (let idx = 0; idx < totChunks; idx++) {
            const isDone = stage === "completed" || doneSet.has(idx);
            const isActive = !isDone && idx === activeChunk;
            const status = isDone ? "done" : isActive ? "active" : "pending";
            const symbol = isDone ? "✓" : isActive ? "▶" : "·";
            const perf = chunkPerfMap.get(idx);
            const tip = perf
              ? `Chunk #${idx + 1}: ${perf.frames} frames • AI ${perf.ai_fps} fps • ${perf.size_mb} MB`
              : `Chunk #${idx + 1} (${status})`;

            chunkHeatmap.append(
              el(
                "span",
                {
                  className: "chunk-cell",
                  dataset: { status },
                  title: tip,
                  "aria-hidden": "true",
                },
                [
                  el("span", { "aria-hidden": "true", textContent: symbol }),
                  String(idx + 1),
                ]
              )
            );
          }
        }
      }
    }

    // Resumable Checkpoints (.temp_*) — only rebuild DOM when checkpoint list changes so 1Hz ticks preserve focus/clicks
    if (checkpointList) {
      const cps = Array.isArray(pipeline.checkpoints) ? pipeline.checkpoints : [];
      const cpSig = cps
        .map((cp) => `${cp.name}|${cp.completed_chunks}|${cp.total_chunks ?? "?"}|${cp.size_mb}`)
        .join(";;");
      if (cpSig !== lastCheckpointSignature) {
        lastCheckpointSignature = cpSig;
        clearChildren(checkpointList);
        for (const cp of cps) {
          checkpointList.append(
            el("div", { className: "checkpoint-chip" }, [
              el("span", {
                textContent: `💾 Checkpoint ${cp.name}: Đã lưu ${cp.completed_chunks}/${cp.total_chunks || "?"} chunk (${cp.size_mb} MB)`,
              }),
              el("button", {
                type: "button",
                className: "btn btn-ghost",
                textContent: "Mở Tab Hàng đợi để Tiếp tục →",
                onClick: () => queueCtrl?.focusWithPath(""),
              }),
            ])
          );
        }
      }
    }
  };

  // REST Fallback & SSE Stream
  let fallbackTimer = null;

  const setSseBadge = (tone, text, symbol) => {
    if (!sseBadge || !sseBadgeText) return;
    sseBadge.dataset.tone = tone;
    sseBadgeText.textContent = text;
    const icon = sseBadge.querySelector(".pill-icon");
    if (icon) icon.textContent = symbol;
  };

  const applySnapshotLogs = (snap) => {
    if (snap.log_cursor && typeof snap.log_cursor === "object") {
      const nextPath = String(snap.log_cursor.path || "");
      if (nextPath && lastLogCursor.path && nextPath !== lastLogCursor.path) {
        logRing = [];
      }
      lastLogCursor = {
        path: nextPath,
        sig: String(snap.log_cursor.sig || ""),
        offset: Number(snap.log_cursor.offset || 0),
      };
    }
    if (Array.isArray(snap.logs)) {
      logRing = snap.logs.slice(-maxLogLines);
      renderLogs();
    }
  };

  const fetchSnapshotFallback = async () => {
    try {
      const snap = await apiFetch("/api/status");
      if (snap.gpu_info) gpuStaticInfo = snap.gpu_info;
      charts.setSnapshot(snap.history || [], snap.pipeline?.live || {});
      applyPipelineAndTelemetry(snap.telemetry, snap.pipeline);
      applySnapshotLogs(snap);
    } catch {
      // Server temporarily unreachable
    }
  };

  const startFallbackPolling = () => {
    if (fallbackTimer) return;
    fallbackTimer = window.setInterval(fetchSnapshotFallback, 2500);
  };

  const stopFallbackPolling = () => {
    if (fallbackTimer) {
      window.clearInterval(fallbackTimer);
      fallbackTimer = null;
    }
  };

  if (typeof EventSource === "function") {
    const es = new EventSource("/api/events");

    es.addEventListener("open", () => {
      stopFallbackPolling();
      setSseBadge("live", "Trực tiếp SSE 1.0 Hz", "●");
    });

    es.addEventListener("snapshot", (ev) => {
      try {
        const snap = JSON.parse(ev.data);
        if (snap.gpu_info) gpuStaticInfo = snap.gpu_info;
        charts.setSnapshot(snap.history || [], snap.pipeline?.live || {});
        applyPipelineAndTelemetry(snap.telemetry, snap.pipeline);
        applySnapshotLogs(snap);
      } catch {
        // Ignore malformed packet
      }
    });

    es.addEventListener("tick", (ev) => {
      try {
        const pkt = JSON.parse(ev.data);
        charts.pushTick(pkt.telemetry, pkt.pipeline?.live || {});
        applyPipelineAndTelemetry(pkt.telemetry, pkt.pipeline);
        const cur = pkt.log_cursor && typeof pkt.log_cursor === "object" ? pkt.log_cursor : null;
        const nextPath = cur ? String(cur.path || "") : "";
        const nextOffset = cur ? Number(cur.offset || 0) : 0;
        const switchedLogFile = Boolean(
          cur &&
            (cur.reset ||
              (nextPath && lastLogCursor.path && nextPath !== lastLogCursor.path) ||
              (nextPath && nextPath === lastLogCursor.path && nextOffset < lastLogCursor.offset))
        );
        if (switchedLogFile) {
          logRing = [];
          if (!Array.isArray(pkt.logs) || pkt.logs.length === 0) {
            renderLogs();
          }
        }
        const alreadySeen = Boolean(
          cur &&
            !switchedLogFile &&
            nextPath === lastLogCursor.path &&
            nextOffset <= lastLogCursor.offset
        );
        if (!alreadySeen && Array.isArray(pkt.logs) && pkt.logs.length > 0) {
          logRing.push(...pkt.logs);
          if (logRing.length > maxLogLines) {
            logRing = logRing.slice(-maxLogLines);
          }
          renderLogs();
        }
        if (cur) {
          lastLogCursor = {
            path: nextPath,
            sig: String(cur.sig || ""),
            offset: nextOffset,
          };
        }
      } catch {
        // Ignore malformed tick
      }
    });

    es.addEventListener("error", () => {
      setSseBadge("warn", "Mất SSE — Đang kết nối lại…", "◌");
      startFallbackPolling();
    });
  } else {
    setSseBadge("info", "Chế độ Polling 2.5s", "●");
    fetchSnapshotFallback();
    startFallbackPolling();
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrapStudio, { once: true });
} else {
  bootstrapStudio();
}
