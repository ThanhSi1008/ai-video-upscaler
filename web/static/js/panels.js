/**
 * web/static/js/panels.js
 * Controllers for:
 * - Tab 2: Queue Builder & Overnight ETA Scheduler
 * - Tab 4: SR Models Registry & Validated shows.yaml Configurator
 * - Tab 5: Subtrans Studio (normal 24/7 operation + 3-Key .env manager) & Mux Audit Vault
 */

import { openFsPicker } from "./fs_picker.js";
import {
  apiFetch,
  clearChildren,
  el,
  formatDuration,
  initDialogLightDismiss,
  showToast,
} from "./ui.js";

const QUEUE_STORAGE_KEY = "upscaler-studio-queue-inputs";
const QUEUE_CONFIG_STORAGE_KEY = "upscaler-studio-queue-config";

function setSelectValueSafely(selectEl, targetValue, fallbackValue = "") {
  if (!(selectEl instanceof HTMLSelectElement)) return;
  const candidate = targetValue == null ? "" : String(targetValue);
  const options = Array.from(selectEl.options);
  if (options.some((opt) => opt.value === candidate)) {
    selectEl.value = candidate;
    return;
  }
  const fb = fallbackValue == null ? "" : String(fallbackValue);
  if (options.some((opt) => opt.value === fb)) {
    selectEl.value = fb;
  } else if (options.length > 0) {
    selectEl.selectedIndex = 0;
  }
}

function setDesiredSelectValue(selectEl, targetValue, fallbackValue = "") {
  if (!(selectEl instanceof HTMLSelectElement)) return;
  selectEl.dataset.desiredValue = targetValue == null ? "" : String(targetValue);
  setSelectValueSafely(selectEl, targetValue, fallbackValue);
}

/* =====================================================================
 * TAB 2: QUEUE & OVERNIGHT SCHEDULER
 * ===================================================================== */
export function initQueuePanel({ selectTabById, onJobLaunched }) {
  const listEl = document.getElementById("queue-inputs-list");
  const manualInput = document.getElementById("manual-queue-path");
  const manualErrorEl = document.getElementById("manual-queue-path-error");
  const btnAddManual = document.getElementById("btn-add-manual-path");
  const btnOpenFs = document.getElementById("btn-open-fs-queue");
  const btnClear = document.getElementById("btn-clear-queue");
  const form = document.getElementById("upscale-config-form");
  const btnCalc = document.getElementById("btn-calc-schedule");
  const btnPreview10s = document.getElementById("btn-run-preview-10s");
  const btnStartUpscale = document.getElementById("btn-start-upscale");
  const summaryPill = document.getElementById("schedule-summary-pill");
  const scheduleTable = document.getElementById("schedule-table");
  const panelQueue = document.getElementById("panel-queue");
  const tbody = document.getElementById("schedule-tbody");

  const optModel = document.getElementById("opt-model");
  const optDedup = document.getElementById("opt-dedup");
  const optNvencMultipass = document.getElementById("opt-nvenc-multipass");
  const optGrainRestore = document.getElementById("opt-grain-restore");
  const optBatch = document.getElementById("opt-batch-size");
  const optBatchError = document.getElementById("opt-batch-size-error");
  const optDeadline = document.getElementById("opt-deadline");
  const optDeadlineError = document.getElementById("opt-deadline-error");
  const optOutDir = document.getElementById("opt-output-dir");
  const btnBrowseOutDir = document.getElementById("btn-browse-opt-outdir");
  const optSubsQuality = document.getElementById("opt-subs-quality");
  const optPreviewSeconds = document.getElementById("opt-preview-seconds");
  const optPreviewStart = document.getElementById("opt-preview-start");
  const optPreviewStartError = document.getElementById("opt-preview-start-error");
  const optZeroDisk = document.getElementById("opt-zero-disk");
  const optTensorrt = document.getElementById("opt-tensorrt");
  const optAllowPartial = document.getElementById("opt-allow-partial");
  const optSubsAfter = document.getElementById("opt-subs-after");
  const optAsyncSubs = document.getElementById("opt-async-subs");
  const optPreferWhisper = document.getElementById("opt-prefer-whisper");
  const optForce = document.getElementById("opt-force");
  const optDryRun = document.getElementById("opt-dry-run");

  let queueInputs = [];
  let queueErrorPaths = new Set();
  let calcTimer = null;
  let hasComputedInitial = false;
  let scheduleDirty = false;
  let scheduleSeq = 0;
  let systemBusy = false;
  let systemBusyReason = "";
  let localLaunchInFlight = false;

  const setFieldError = (inputEl, errorEl, msg) => {
    if (msg) {
      inputEl?.setAttribute("aria-invalid", "true");
      if (errorEl) {
        errorEl.textContent = msg;
        errorEl.hidden = false;
      }
    } else {
      inputEl?.removeAttribute("aria-invalid");
      if (errorEl) {
        errorEl.textContent = "";
        errorEl.hidden = true;
      }
    }
  };

  const syncStartUpscaleLabel = () => {
    if (!(btnStartUpscale instanceof HTMLButtonElement)) return;
    const subsOn = Boolean(optSubsAfter?.checked);
    clearChildren(btnStartUpscale);
    btnStartUpscale.append(
      el("span", { "aria-hidden": "true", textContent: "🚀" }),
      " ",
      subsOn ? "Bắt đầu Nâng cấp 4K + Dịch VI" : "Bắt đầu Nâng cấp 4K"
    );
    if (systemBusy) {
      btnStartUpscale.title = systemBusyReason || "Hệ thống đang có tiến trình chạy.";
    } else if (subsOn && new Date().getHours() >= 14) {
      btnStartUpscale.title =
        "Cảnh báo: Đang trong khung giờ cao điểm (>= 14:00 VN), dịch Gemini tự động sau khi upscale có thể chậm hoặc chạm hạn mức.";
    } else {
      btnStartUpscale.title = "";
    }
  };

  const getPreviewSeconds = () => {
    const n = Number(optPreviewSeconds?.value || 10);
    return Number.isFinite(n) && n >= 3 && n <= 60 ? Math.round(n) : 10;
  };

  const parsePreviewStartInput = ({ focusOnError = false, toastOnError = false } = {}) => {
    if (!(optPreviewStart instanceof HTMLInputElement)) return { ok: true, sec: null, label: "30% phim" };
    const raw = optPreviewStart.value.trim();
    if (!raw) {
      setFieldError(optPreviewStart, optPreviewStartError, null);
      return { ok: true, sec: null, label: "30% phim" };
    }
    let sec = Number.NaN;
    if (raw.includes(":")) {
      const parts = raw.split(":");
      if (
        (parts.length === 2 || parts.length === 3) &&
        parts.slice(0, -1).every((p) => /^\d+$/.test(p)) &&
        /^\d+(\.\d+)?$/.test(parts[parts.length - 1])
      ) {
        const nums = parts.map((p) => Number(p));
        const validParts =
          nums.every((n) => Number.isFinite(n) && n >= 0) &&
          (parts.length === 2 ? nums[1] < 60 : nums[1] < 60 && nums[2] < 60);
        if (validParts) {
          sec = parts.length === 2 ? nums[0] * 60 + nums[1] : nums[0] * 3600 + nums[1] * 60 + nums[2];
        }
      }
    } else if (/^\d+(\.\d+)?$/.test(raw)) {
      sec = Number(raw);
    }
    if (!Number.isFinite(sec) || sec < 0 || sec > 86400) {
      const msg = "Mốc bắt đầu Preview phải là số giây >= 0 hoặc MM:SS / HH:MM:SS (vd: 14:20 hoặc 860).";
      setFieldError(optPreviewStart, optPreviewStartError, msg);
      if (focusOnError) optPreviewStart.focus();
      if (toastOnError) showToast(msg, "warn");
      return { ok: false, sec: null, label: "30% phim" };
    }
    setFieldError(optPreviewStart, optPreviewStartError, null);
    return {
      ok: true,
      sec: Math.round(sec * 1000) / 1000,
      label: raw.includes(":") ? `từ ${raw}` : `từ ${Math.round(sec * 10) / 10}s`,
    };
  };

  const syncPreviewButtonLabel = () => {
    if (!(btnPreview10s instanceof HTMLButtonElement)) return;
    const sec = getPreviewSeconds();
    const startRes = parsePreviewStartInput();
    const posText = startRes.ok && startRes.sec !== null ? startRes.label : "30% phim";
    clearChildren(btnPreview10s);
    btnPreview10s.append(
      el("span", { "aria-hidden": "true", textContent: "🎬" }),
      ` Chạy Xem trước ${sec}s (${posText})`
    );
  };

  const syncStageBControls = () => {
    const subsOn = Boolean(optSubsAfter?.checked);
    for (const depEl of [optSubsQuality, optAsyncSubs, optPreferWhisper]) {
      if (!depEl) continue;
      depEl.disabled = !subsOn;
      depEl.closest(".field-group, .check-item")?.classList.toggle("is-disabled-dep", !subsOn);
    }
  };

  const syncQueueLaunchButtons = () => {
    const busyOrLaunching = systemBusy || localLaunchInFlight;
    if (btnPreview10s instanceof HTMLButtonElement) {
      btnPreview10s.disabled = busyOrLaunching;
      btnPreview10s.title = systemBusy ? systemBusyReason || "Hệ thống đang có tiến trình chạy." : "";
    }
    if (btnStartUpscale instanceof HTMLButtonElement) {
      btnStartUpscale.disabled = busyOrLaunching;
    }
    syncPreviewButtonLabel();
    syncStartUpscaleLabel();
  };

  const validateBatchSizeInput = ({ focusOnError = false, toastOnError = false } = {}) => {
    const rawBatch = optBatch?.value.trim() ?? "";
    const batchBadInput = optBatch instanceof HTMLInputElement && optBatch.validity.badInput;
    const isInt = !batchBadInput && (rawBatch === "" || /^[+-]?\d+$/.test(rawBatch));
    const batchVal = !batchBadInput && rawBatch === "" ? 8 : isInt ? Number(rawBatch) : Number.NaN;
    if (!Number.isFinite(batchVal) || !Number.isInteger(batchVal) || batchVal < 1 || batchVal > 32) {
      const msg = "Batch Size phải là số nguyên từ 1 đến 32.";
      if (optBatch instanceof HTMLInputElement) {
        optBatch.setCustomValidity(msg);
      }
      setFieldError(optBatch, optBatchError, msg);
      if (focusOnError) optBatch?.focus();
      if (toastOnError) showToast(msg, "warn");
      return null;
    }
    if (optBatch instanceof HTMLInputElement) {
      optBatch.setCustomValidity("");
    }
    setFieldError(optBatch, optBatchError, null);
    return batchVal;
  };

  const validateDeadlineInput = () => {
    if (!(optDeadline instanceof HTMLInputElement)) return true;
    const val = optDeadline.value.trim();
    if (!val) {
      setFieldError(optDeadline, optDeadlineError, null);
      return true;
    }
    if (!optDeadline.checkValidity()) {
      setFieldError(
        optDeadline,
        optDeadlineError,
        "Định dạng giờ hạn chót phải là HH:MM (00:00 đến 23:59, ví dụ 07:00)."
      );
      return false;
    }
    setFieldError(optDeadline, optDeadlineError, null);
    return true;
  };

  try {
    const saved = JSON.parse(localStorage.getItem(QUEUE_STORAGE_KEY) || "[]");
    if (Array.isArray(saved)) {
      queueInputs = saved.filter((x) => typeof x === "string" && x.trim().length > 0);
    }
  } catch {
    queueInputs = [];
  }

  const persistInputs = () => {
    try {
      localStorage.setItem(QUEUE_STORAGE_KEY, JSON.stringify(queueInputs));
    } catch {
      // Ignore storage quota errors
    }
  };

  const persistQueueConfig = () => {
    try {
      const cfg = {
        model: optModel?.dataset?.desiredValue ?? optModel?.value ?? "",
        dedup_policy: optDedup?.dataset?.desiredValue ?? optDedup?.value ?? "",
        nvenc_multipass: optNvencMultipass?.dataset?.desiredValue ?? optNvencMultipass?.value ?? "auto",
        grain_restore: optGrainRestore?.value ?? "auto",
        batch_size: optBatch?.value.trim() || "8",
        deadline: optDeadline?.value.trim() || "",
        output_dir: optOutDir?.value.trim() || "",
        subs_quality: optSubsQuality?.value || "normal",
        preview_seconds: String(getPreviewSeconds()),
        preview_start: optPreviewStart?.value.trim() || "",
        zero_disk: Boolean(optZeroDisk?.checked ?? true),
        use_tensorrt: Boolean(optTensorrt?.checked ?? true),
        allow_partial: Boolean(optAllowPartial?.checked),
        subs_after_upscale: Boolean(optSubsAfter?.checked),
        async_subs: Boolean(optAsyncSubs?.checked),
        prefer_whisper: Boolean(optPreferWhisper?.checked),
      };
      localStorage.setItem(QUEUE_CONFIG_STORAGE_KEY, JSON.stringify(cfg));
    } catch {
      // Ignore storage quota errors
    }
  };

  const restoreQueueConfig = () => {
    try {
      const raw = localStorage.getItem(QUEUE_CONFIG_STORAGE_KEY);
      if (!raw) return;
      const cfg = JSON.parse(raw);
      if (!cfg || typeof cfg !== "object") return;
      if (typeof cfg.model === "string") setDesiredSelectValue(optModel, cfg.model, "");
      if (typeof cfg.dedup_policy === "string") setDesiredSelectValue(optDedup, cfg.dedup_policy, "");
      if (typeof cfg.nvenc_multipass === "string") {
        setDesiredSelectValue(optNvencMultipass, cfg.nvenc_multipass, "auto");
      }
      if (typeof cfg.grain_restore === "string") {
        setSelectValueSafely(optGrainRestore, cfg.grain_restore, "auto");
      }
      if (typeof cfg.batch_size === "string" && optBatch instanceof HTMLInputElement) {
        const bNum = Number(cfg.batch_size);
        if (Number.isInteger(bNum) && bNum >= 1 && bNum <= 32) {
          optBatch.value = String(bNum);
        }
      }
      if (typeof cfg.deadline === "string" && optDeadline instanceof HTMLInputElement) {
        optDeadline.value = cfg.deadline;
      }
      if (typeof cfg.output_dir === "string" && optOutDir instanceof HTMLInputElement) {
        optOutDir.value = cfg.output_dir;
      }
      if (typeof cfg.subs_quality === "string") {
        setSelectValueSafely(optSubsQuality, cfg.subs_quality, "normal");
      }
      if (typeof cfg.preview_seconds === "string") {
        setSelectValueSafely(optPreviewSeconds, cfg.preview_seconds, "10");
      }
      if (typeof cfg.preview_start === "string" && optPreviewStart instanceof HTMLInputElement) {
        optPreviewStart.value = cfg.preview_start;
      }
      if (typeof cfg.zero_disk === "boolean" && optZeroDisk instanceof HTMLInputElement) {
        optZeroDisk.checked = cfg.zero_disk;
      }
      if (typeof cfg.use_tensorrt === "boolean" && optTensorrt instanceof HTMLInputElement) {
        optTensorrt.checked = cfg.use_tensorrt;
      }
      if (typeof cfg.allow_partial === "boolean" && optAllowPartial instanceof HTMLInputElement) {
        optAllowPartial.checked = cfg.allow_partial;
      }
      if (typeof cfg.subs_after_upscale === "boolean" && optSubsAfter instanceof HTMLInputElement) {
        optSubsAfter.checked = cfg.subs_after_upscale;
      }
      if (typeof cfg.async_subs === "boolean" && optAsyncSubs instanceof HTMLInputElement) {
        optAsyncSubs.checked = cfg.async_subs;
      }
      if (typeof cfg.prefer_whisper === "boolean" && optPreferWhisper instanceof HTMLInputElement) {
        optPreferWhisper.checked = cfg.prefer_whisper;
      }
    } catch {
      // Ignore malformed stored state
    }
  };

  restoreQueueConfig();
  if (optForce instanceof HTMLInputElement) optForce.checked = false;
  if (optDryRun instanceof HTMLInputElement) optDryRun.checked = false;

  window.addEventListener("studio:models-loaded", () => {
    let changed = false;
    for (const [selEl, fb] of [
      [optModel, ""],
      [optDedup, ""],
      [optNvencMultipass, "auto"],
    ]) {
      if (!(selEl instanceof HTMLSelectElement)) continue;
      const desired = selEl.dataset.desiredValue;
      if (desired !== undefined) {
        const before = selEl.value;
        setSelectValueSafely(selEl, desired, fb);
        if (selEl.value !== before) changed = true;
      }
    }
    if (changed && hasComputedInitial && queueInputs.length > 0) {
      if (panelQueue && panelQueue.hasAttribute("hidden")) {
        scheduleDirty = true;
      } else {
        schedulePreviewDebounced();
      }
    }
  });

  const addInputs = (paths) => {
    let added = 0;
    for (const raw of paths) {
      const cleaned = String(raw || "").trim();
      if (!cleaned) continue;
      const norm = cleaned.toLowerCase();
      if (!queueInputs.some((existing) => existing.toLowerCase() === norm)) {
        queueInputs.push(cleaned);
        added++;
      }
    }
    if (added > 0) {
      persistInputs();
      renderInputsList();
      schedulePreviewDebounced();
    }
    return added;
  };

  const removeInputAt = (index) => {
    const removed = queueInputs.splice(index, 1);
    for (const r of removed) {
      queueErrorPaths.delete(String(r).toLowerCase());
    }
    persistInputs();
    renderInputsList();
    schedulePreviewDebounced();
  };

  const renderInputsList = () => {
    if (!listEl) return;
    clearChildren(listEl);
    if (queueInputs.length === 0) {
      listEl.append(
        el("li", {
          className: "empty-hint",
          textContent:
            "Chưa có file hay thư mục nào. Nhấn “Duyệt Ổ đĩa & Thư mục” để chọn nhanh các tập anime.",
        })
      );
      return;
    }

    queueInputs.forEach((p, idx) => {
      const removeBtn = el("button", {
        type: "button",
        className: "btn btn-ghost",
        "aria-label": `Xóa ${p} khỏi hàng đợi`,
        textContent: "✕",
        onClick: () => removeInputAt(idx),
      });
      const isErr = queueErrorPaths.has(String(p).toLowerCase());
      const li = el(
        "li",
        {
          className: "selected-input-item",
          ...(isErr ? { dataset: { error: "true" } } : {}),
        },
        [el("code", { textContent: p }), removeBtn]
      );
      listEl.append(li);
    });
  };

  const renderEmptySchedule = (msg) => {
    if (!tbody) return;
    clearChildren(tbody);
    tbody.append(
      el("tr", {}, [
        el("th", { scope: "row", textContent: "—" }),
        el("td", {
          colspan: "7",
          textContent: msg,
        }),
      ])
    );
  };

  const computeScheduleNow = async () => {
    hasComputedInitial = true;
    scheduleDirty = false;
    if (queueInputs.length === 0) {
      if (queueErrorPaths.size > 0) {
        queueErrorPaths.clear();
        renderInputsList();
      }
      if (summaryPill) {
        summaryPill.classList.remove("text-error");
        summaryPill.textContent = "Chọn tập phim để xem dự báo tự động theo shows.yaml";
      }
      renderEmptySchedule("Chưa có dữ liệu dự báo. Hãy thêm ít nhất 1 file hoặc thư mục ở mục 1.");
      return;
    }

    const batchVal = validateBatchSizeInput();
    if (batchVal === null) {
      if (summaryPill) {
        summaryPill.classList.add("text-error");
        summaryPill.textContent = "Batch Size không hợp lệ (cần số nguyên từ 1 đến 32).";
      }
      return;
    }

    if (!validateDeadlineInput()) {
      if (summaryPill) {
        summaryPill.classList.add("text-error");
        summaryPill.textContent = "Định dạng hạn chót không hợp lệ (cần HH:MM, ví dụ 07:00).";
      }
      return;
    }

    const reqId = ++scheduleSeq;
    if (summaryPill) {
      summaryPill.classList.remove("text-error");
      summaryPill.textContent = "Đang tính toán lịch…";
    }
    scheduleTable?.setAttribute("aria-busy", "true");

    try {
      const payload = {
        inputs: queueInputs,
        deadline: optDeadline?.value.trim() || null,
        allow_partial: Boolean(optAllowPartial?.checked),
        model: optModel?.value || null,
        dedup_policy: optDedup?.value || null,
        grain_restore: optGrainRestore?.value || "auto",
        batch_size: batchVal,
        zero_disk: Boolean(optZeroDisk?.checked ?? true),
        use_tensorrt: Boolean(optTensorrt?.checked ?? true),
        subs_after_upscale: Boolean(optSubsAfter?.checked),
        output_dir: optOutDir?.value.trim() || null,
        force: Boolean(optForce?.checked),
      };

      const data = await apiFetch("/api/queue/preview-schedule", {
        method: "POST",
        body: payload,
      });
      if (reqId !== scheduleSeq) return;

      if (queueErrorPaths.size > 0) {
        queueErrorPaths.clear();
        renderInputsList();
      }

      if (summaryPill) {
        summaryPill.classList.remove("text-error");
        const dlText = data.deadline_str ? ` • Deadline ${data.deadline_str}` : "";
        summaryPill.textContent =
          `Tổng ${data.total_episodes} tập: Sẽ chạy ${data.accepted_count} | ` +
          `Đã có 4K: ${data.already_done_count} | Quá hạn: ${data.skipped_count} • ` +
          `Tổng ETA ~${data.total_eta_minutes} phút${dlText}`;
      }

      if (!tbody) return;
      clearChildren(tbody);
      const episodes = Array.isArray(data.episodes) ? data.episodes : [];
      if (episodes.length === 0) {
        renderEmptySchedule("Không tìm thấy tập video nguồn nào trong đường dẫn đã chọn.");
        return;
      }

      for (const ep of episodes) {
        let badgeTone = "badge-ok";
        if (ep.already_done || ep.is_partial) badgeTone = "badge-warn";
        else if (!ep.can_run) badgeTone = "badge-err";

        const tr = el("tr", {}, [
          el("th", { scope: "row", textContent: `${ep.index}. ${ep.file_name}` }),
          el("td", {}, [
            el("strong", { textContent: ep.model_id }),
            el("div", { className: "muted-meta", textContent: ep.routing_source }),
          ]),
          el("td", { textContent: ep.dedup_policy }),
          el("td", {
            textContent: `${ep.n_unique.toLocaleString("vi-VN")} / ${ep.total_frames.toLocaleString("vi-VN")} (tiết kiệm ${ep.dedup_ratio_percent}% dedup)`,
          }),
          el("td", { textContent: `${ep.ref_fps} fps` }),
          el("td", { textContent: ep.already_done ? "0 phút" : `${ep.eta_minutes} phút` }),
          el("td", { textContent: `${ep.start_time} → ${ep.finish_time}` }),
          el("td", {}, [el("span", { className: `badge ${badgeTone}`, textContent: ep.decision })]),
        ]);
        tbody.append(tr);
      }
    } catch (err) {
      if (reqId !== scheduleSeq) return;
      if (summaryPill) {
        summaryPill.classList.add("text-error");
        summaryPill.textContent = `Lỗi dự báo lịch trình: ${err.message}`;
      }
      const errLow = String(err.message || "").toLowerCase();
      queueErrorPaths.clear();
      for (const p of queueInputs) {
        if (errLow.includes(String(p).toLowerCase())) {
          queueErrorPaths.add(String(p).toLowerCase());
        }
      }
      if (queueErrorPaths.size === 0 && queueInputs.length === 1) {
        queueErrorPaths.add(String(queueInputs[0]).toLowerCase());
      }
      renderInputsList();
      renderEmptySchedule(`Không thể tính dự báo: ${err.message}`);
      showToast(err.message, "error");
    } finally {
      if (reqId === scheduleSeq) {
        scheduleTable?.removeAttribute("aria-busy");
      }
    }
  };

  const schedulePreviewDebounced = () => {
    if (calcTimer) window.clearTimeout(calcTimer);
    calcTimer = window.setTimeout(computeScheduleNow, 220);
  };

  btnOpenFs?.addEventListener("click", async () => {
    const picked = await openFsPicker({
      title: "Chọn các Tập Video hoặc Thư mục Bộ Phim",
      multi: true,
      allowFolders: true,
    });
    const added = addInputs(picked);
    if (added > 0) {
      showToast(`Đã thêm ${added} mục vào danh sách hàng đợi.`, "success", 3000);
    }
  });

  const handleManualAdd = async () => {
    if (!manualInput) return;
    let val = manualInput.value.trim();
    if (val.length >= 2 && val[0] === val[val.length - 1] && (val[0] === '"' || val[0] === "'")) {
      val = val.slice(1, -1).trim();
    }
    if (!val) return;
    if (/^(\\\\|\/\/)/.test(val) || val.includes("\0")) {
      const msg = "Không hỗ trợ đường dẫn mạng UNC (\\\\server\\share) hoặc ký tự không hợp lệ.";
      setFieldError(manualInput, manualErrorEl, msg);
      manualInput.focus();
      showToast(msg, "error");
      return;
    }
    const norm = val.toLowerCase();
    if (queueInputs.some((existing) => existing.toLowerCase() === norm)) {
      setFieldError(manualInput, manualErrorEl, null);
      showToast("Đường dẫn này đã có trong hàng đợi.", "info", 2400);
      manualInput.value = "";
      return;
    }

    if (btnAddManual instanceof HTMLButtonElement) btnAddManual.disabled = true;
    manualInput.setAttribute("aria-busy", "true");
    try {
      await apiFetch("/api/queue/preview-schedule", {
        method: "POST",
        body: {
          inputs: [val],
          deadline: null,
          batch_size: 8,
        },
      });
      setFieldError(manualInput, manualErrorEl, null);
      addInputs([val]);
      manualInput.value = "";
    } catch (err) {
      setFieldError(manualInput, manualErrorEl, err.message);
      manualInput.focus();
      showToast(err.message, "error");
    } finally {
      manualInput.removeAttribute("aria-busy");
      if (btnAddManual instanceof HTMLButtonElement) btnAddManual.disabled = false;
    }
  };

  btnAddManual?.addEventListener("click", handleManualAdd);
  manualInput?.addEventListener("input", () => setFieldError(manualInput, manualErrorEl, null));
  manualInput?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      if (ev.isComposing || ev.keyCode === 229) return;
      ev.preventDefault();
      handleManualAdd();
    }
  });

  btnClear?.addEventListener("click", () => {
    queueInputs = [];
    queueErrorPaths.clear();
    setFieldError(manualInput, manualErrorEl, null);
    persistInputs();
    renderInputsList();
    computeScheduleNow();
  });

  btnCalc?.addEventListener("click", computeScheduleNow);

  btnBrowseOutDir?.addEventListener("click", async () => {
    const picked = await openFsPicker({
      title: "Chọn Thư mục Xuất (-o)",
      multi: false,
      allowFolders: true,
      foldersOnly: true,
      initialPath: optOutDir?.value.trim() || "out",
    });
    if (picked.length > 0 && optOutDir) {
      optOutDir.value = picked[0];
      persistQueueConfig();
      schedulePreviewDebounced();
    }
  });

  optDeadline?.addEventListener("input", () => {
    const val = optDeadline.value.trim();
    if (!val || optDeadline.checkValidity()) {
      setFieldError(optDeadline, optDeadlineError, null);
    }
  });
  optDeadline?.addEventListener("blur", () => {
    validateDeadlineInput();
  });
  optBatch?.addEventListener("input", () => {
    if (optBatch instanceof HTMLInputElement) optBatch.setCustomValidity("");
    setFieldError(optBatch, optBatchError, null);
  });
  optBatch?.addEventListener("blur", () => {
    validateBatchSizeInput();
  });
  optSubsAfter?.addEventListener("change", () => {
    syncStageBControls();
    syncStartUpscaleLabel();
  });
  optPreviewSeconds?.addEventListener("change", () => {
    syncPreviewButtonLabel();
    persistQueueConfig();
  });
  optPreviewStart?.addEventListener("input", () => {
    setFieldError(optPreviewStart, optPreviewStartError, null);
    syncPreviewButtonLabel();
  });
  optPreviewStart?.addEventListener("blur", () => {
    parsePreviewStartInput();
    syncPreviewButtonLabel();
    persistQueueConfig();
  });

  for (const ctrl of [
    optModel,
    optDedup,
    optGrainRestore,
    optBatch,
    optDeadline,
    optOutDir,
    optZeroDisk,
    optTensorrt,
    optAllowPartial,
    optSubsAfter,
    optForce,
  ]) {
    ctrl?.addEventListener("change", () => {
      if (ctrl === optModel && optModel instanceof HTMLSelectElement) {
        optModel.dataset.desiredValue = optModel.value;
      } else if (ctrl === optDedup && optDedup instanceof HTMLSelectElement) {
        optDedup.dataset.desiredValue = optDedup.value;
      }
      if (ctrl !== optForce) {
        persistQueueConfig();
      }
      schedulePreviewDebounced();
    });
  }

  for (const extraCtrl of [optNvencMultipass, optSubsQuality, optAsyncSubs, optPreferWhisper]) {
    extraCtrl?.addEventListener("change", () => {
      if (extraCtrl === optNvencMultipass && optNvencMultipass instanceof HTMLSelectElement) {
        optNvencMultipass.dataset.desiredValue = optNvencMultipass.value;
      }
      persistQueueConfig();
    });
  }

  btnPreview10s?.addEventListener("click", async () => {
    if (systemBusy) {
      showToast(systemBusyReason || "Hệ thống đang có tiến trình chạy.", "warn");
      return;
    }
    const previewSec = getPreviewSeconds();
    if (queueInputs.length === 0) {
      showToast(`Hãy chọn ít nhất 1 file video hoặc thư mục để chạy Xem trước ${previewSec}s.`, "warn");
      return;
    }
    const batchVal = validateBatchSizeInput({ focusOnError: true, toastOnError: true });
    if (batchVal === null) return;
    const startParsed = parsePreviewStartInput({ focusOnError: true, toastOnError: true });
    if (!startParsed.ok) return;

    localLaunchInFlight = true;
    syncQueueLaunchButtons();
    try {
      const res = await apiFetch("/api/jobs/preview", {
        method: "POST",
        body: {
          video_path: queueInputs[0],
          seconds: previewSec,
          preview_start_sec: startParsed.sec,
          output_dir: optOutDir?.value.trim() || null,
          model: optModel?.value || null,
          dedup_policy: optDedup?.value || null,
          nvenc_multipass: optNvencMultipass?.value || "auto",
          grain_restore: optGrainRestore?.value || "auto",
          batch_size: batchVal,
          zero_disk: Boolean(optZeroDisk?.checked ?? true),
          use_tensorrt: Boolean(optTensorrt?.checked ?? true),
          force: Boolean(optForce?.checked),
          auto_open_player: false,
        },
      });
      showToast(`Đã khởi chạy Xem trước ${previewSec}s (PID=${res.job.pid}).`, "success");
      selectTabById?.("tab-monitor");
      onJobLaunched?.(res.job);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      localLaunchInFlight = false;
      syncQueueLaunchButtons();
    }
  });

  form?.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (systemBusy) {
      showToast(systemBusyReason || "Hệ thống đang có tiến trình chạy.", "warn");
      return;
    }
    if (queueInputs.length === 0) {
      showToast("Vui lòng chọn ít nhất 1 video hoặc thư mục nguồn trước khi bắt đầu.", "warn");
      return;
    }
    const batchVal = validateBatchSizeInput({ focusOnError: true, toastOnError: true });
    if (batchVal === null) return;
    if (!validateDeadlineInput()) {
      optDeadline?.focus();
      showToast("Định dạng giờ hạn chót phải là HH:MM (ví dụ 07:00).", "warn");
      return;
    }

    localLaunchInFlight = true;
    syncQueueLaunchButtons();
    try {
      const payload = {
        inputs: queueInputs,
        output_dir: optOutDir?.value.trim() || null,
        model: optModel?.value || null,
        dedup_policy: optDedup?.value || null,
        nvenc_multipass: optNvencMultipass?.value || "auto",
        grain_restore: optGrainRestore?.value || "auto",
        batch_size: batchVal,
        deadline: optDeadline?.value.trim() || null,
        allow_partial: Boolean(optAllowPartial?.checked),
        zero_disk: Boolean(optZeroDisk?.checked ?? true),
        use_tensorrt: Boolean(optTensorrt?.checked ?? true),
        subs_after_upscale: Boolean(optSubsAfter?.checked),
        subs_quality: optSubsQuality?.value || "normal",
        async_subs: Boolean(optAsyncSubs?.checked),
        prefer_whisper: Boolean(optPreferWhisper?.checked),
        auto_align_en: true,
        sub_offset_ms: 0,
        keep_temp: false,
        force: Boolean(optForce?.checked),
        dry_run: Boolean(optDryRun?.checked),
      };

      const res = await apiFetch("/api/jobs/start", {
        method: "POST",
        body: payload,
      });
      showToast(`Đã khởi chạy tiến trình nâng cấp 4K (PID=${res.job.pid}).`, "success");
      selectTabById?.("tab-monitor");
      onJobLaunched?.(res.job);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      localLaunchInFlight = false;
      syncQueueLaunchButtons();
    }
  });

  const applyCheckpointOptions = (opts, { replaceQueue = false } = {}) => {
    const isObj = typeof opts === "object" && opts !== null;
    const sourceFile = isObj
      ? String(opts.sourceFile ?? opts.source_file ?? "").trim()
      : String(opts || "").trim();
    const outputDir = isObj ? String(opts.outputDir ?? opts.output_dir ?? "").trim() : "";
    const isCustomOut = isObj
      ? typeof (opts.isCustomOutputDir ?? opts.is_custom_output_dir) === "boolean"
        ? Boolean(opts.isCustomOutputDir ?? opts.is_custom_output_dir)
        : Boolean(outputDir && !/[\\/]out$/i.test(outputDir))
      : false;

    let cfgChanged = false;
    if (isObj) {
      if (optOutDir instanceof HTMLInputElement) {
        const nextOutDir = isCustomOut ? outputDir : "";
        if (optOutDir.value.trim() !== nextOutDir) {
          optOutDir.value = nextOutDir;
          cfgChanged = true;
        }
      }
      const cpModel = String(opts.modelId ?? opts.model_id ?? "").trim();
      if (cpModel) {
        setDesiredSelectValue(optModel, cpModel, "");
        cfgChanged = true;
      }
      const cpDedup = String(
        opts.dedupPolicy ?? opts.dedup_policy_requested ?? opts.dedup_policy ?? ""
      ).trim();
      if (cpDedup) {
        setDesiredSelectValue(optDedup, cpDedup, "");
        cfgChanged = true;
      }
      const cpGrain = String(opts.grainRestore ?? opts.grain_restore ?? "").trim();
      if (cpGrain) {
        setSelectValueSafely(optGrainRestore, cpGrain, "auto");
        cfgChanged = true;
      }
      const cpMp = String(opts.nvencMultipass ?? opts.nvenc_multipass_policy ?? "").trim();
      if (cpMp) {
        setDesiredSelectValue(optNvencMultipass, cpMp, "auto");
        cfgChanged = true;
      }
      const cpZeroDisk = opts.zeroDisk ?? opts.zero_disk;
      if (typeof cpZeroDisk === "boolean" && optZeroDisk instanceof HTMLInputElement) {
        optZeroDisk.checked = cpZeroDisk;
        cfgChanged = true;
      }
      const cpTrt = opts.useTensorrt ?? opts.use_tensorrt;
      if (typeof cpTrt === "boolean" && optTensorrt instanceof HTMLInputElement) {
        optTensorrt.checked = cpTrt;
        cfgChanged = true;
      }
      const cpSubsAfter = opts.subsAfterUpscale ?? opts.subs_after_upscale;
      if (typeof cpSubsAfter === "boolean" && optSubsAfter instanceof HTMLInputElement) {
        optSubsAfter.checked = cpSubsAfter;
        cfgChanged = true;
      }
      const cpSubsQual = String(opts.subsQuality ?? opts.subs_quality ?? "").trim();
      if (cpSubsQual) {
        setSelectValueSafely(optSubsQuality, cpSubsQual, "normal");
        cfgChanged = true;
      }
      const cpWhisper = opts.preferWhisper ?? opts.prefer_whisper;
      if (typeof cpWhisper === "boolean" && optPreferWhisper instanceof HTMLInputElement) {
        optPreferWhisper.checked = cpWhisper;
        cfgChanged = true;
      }
      const cpAsyncSubs = opts.asyncSubs ?? opts.async_subs;
      if (typeof cpAsyncSubs === "boolean" && optAsyncSubs instanceof HTMLInputElement) {
        optAsyncSubs.checked = cpAsyncSubs;
        cfgChanged = true;
      }
      syncStageBControls();
      syncStartUpscaleLabel();
      const rawCpBatch = opts.batchSize ?? opts.batch_size;
      const cpBatch = rawCpBatch === null || rawCpBatch === undefined || rawCpBatch === "" ? Number.NaN : Number(rawCpBatch);
      if (
        Number.isInteger(cpBatch) &&
        cpBatch >= 1 &&
        cpBatch <= 32 &&
        optBatch instanceof HTMLInputElement
      ) {
        optBatch.value = String(cpBatch);
        optBatch.setCustomValidity("");
        setFieldError(optBatch, optBatchError, null);
        cfgChanged = true;
      } else if (
        optBatch instanceof HTMLInputElement &&
        validateBatchSizeInput({ focusOnError: false, toastOnError: false }) === null
      ) {
        optBatch.value = "8";
        optBatch.setCustomValidity("");
        setFieldError(optBatch, optBatchError, null);
        cfgChanged = true;
      }
      if (optForce instanceof HTMLInputElement && optForce.checked) {
        optForce.checked = false;
        cfgChanged = true;
      }
      if (optDryRun instanceof HTMLInputElement && optDryRun.checked) {
        optDryRun.checked = false;
      }
      if (cfgChanged) {
        persistQueueConfig();
      }
    }

    if (sourceFile) {
      if (replaceQueue) {
        queueInputs = [sourceFile];
        queueErrorPaths.clear();
        persistInputs();
        renderInputsList();
        scheduleDirty = true;
      } else {
        const added = addInputs([sourceFile]);
        if (added === 0 && cfgChanged) {
          scheduleDirty = true;
          if (panelQueue && !panelQueue.hasAttribute("hidden")) {
            schedulePreviewDebounced();
          }
        }
      }
    }
    return { sourceFile, isObj };
  };

  renderInputsList();
  syncStageBControls();
  syncQueueLaunchButtons();
  if (queueInputs.length > 0 && summaryPill) {
    summaryPill.textContent = `Đã khôi phục ${queueInputs.length} mục đã chọn — mở tab Hàng đợi để tính lịch trình.`;
  }

  return {
    onTabActivated() {
      if (queueInputs.length > 0 && (!hasComputedInitial || scheduleDirty)) {
        computeScheduleNow();
      }
    },
    refreshSchedule() {
      if (panelQueue && panelQueue.hasAttribute("hidden")) {
        scheduleDirty = true;
        return;
      }
      computeScheduleNow();
    },
    setSystemBusy(isBusy, reason = "") {
      const nextBusy = Boolean(isBusy);
      const nextReason = String(reason || "");
      if (systemBusy === nextBusy && systemBusyReason === nextReason) return;
      systemBusy = nextBusy;
      systemBusyReason = nextReason;
      syncQueueLaunchButtons();
    },
    focusWithPath(opts) {
      const { sourceFile, isObj } = applyCheckpointOptions(opts, { replaceQueue: false });
      if (!sourceFile && isObj) {
        showToast(
          "Checkpoint cũ thiếu đường dẫn tệp nguồn đầy đủ — vui lòng chọn lại file nguồn.",
          "warn"
        );
      }
      selectTabById?.("tab-queue");
    },
    async resumeCheckpointNow(cp) {
      if (systemBusy || localLaunchInFlight) {
        showToast(systemBusyReason || "Hệ thống đang có tiến trình chạy.", "warn");
        return;
      }
      const { sourceFile } = applyCheckpointOptions(cp, { replaceQueue: true });
      if (!sourceFile) {
        showToast(
          "Checkpoint cũ thiếu đường dẫn tệp nguồn đầy đủ — vui lòng chọn lại file nguồn ở Tab 2.",
          "warn"
        );
        selectTabById?.("tab-queue");
        return;
      }
      const batchVal = validateBatchSizeInput({ focusOnError: false, toastOnError: true }) ?? 8;
      localLaunchInFlight = true;
      syncQueueLaunchButtons();
      try {
        const payload = {
          inputs: [sourceFile],
          output_dir: optOutDir?.value.trim() || null,
          model: (optModel?.dataset?.desiredValue ?? optModel?.value) || null,
          dedup_policy: (optDedup?.dataset?.desiredValue ?? optDedup?.value) || null,
          nvenc_multipass: (optNvencMultipass?.dataset?.desiredValue ?? optNvencMultipass?.value) || "auto",
          grain_restore: optGrainRestore?.value || "auto",
          batch_size: batchVal,
          deadline: null,
          allow_partial: false,
          zero_disk: Boolean(optZeroDisk?.checked ?? true),
          use_tensorrt: Boolean(optTensorrt?.checked ?? true),
          subs_after_upscale:
            typeof (cp?.subsAfterUpscale ?? cp?.subs_after_upscale) === "boolean"
              ? Boolean(cp?.subsAfterUpscale ?? cp?.subs_after_upscale)
              : false,
          subs_quality: optSubsQuality?.value || "normal",
          async_subs: Boolean(optAsyncSubs?.checked),
          prefer_whisper: Boolean(optPreferWhisper?.checked),
          auto_align_en: true,
          sub_offset_ms: 0,
          keep_temp: false,
          force: false,
          dry_run: false,
        };
        const res = await apiFetch("/api/jobs/start", {
          method: "POST",
          body: payload,
        });
        showToast(
          `Đã tiếp tục nâng cấp từ checkpoint chunk ${Number(cp?.completed_chunks || 0)}/${Number(cp?.total_chunks || 0)} (PID=${res.job.pid}).`,
          "success"
        );
        selectTabById?.("tab-monitor");
        onJobLaunched?.(res.job);
      } catch (err) {
        showToast(err.message, "error");
      } finally {
        localLaunchInFlight = false;
        syncQueueLaunchButtons();
      }
    },
  };
}

/* =====================================================================
 * TAB 4: MODELS REGISTRY & SHOWS.YAML CONFIGURATOR
 * ===================================================================== */
export function initModelsPanel({ onShowsUpdated } = {}) {
  const tbody = document.getElementById("models-tbody");
  const btnRefresh = document.getElementById("btn-refresh-models");
  const testInput = document.getElementById("test-match-input");
  const btnTestMatch = document.getElementById("btn-test-match");
  const testMatchCard = document.getElementById("test-match-result");
  const resModel = document.getElementById("match-res-model");
  const resDedup = document.getElementById("match-res-dedup");
  const resSource = document.getElementById("match-res-source");
  const resFps = document.getElementById("match-res-fps");

  const editor = document.getElementById("shows-yaml-editor");
  const showsErrorEl = document.getElementById("shows-yaml-error");
  const btnReloadShows = document.getElementById("btn-reload-shows");
  const btnSaveShows = document.getElementById("btn-save-shows");

  const optModelSelect = document.getElementById("opt-model");
  const optDedupSelect = document.getElementById("opt-dedup");
  const optNvencMultipassSelect = document.getElementById("opt-nvenc-multipass");

  let editorDirty = false;
  let modelsSystemBusy = false;
  let modelsSystemBusyReason = "";

  const syncShowsSaveButton = () => {
    if (!(btnSaveShows instanceof HTMLButtonElement)) return;
    btnSaveShows.disabled = modelsSystemBusy;
    btnSaveShows.title = modelsSystemBusy
      ? modelsSystemBusyReason || "Đang có tiến trình chạy — tạm khóa lưu shows.yaml để tránh lệch cấu hình giữa ca."
      : "";
  };

  const setShowsYamlError = (errMsg) => {
    if (errMsg) {
      editor?.setAttribute("aria-invalid", "true");
      if (showsErrorEl) {
        showsErrorEl.textContent = errMsg;
        showsErrorEl.hidden = false;
      }
      testMatchCard?.setAttribute("data-state", "error");
      if (resModel) resModel.textContent = "Lỗi shows.yaml";
      if (resDedup) resDedup.textContent = "—";
      if (resSource) resSource.textContent = errMsg;
      if (resFps) resFps.textContent = "—";
    } else {
      editor?.removeAttribute("aria-invalid");
      if (showsErrorEl) {
        showsErrorEl.textContent = "";
        showsErrorEl.hidden = true;
      }
      testMatchCard?.removeAttribute("data-state");
    }
  };

  editor?.addEventListener("input", () => {
    editorDirty = true;
    setShowsYamlError(null);
  });

  const runTestMatch = async ({ showUserWarning = false } = {}) => {
    const sample = testInput?.value.trim() || "";
    if (!sample) {
      if (showUserWarning) {
        testInput?.focus();
        showToast("Vui lòng nhập tên file hoặc đường dẫn mẫu để kiểm tra quy tắc shows.yaml.", "warn");
      }
      return;
    }
    try {
      const res = await apiFetch("/api/config/test-match", {
        method: "POST",
        body: {
          sample_path: sample,
          yaml_content: editor ? editor.value : null,
        },
      });
      setShowsYamlError(null);
      if (resModel) resModel.textContent = `${res.model_id} (${res.model_name})`;
      if (resDedup) resDedup.textContent = res.dedup_policy;
      if (resSource) resSource.textContent = res.resolution_source;
      if (resFps) resFps.textContent = `~${res.ref_median_fps} fps (RTX 3060 TensorRT FP16)`;
    } catch (err) {
      setShowsYamlError(err.message);
      showToast(err.message, "error");
    }
  };

  const loadModelsAndShows = async (forceOverwriteEditor = false) => {
    try {
      const data = await apiFetch("/api/config/models-and-shows");
      const models = Array.isArray(data.models) ? data.models : [];
      const routingModes = Array.isArray(data.routing_modes) ? data.routing_modes : [];
      const policies = Array.isArray(data.dedup_policies) ? data.dedup_policies : [];
      const multipassPolicies = Array.isArray(data.nvenc_multipass_policies)
        ? data.nvenc_multipass_policies
        : ["auto", "fullres", "qres"];

      // Populate Queue dropdowns
      if (optModelSelect instanceof HTMLSelectElement) {
        const cur = optModelSelect.value || optModelSelect.dataset.desiredValue || "";
        const addedIds = new Set();
        clearChildren(optModelSelect);
        optModelSelect.append(
          el("option", {
            value: "",
            textContent: "Tự động phân giải theo shows.yaml (Khuyên dùng)",
          })
        );
        for (const rm of routingModes) {
          if (!rm?.id || addedIds.has(rm.id)) continue;
          addedIds.add(rm.id);
          optModelSelect.append(
            el("option", {
              value: rm.id,
              textContent: `${rm.id} — ~${rm.ref_median_fps} fps (Auto-Pareto)`,
            })
          );
        }
        for (const m of models) {
          if (!m?.id || addedIds.has(m.id)) continue;
          addedIds.add(m.id);
          optModelSelect.append(
            el("option", {
              value: m.id,
              textContent: `${m.id} — ~${m.ref_median_fps} fps (${m.native_upscale}x)`,
            })
          );
        }
        setSelectValueSafely(optModelSelect, cur, "");
      }

      if (optNvencMultipassSelect instanceof HTMLSelectElement) {
        const curMp =
          optNvencMultipassSelect.dataset.desiredValue || optNvencMultipassSelect.value || "auto";
        const mpLabels = {
          auto: "auto — Tự động theo độ phân giải (Khuyên dùng)",
          fullres: "fullres — 2-Pass toàn khung (Chất lượng tối đa)",
          qres: "qres — 2-Pass 1/4 độ phân giải (Tốc độ cao)",
        };
        clearChildren(optNvencMultipassSelect);
        for (const mp of multipassPolicies) {
          optNvencMultipassSelect.append(
            el("option", {
              value: mp,
              textContent: mpLabels[mp] || mp,
            })
          );
        }
        setSelectValueSafely(optNvencMultipassSelect, curMp, "auto");
      }

      if (optDedupSelect instanceof HTMLSelectElement) {
        const curPol = optDedupSelect.value || optDedupSelect.dataset.desiredValue || "";
        clearChildren(optDedupSelect);
        optDedupSelect.append(
          el("option", {
            value: "",
            textContent: "Tự động theo shows.yaml / Mặc định",
          })
        );
        for (const p of policies) {
          const desc = p.description ? ` — ${p.description}` : "";
          optDedupSelect.append(
            el("option", {
              value: p.id,
              textContent: `${p.id}${desc}`,
            })
          );
        }
        setSelectValueSafely(optDedupSelect, curPol, "");
      }
      window.dispatchEvent(new CustomEvent("studio:models-loaded"));

      // Render Models Table
      if (tbody) {
        clearChildren(tbody);
        for (const m of models) {
          const idChildren = [el("code", { textContent: m.id })];
          if (m.default) {
            idChildren.push(" ", el("span", { className: "badge badge-ok", textContent: "Mặc định" }));
          }
          const archDetail =
            m.num_feat && m.num_conv ? `${m.arch_type} (${m.num_feat}f${m.num_conv}c)` : m.arch_type;

          const weightBadge = m.weights_ready
            ? el("span", {
                className: "badge badge-ok",
                textContent: `✓ Sẵn sàng (${m.weights_size_mb} MB)`,
              })
            : el("span", {
                className: "badge badge-warn",
                textContent: "⬇ Tự tải SHA-256 khi dùng",
              });

          const trtBadge = m.trt_cached
            ? el("span", {
                className: "badge badge-ok",
                textContent: `⚡ Đã cache (${m.trt_engine_files.length} engine)`,
              })
            : el("span", {
                className: "badge",
                textContent: "Biên dịch lần đầu",
              });

          const scaleText =
            m.downsample && m.downsample !== "none"
              ? `Native ${m.native_upscale}x → 2x (${m.downsample})`
              : `Native ${m.native_upscale}x`;

          tbody.append(
            el("tr", {}, [
              el("th", { scope: "row" }, idChildren),
              el("td", {}, [
                el("div", { textContent: m.name }),
                el("div", { className: "muted-meta", textContent: archDetail }),
              ]),
              el("td", { textContent: scaleText }),
              el("td", {}, [weightBadge]),
              el("td", {}, [trtBadge]),
              el("td", { textContent: `~${m.ref_median_fps} fps (P5: ${m.ref_p5_fps})` }),
              el("td", {}, [
                el("code", { textContent: m.sha256_prefix ? `${m.sha256_prefix}…` : "N/A" }),
                el("div", { className: "muted-meta", textContent: m.license || "—" }),
              ]),
            ])
          );
        }
      }

      if (editor && (forceOverwriteEditor || !editorDirty)) {
        editor.value = data.shows_yaml || "shows: []\n";
        editorDirty = false;
        setShowsYamlError(null);
      }

      await runTestMatch();
    } catch (err) {
      showToast(`Lỗi nạp danh mục model: ${err.message}`, "error");
    }
  };

  btnRefresh?.addEventListener("click", () => {
    loadModelsAndShows(false).then(() =>
      showToast("Đã kiểm tra lại trọng số và TensorRT Engine.", "info", 2800)
    );
  });

  btnTestMatch?.addEventListener("click", () => runTestMatch({ showUserWarning: true }));
  testInput?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      if (ev.isComposing || ev.keyCode === 229) return;
      ev.preventDefault();
      runTestMatch({ showUserWarning: true });
    }
  });

  btnReloadShows?.addEventListener("click", () => {
    loadModelsAndShows(true).then(() => showToast("Đã nạp lại shows.yaml từ đĩa.", "info", 2600));
  });

  btnSaveShows?.addEventListener("click", async () => {
    if (!editor) return;
    if (modelsSystemBusy) {
      showToast(
        modelsSystemBusyReason || "Đang có tiến trình chạy — vui lòng chờ hoàn tất trước khi lưu shows.yaml.",
        "warn"
      );
      return;
    }
    try {
      const res = await apiFetch("/api/config/shows", {
        method: "PUT",
        body: { yaml_content: editor.value },
      });
      editorDirty = false;
      setShowsYamlError(null);
      showToast(`Đã lưu shows.yaml hợp lệ (${res.rules_count} quy tắc).`, "success");
      await runTestMatch();
      onShowsUpdated?.();
    } catch (err) {
      setShowsYamlError(err.message);
      editor.focus();
      showToast(err.message, "error");
    }
  });

  syncShowsSaveButton();
  loadModelsAndShows(true);

  return {
    refresh: () => loadModelsAndShows(false),
    setSystemBusy(isBusy, reason = "") {
      const nextBusy = Boolean(isBusy);
      const nextReason = String(reason || "");
      if (modelsSystemBusy === nextBusy && modelsSystemBusyReason === nextReason) return;
      modelsSystemBusy = nextBusy;
      modelsSystemBusyReason = nextReason;
      syncShowsSaveButton();
    },
  };
}

/* =====================================================================
 * TAB 5: SUBTRANS STUDIO & AUDIT VAULT
 * ===================================================================== */
export function initSubtransAndAuditsPanel({ selectTabById, compareCtrl, onJobLaunched }) {
  const subForm = document.getElementById("subtrans-form");
  const subInputPath = document.getElementById("sub-input-path");
  const subInputError = document.getElementById("sub-input-error");
  const btnBrowseSub = document.getElementById("btn-browse-sub-input");
  const subQuality = document.getElementById("sub-quality-select");
  const subGlossary = document.getElementById("sub-glossary-select");
  const subOffset = document.getElementById("sub-offset-input");
  const subOffsetError = document.getElementById("sub-offset-error");
  const subOutDir = document.getElementById("sub-output-dir");
  const btnBrowseSubOutDir = document.getElementById("btn-browse-sub-outdir");
  const subAutoAlign = document.getElementById("sub-auto-align");
  const subPreferWhisper = document.getElementById("sub-prefer-whisper");
  const subUpgrade = document.getElementById("sub-upgrade");
  const subNoEmbed = document.getElementById("sub-no-embed");
  const subDryRun = document.getElementById("sub-dry-run");
  const btnStartSubtrans = document.getElementById("btn-start-subtrans");
  const btnRunPending = document.getElementById("btn-run-subs-pending");
  const pendingCountSpan = document.getElementById("pending-subs-count");

  const btnRefreshSubtrans = document.getElementById("btn-refresh-subtrans");
  const kpiKeys = document.getElementById("sub-kpi-keys");
  const kpiCache = document.getElementById("sub-kpi-cache");
  const kpiReset = document.getElementById("sub-kpi-reset");
  const todayUsageBox = document.getElementById("sub-today-usage");
  const keySlotsStatus = document.getElementById("key-slots-status");
  const apiKeysForm = document.getElementById("api-keys-form");
  const keyInput1 = document.getElementById("input-api-key-1");
  const keyInput2 = document.getElementById("input-api-key-2");
  const keyInput3 = document.getElementById("input-api-key-3");
  const keyClear1 = document.getElementById("clear-api-key-1");
  const keyClear2 = document.getElementById("clear-api-key-2");
  const keyClear3 = document.getElementById("clear-api-key-3");

  const btnRefreshAudits = document.getElementById("btn-refresh-audits");
  const auditsTbody = document.getElementById("audits-tbody");
  const auditDialog = document.getElementById("audit-detail-dialog");
  const auditTitle = document.getElementById("audit-detail-title");
  const auditSwitcher = document.getElementById("audit-doc-switcher");
  const auditCode = document.getElementById("audit-detail-code");

  const nightReportDetails = document.getElementById("night-report-details");
  const nightReportSession = document.getElementById("night-report-session");
  const btnRefreshNightReport = document.getElementById("btn-refresh-night-report");
  const btnCopyNightReport = document.getElementById("btn-copy-night-report");
  const nightReportRendered = document.getElementById("night-report-rendered");
  const nightReportPre = document.getElementById("night-report-pre");
  const nightReportCode = document.getElementById("night-report-code");

  let auditLoadSeq = 0;
  let pendingSubsCount = 0;
  let systemBusy = false;
  let systemBusyReason = "";
  let localSubLaunchInFlight = false;
  let nightReportLoaded = false;
  let nightReportSeq = 0;

  const syncSubtransLaunchButtons = () => {
    const busyOrLaunching = systemBusy || localSubLaunchInFlight;
    if (btnStartSubtrans instanceof HTMLButtonElement) {
      btnStartSubtrans.disabled = busyOrLaunching;
      btnStartSubtrans.title = systemBusy ? systemBusyReason || "Hệ thống đang có tiến trình chạy." : "";
    }
    if (btnRunPending instanceof HTMLButtonElement) {
      btnRunPending.disabled = busyOrLaunching || pendingSubsCount === 0;
      btnRunPending.title = systemBusy
        ? systemBusyReason || "Hệ thống đang có tiến trình chạy."
        : pendingSubsCount === 0
          ? "Không có tập nào trong hàng đợi tồn đọng."
          : "";
    }
  };

  if (auditDialog instanceof HTMLDialogElement) {
    initDialogLightDismiss(auditDialog);
    auditDialog.addEventListener("close", () => {
      auditDialog.returnValue = "";
      if (!auditDialog.open) {
        auditLoadSeq++;
      }
    });
  }

  const clearKeySlot = async (slotNum) => {
    try {
      const res = await apiFetch("/api/subtrans/api-keys", {
        method: "PUT",
        body: { [`key_${slotNum}`]: "" },
      });
      renderKeySlots(res.key_slots);
      await loadSubtransOverview();
      showToast(`Đã xóa khóa ở Khe ${slotNum} khỏi .env cục bộ.`, "info");
    } catch (err) {
      showToast(err.message, "error");
    }
  };

  const renderKeySlots = (slots) => {
    if (!keySlotsStatus) return;
    clearChildren(keySlotsStatus);
    for (const s of slots || []) {
      const chip = el("div", { className: "key-slot-chip" });
      const cls = s.configured ? "badge badge-ok" : "badge";
      const srcNote =
        s.configured && s.source === "env"
          ? " [ENV]"
          : s.configured && s.env_name === "GEMINI_API_KEYS"
            ? " [LIST]"
            : "";
      const label = s.configured
        ? `✓ Khe ${s.slot}: ${s.masked} (ID: ${s.key_id})${srcNote}`
        : `◌ Khe ${s.slot}: Chưa đặt`;
      chip.append(el("span", { className: cls, textContent: label }));
      if (s.configured && s.source !== "env") {
        chip.append(
          el(
            "button",
            {
              type: "button",
              className: "btn-ghost btn-xs",
              "aria-label": `Xóa khóa Gemini API Khe ${s.slot}`,
              onClick: () => clearKeySlot(s.slot),
            },
            [el("span", { "aria-hidden": "true", textContent: "✕ " }), "Xóa"]
          )
        );
      } else if (s.configured && s.source === "env") {
        chip.append(
          el("span", {
            className: "muted-meta",
            textContent: "Biến môi trường hệ thống",
          })
        );
      }
      keySlotsStatus.append(chip);
    }
  };

  const loadSubtransOverview = async () => {
    try {
      const data = await apiFetch("/api/subtrans/overview");

      if (kpiKeys) {
        const slotCap = Math.max(3, Number(data.configured_keys_count) || 0);
        kpiKeys.textContent = `${data.configured_keys_count} / ${slotCap} khe`;
      }
      if (kpiCache) {
        const tot = Number(data.cache?.total_cached_cues || 0).toLocaleString("vi-VN");
        const today = Number(data.cache?.today_cached_cues || 0).toLocaleString("vi-VN");
        kpiCache.textContent = `${tot} câu (+${today})`;
      }
      if (kpiReset) {
        const hhmm = (data.next_reset_vn || "").slice(11, 16) || "14:00";
        kpiReset.textContent = `${formatDuration(data.reset_in_seconds)} (${hhmm})`;
      }
      if (pendingCountSpan) {
        const count = Array.isArray(data.pending_subs) ? data.pending_subs.length : 0;
        pendingCountSpan.textContent = String(count);
        pendingSubsCount = count;
        syncSubtransLaunchButtons();
      }

      if (subGlossary instanceof HTMLSelectElement) {
        const curG = subGlossary.value;
        clearChildren(subGlossary);
        subGlossary.append(
          el("option", { value: "", textContent: "Tự động nhận diện theo tên phim" })
        );
        for (const g of data.glossaries || []) {
          subGlossary.append(el("option", { value: g.path, textContent: g.name }));
        }
        setSelectValueSafely(subGlossary, curG, "");
      }

      renderKeySlots(data.key_slots);

      if (todayUsageBox) {
        clearChildren(todayUsageBox);
        const reqs = data.cache?.today_requests || [];
        const exModels = data.cache?.exhausted_models || [];
        if (reqs.length === 0 && exModels.length === 0) {
          todayUsageBox.append(
            el("p", {
              className: "muted-meta",
              textContent:
                "Chưa có lượt gọi Gemini API nào ghi nhận trong ngày PT hôm nay — Toàn bộ hạn mức sẵn sàng.",
            })
          );
        } else {
          for (const r of reqs) {
            const tone = r.status === "success" || r.status === "ok" ? "badge-ok" : "badge-warn";
            todayUsageBox.append(
              el("span", {
                className: `badge ${tone}`,
                textContent: `${r.model} [${r.status}]: ${r.requests} req • ${r.cues} câu`,
              })
            );
          }
          for (const em of exModels) {
            todayUsageBox.append(
              el("span", {
                className: "badge badge-err",
                textContent: `Chạm trần RPD: ${em.model}`,
              })
            );
          }
        }
      }
    } catch (err) {
      showToast(`Lỗi tải trạng thái Subtrans: ${err.message}`, "error");
    }
  };

  const openAuditDocument = async (episodeTitle, docsList, initialPath) => {
    if (!(auditDialog instanceof HTMLDialogElement) || !auditCode) return;
    if (auditTitle) {
      auditTitle.textContent = `Biên bản Kiểm định — ${episodeTitle}`;
    }

    const switcherBtns = new Map();
    const setActiveSwitcher = (activePath) => {
      for (const [docPath, btn] of switcherBtns.entries()) {
        const isActive = docPath === activePath;
        btn.setAttribute("aria-pressed", String(isActive));
        btn.className = isActive ? "btn btn-primary" : "btn btn-secondary";
      }
    };

    const loadDoc = async (docPath) => {
      if (!docPath) {
        auditCode.textContent = "Không có tệp biên bản hợp lệ để hiển thị.";
        return;
      }
      const reqId = ++auditLoadSeq;
      setActiveSwitcher(docPath);
      auditCode.textContent = "Đang đọc biên bản…";
      try {
        const qs = new URLSearchParams({ path: docPath });
        const res = await apiFetch(`/api/audits/detail?${qs.toString()}`);
        if (reqId !== auditLoadSeq || !auditDialog.open) return;
        auditCode.textContent =
          res.type === "json" ? JSON.stringify(res.data, null, 2) : String(res.content || "");
      } catch (err) {
        if (reqId !== auditLoadSeq || !auditDialog.open) return;
        const rawPreview = err?.payload?.raw_preview;
        auditCode.textContent = rawPreview
          ? `Không thể đọc tệp kiểm định: ${err.message}\n\n--- Nội dung thô (.audit.json bị hỏng) ---\n${rawPreview}`
          : `Không thể đọc tệp kiểm định: ${err.message}`;
      }
    };

    if (auditSwitcher) {
      clearChildren(auditSwitcher);
      for (const d of docsList) {
        if (!d.path) {
          auditSwitcher.append(
            el("button", {
              type: "button",
              className: "btn btn-ghost",
              disabled: true,
              "aria-disabled": "true",
              title: "Chưa có tệp kiểm định cho mục này (chưa dịch phụ đề hoặc chưa xuất cast).",
              textContent: d.label,
            })
          );
          continue;
        }
        const isInitial = d.path === initialPath;
        const btn = el("button", {
          type: "button",
          className: isInitial ? "btn btn-primary" : "btn btn-secondary",
          "aria-pressed": String(isInitial),
          textContent: d.label,
          onClick: () => loadDoc(d.path),
        });
        switcherBtns.set(d.path, btn);
        auditSwitcher.append(btn);
      }
    }

    auditDialog.returnValue = "";
    if (!auditDialog.open) {
      auditDialog.showModal();
    }
    await loadDoc(initialPath);
  };

  const loadAudits = async () => {
    if (!auditsTbody) return;
    try {
      const data = await apiFetch("/api/audits");
      const audits = Array.isArray(data.audits) ? data.audits : [];
      clearChildren(auditsTbody);

      if (audits.length === 0) {
        auditsTbody.append(
          el("tr", {}, [
            el("th", { scope: "row", textContent: "Chưa có biên bản" }),
            el("td", {
              colspan: "6",
              textContent:
                "Các biên bản *.audit.json sẽ tự động xuất hiện tại đây sau khi hoàn tất nâng cấp tập phim.",
            }),
          ])
        );
        return;
      }

      for (const a of audits) {
        const muxTone =
          a.status === "PASS"
            ? "badge-ok"
            : a.status === "FAIL" || a.status === "CORRUPT" || a.corrupt
              ? "badge-err"
              : "badge-warn";
        const sanTone =
          a.sanity_status === "PASS"
            ? "badge-ok"
            : a.sanity_status === "FAIL"
              ? "badge-err"
              : "";

        const docs = [
          { label: "📄 Mux Audit (.json)", path: a.audit_json_path },
          { label: "📝 Subtrans Audit (.md)", path: a.sub_audit_md_path },
          { label: "🎭 Hồ sơ Cast (.yaml)", path: a.cast_yaml_path },
        ];

        const actionBtns = el("div", { className: "button-cluster" }, [
          el(
            "button",
            {
              type: "button",
              className: "btn btn-secondary",
              "aria-label": `Biên bản kiểm định ${a.episode}`,
              onClick: () => openAuditDocument(a.episode, docs, a.audit_json_path),
            },
            [el("span", { "aria-hidden": "true", textContent: "📄 " }), "Biên bản"]
          ),
        ]);

        if (a.source_file && a.output_file) {
          actionBtns.append(
            el(
              "button",
              {
                type: "button",
                className: "btn btn-ghost",
                "aria-label": `Soi 4K ${a.episode}`,
                onClick: () => compareCtrl?.openPair(a.source_file, a.output_file),
              },
              [el("span", { "aria-hidden": "true", textContent: "🔍 " }), "Soi 4K"]
            )
          );
        }

        if (a.output_file) {
          actionBtns.append(
            el(
              "button",
              {
                type: "button",
                className: "btn btn-ghost",
                "aria-label": `Phát ${a.episode}`,
                onClick: async () => {
                  try {
                    const res = await apiFetch("/api/open-media", {
                      method: "POST",
                      body: { path: a.output_file, prefer_mpv: true },
                    });
                    showToast(`Đã mở trình phát (${res.launcher}).`, "success", 2800);
                  } catch (err) {
                    showToast(err.message, "error");
                  }
                },
              },
              [el("span", { "aria-hidden": "true", textContent: "🎬 " }), "Phát"]
            )
          );
        }

        const framesText =
          a.source_frames && a.output_frames
            ? `${a.frames_matched ? "✓ " : "⚠ "}${a.output_frames.toLocaleString("vi-VN")} / ${a.source_frames.toLocaleString("vi-VN")}`
            : "—";

        auditsTbody.append(
          el("tr", {}, [
            el("th", { scope: "row" }, [
              el("div", { textContent: a.episode }),
              el("div", {
                className: a.corrupt ? "muted-meta text-error" : "muted-meta",
                textContent: a.corrupt && a.error
                  ? a.error
                  : `${a.model_id || "—"} ${a.completed_at ? `• ${a.completed_at}` : ""}`,
              }),
            ]),
            el("td", {}, [
              el("span", {
                className: `badge ${muxTone}`,
                textContent: a.status,
                ...(a.corrupt && a.error ? { title: a.error } : {}),
              }),
            ]),
            el("td", {}, [
              el("span", { className: `badge ${sanTone}`, textContent: a.sanity_status }),
            ]),
            el("td", { textContent: framesText }),
            el("td", {
              textContent: `${a.audio_tracks_count} Audio / ${a.subtitle_tracks_count} Sub / ${a.fonts_count} Font`,
            }),
            el("td", {
              textContent: `${Number(a.average_pipeline_fps || 0).toFixed(2)} fps • ${Number(a.output_file_size_mb || 0).toFixed(1)} MB`,
            }),
            el("td", {}, [actionBtns]),
          ])
        );
      }
    } catch (err) {
      showToast(`Lỗi nạp danh sách kiểm định: ${err.message}`, "error");
    }
  };

  const setSubFieldError = (inputEl, errorEl, msg) => {
    if (msg) {
      inputEl?.setAttribute("aria-invalid", "true");
      if (errorEl) {
        errorEl.textContent = msg;
        errorEl.hidden = false;
      }
    } else {
      inputEl?.removeAttribute("aria-invalid");
      if (errorEl) {
        errorEl.textContent = "";
        errorEl.hidden = true;
      }
    }
  };

  subInputPath?.addEventListener("input", () => setSubFieldError(subInputPath, subInputError, null));
  subOffset?.addEventListener("input", () => setSubFieldError(subOffset, subOffsetError, null));

  btnBrowseSub?.addEventListener("click", async () => {
    const picked = await openFsPicker({
      title: "Chọn Video (.mkv/.mp4) hoặc Phụ đề rời (.srt/.ass)",
      multi: false,
      allowFolders: false,
      includeSubtitles: true,
      allow4k: true,
    });
    if (picked.length > 0 && subInputPath) {
      subInputPath.value = picked[0];
      setSubFieldError(subInputPath, subInputError, null);
    }
  });

  btnBrowseSubOutDir?.addEventListener("click", async () => {
    const picked = await openFsPicker({
      title: "Chọn Thư mục Xuất Phụ đề (-o)",
      multi: false,
      allowFolders: true,
      foldersOnly: true,
      initialPath: subOutDir?.value.trim() || "out",
    });
    if (picked.length > 0 && subOutDir) {
      subOutDir.value = picked[0];
    }
  });

  subForm?.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (systemBusy) {
      showToast(systemBusyReason || "Hệ thống đang có tiến trình chạy.", "warn");
      return;
    }
    const target = subInputPath?.value.trim() || "";
    if (!target) {
      setSubFieldError(subInputPath, subInputError, "Vui lòng chọn tệp video hoặc phụ đề cần dịch.");
      subInputPath?.focus();
      showToast("Vui lòng chọn tệp video hoặc phụ đề cần dịch.", "warn");
      return;
    }
    setSubFieldError(subInputPath, subInputError, null);
    const rawOffset = subOffset?.value.trim() ?? "";
    const subOffsetBad = subOffset instanceof HTMLInputElement && subOffset.validity.badInput;
    const offsetMs = !subOffsetBad && rawOffset === "" ? 0 : Number(rawOffset);
    if (subOffsetBad || !Number.isFinite(offsetMs) || !Number.isInteger(offsetMs)) {
      setSubFieldError(subOffset, subOffsetError, "Sub Offset (ms) phải là số nguyên hợp lệ.");
      subOffset?.focus();
      showToast("Sub Offset (ms) phải là số nguyên hợp lệ.", "warn");
      return;
    }
    setSubFieldError(subOffset, subOffsetError, null);
    localSubLaunchInFlight = true;
    syncSubtransLaunchButtons();
    try {
      const res = await apiFetch("/api/jobs/subtrans", {
        method: "POST",
        body: {
          input_path: target,
          output_dir: subOutDir?.value.trim() || null,
          subs_quality: subQuality?.value || "normal",
          subs_upgrade: Boolean(subUpgrade?.checked),
          prefer_whisper: Boolean(subPreferWhisper?.checked),
          auto_align_en: Boolean(subAutoAlign?.checked ?? true),
          no_embed: Boolean(subNoEmbed?.checked),
          sub_offset_ms: offsetMs,
          glossary: subGlossary?.value || null,
          dry_run: Boolean(subDryRun?.checked),
        },
      });
      showToast(`Đã khởi chạy dịch phụ đề (PID=${res.job.pid}).`, "success");
      selectTabById?.("tab-monitor");
      onJobLaunched?.(res.job);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      localSubLaunchInFlight = false;
      syncSubtransLaunchButtons();
    }
  });

  btnRunPending?.addEventListener("click", async () => {
    if (systemBusy) {
      showToast(systemBusyReason || "Hệ thống đang có tiến trình chạy.", "warn");
      return;
    }
    localSubLaunchInFlight = true;
    syncSubtransLaunchButtons();
    try {
      const res = await apiFetch("/api/jobs/subs-pending", {
        method: "POST",
        body: {
          subs_quality: subQuality?.value || "normal",
          dry_run: Boolean(subDryRun?.checked),
        },
      });
      showToast(`Đã khởi chạy dịch hàng đợi tồn đọng (PID=${res.job.pid}).`, "success");
      selectTabById?.("tab-monitor");
      onJobLaunched?.(res.job);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      localSubLaunchInFlight = false;
      syncSubtransLaunchButtons();
    }
  });

  for (const [inp, chk] of [
    [keyInput1, keyClear1],
    [keyInput2, keyClear2],
    [keyInput3, keyClear3],
  ]) {
    inp?.addEventListener("input", () => {
      if (chk && inp.value.trim()) chk.checked = false;
    });
    chk?.addEventListener("change", () => {
      if (chk.checked && inp) inp.value = "";
    });
  }

  apiKeysForm?.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const v1 = keyInput1?.value.trim() || "";
    const v2 = keyInput2?.value.trim() || "";
    const v3 = keyInput3?.value.trim() || "";
    const c1 = Boolean(keyClear1?.checked);
    const c2 = Boolean(keyClear2?.checked);
    const c3 = Boolean(keyClear3?.checked);
    if (!v1 && !v2 && !v3 && !c1 && !c2 && !c3) {
      showToast("Hãy nhập khóa API mới hoặc tích chọn xóa khe cần cập nhật.", "warn");
      return;
    }
    const submitBtn = ev.submitter instanceof HTMLButtonElement ? ev.submitter : null;
    if (submitBtn) submitBtn.disabled = true;
    try {
      const res = await apiFetch("/api/subtrans/api-keys", {
        method: "PUT",
        body: {
          key_1: c1 && !v1 ? "" : v1 ? v1 : null,
          key_2: c2 && !v2 ? "" : v2 ? v2 : null,
          key_3: c3 && !v3 ? "" : v3 ? v3 : null,
        },
      });
      if (keyInput1) keyInput1.value = "";
      if (keyInput2) keyInput2.value = "";
      if (keyInput3) keyInput3.value = "";
      if (keyClear1) keyClear1.checked = false;
      if (keyClear2) keyClear2.checked = false;
      if (keyClear3) keyClear3.checked = false;
      renderKeySlots(res.key_slots);
      await loadSubtransOverview();
      showToast("Đã cập nhật khóa Gemini API an toàn vào .env cục bộ.", "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });

  const REFRESH_TTL_MS = 5000;
  let lastRefreshMs = 0;

  btnRefreshSubtrans?.addEventListener("click", () => {
    lastRefreshMs = Date.now();
    loadSubtransOverview().then(() => showToast("Đã cập nhật thống kê Subtrans.", "info", 2600));
  });
  btnRefreshAudits?.addEventListener("click", () => {
    lastRefreshMs = Date.now();
    loadAudits().then(() => showToast("Đã làm mới Kho Kiểm định.", "info", 2600));
  });

  const appendInlineMarkdownNodes = (parent, lineText) => {
    const tokenRe = /(\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*)/g;
    let lastIdx = 0;
    let match;
    while ((match = tokenRe.exec(lineText)) !== null) {
      if (match.index > lastIdx) {
        parent.append(document.createTextNode(lineText.slice(lastIdx, match.index)));
      }
      if (match[2] !== undefined) {
        parent.append(el("strong", { textContent: match[2] }));
      } else if (match[3] !== undefined) {
        parent.append(el("code", { textContent: match[3] }));
      } else if (match[4] !== undefined) {
        parent.append(el("em", { textContent: match[4] }));
      }
      lastIdx = tokenRe.lastIndex;
    }
    if (lastIdx < lineText.length) {
      parent.append(document.createTextNode(lineText.slice(lastIdx)));
    }
  };

  const renderSafeMarkdownToDom = (container, rawMarkdown) => {
    if (!container) return;
    clearChildren(container);
    const lines = String(rawMarkdown || "").split(/\r?\n/);
    let curList = null;
    let curListTag = "";
    const flushList = () => {
      if (curList) {
        container.append(curList);
        curList = null;
        curListTag = "";
      }
    };
    for (const rawLine of lines) {
      const trimmed = rawLine.trim();
      if (!trimmed) {
        flushList();
        continue;
      }
      const hMatch = /^(#{1,3})\s+(.*)$/.exec(trimmed);
      if (hMatch) {
        flushList();
        const tag = hMatch[1].length === 1 ? "h3" : hMatch[1].length === 2 ? "h4" : "h5";
        const hEl = el(tag);
        appendInlineMarkdownNodes(hEl, hMatch[2]);
        container.append(hEl);
        continue;
      }
      const ulMatch = /^[-*]\s+(.*)$/.exec(trimmed);
      if (ulMatch) {
        if (curListTag !== "ul") {
          flushList();
          curList = el("ul");
          curListTag = "ul";
        }
        const li = el("li");
        appendInlineMarkdownNodes(li, ulMatch[1]);
        curList.append(li);
        continue;
      }
      const olMatch = /^\d+\.\s+(.*)$/.exec(trimmed);
      if (olMatch) {
        if (curListTag !== "ol") {
          flushList();
          curList = el("ol");
          curListTag = "ol";
        }
        const li = el("li");
        appendInlineMarkdownNodes(li, olMatch[1]);
        curList.append(li);
        continue;
      }
      flushList();
      const pEl = el("p");
      appendInlineMarkdownNodes(pEl, trimmed);
      container.append(pEl);
    }
    flushList();
  };

  const syncNightReportSessions = (sessions, activeName) => {
    if (!(nightReportSession instanceof HTMLSelectElement) || !Array.isArray(sessions) || sessions.length === 0) {
      return;
    }
    const desired = activeName || nightReportSession.value || "night1";
    clearChildren(nightReportSession);
    for (const s of sessions) {
      const cleanS = String(s || "").trim();
      if (!cleanS) continue;
      nightReportSession.append(
        el("option", {
          value: cleanS,
          textContent: `Phiên: ${cleanS}`,
        })
      );
    }
    setSelectValueSafely(nightReportSession, desired, "night1");
  };

  const loadNightReport = async ({ showToastOnDone = false } = {}) => {
    if (!nightReportCode) return;
    const reqId = ++nightReportSeq;
    const chosenSession =
      nightReportSession instanceof HTMLSelectElement && nightReportSession.value
        ? nightReportSession.value
        : "night1";
    if (btnRefreshNightReport instanceof HTMLButtonElement) {
      btnRefreshNightReport.disabled = true;
      btnRefreshNightReport.setAttribute("aria-busy", "true");
    }
    nightReportCode.textContent = "Đang tổng hợp báo cáo ca đêm từ logs/ & out/…";
    if (nightReportRendered) nightReportRendered.hidden = true;
    if (nightReportPre) nightReportPre.hidden = false;
    try {
      const qs = new URLSearchParams({ session_name: chosenSession });
      const res = await apiFetch(`/api/night-report?${qs.toString()}`);
      if (reqId !== nightReportSeq) return;
      const mdText = String(res.markdown || "Chưa có dữ liệu báo cáo ca đêm.");
      nightReportCode.textContent = mdText;
      if (Array.isArray(res.available_sessions)) {
        syncNightReportSessions(res.available_sessions, res.session_name || chosenSession);
      }
      if (nightReportRendered) {
        renderSafeMarkdownToDom(nightReportRendered, mdText);
        nightReportRendered.hidden = false;
        if (nightReportPre) nightReportPre.hidden = true;
      }
      nightReportLoaded = true;
      if (showToastOnDone) {
        showToast("Đã làm mới Báo cáo Ca đêm.", "info", 2400);
      }
    } catch (err) {
      if (reqId !== nightReportSeq) return;
      nightReportCode.textContent = `Không thể tạo báo cáo ca đêm: ${err.message}`;
      if (nightReportRendered) nightReportRendered.hidden = true;
      if (nightReportPre) nightReportPre.hidden = false;
      showToast(err.message, "error");
    } finally {
      if (reqId === nightReportSeq && btnRefreshNightReport instanceof HTMLButtonElement) {
        btnRefreshNightReport.disabled = false;
        btnRefreshNightReport.removeAttribute("aria-busy");
      }
    }
  };

  nightReportSession?.addEventListener("change", () => {
    if (nightReportDetails instanceof HTMLDetailsElement && !nightReportDetails.open) {
      nightReportDetails.open = true;
    }
    loadNightReport({ showToastOnDone: true });
  });

  nightReportDetails?.addEventListener("toggle", () => {
    if (nightReportDetails instanceof HTMLDetailsElement && nightReportDetails.open && !nightReportLoaded) {
      loadNightReport();
    }
  });

  btnRefreshNightReport?.addEventListener("click", () => {
    if (nightReportDetails instanceof HTMLDetailsElement && !nightReportDetails.open) {
      nightReportDetails.open = true;
    }
    loadNightReport({ showToastOnDone: true });
  });

  btnCopyNightReport?.addEventListener("click", async () => {
    const text = nightReportCode?.textContent || "";
    if (!text.trim()) {
      showToast("Chưa có nội dung báo cáo để sao chép.", "warn");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      showToast("Đã sao chép Báo cáo Ca đêm vào bộ nhớ tạm.", "success", 2500);
    } catch {
      showToast("Trình duyệt không cho phép sao chép tự động.", "warn");
    }
  });

  lastRefreshMs = Date.now();
  syncSubtransLaunchButtons();
  loadSubtransOverview();
  loadAudits();

  return {
    refreshAll({ force = false } = {}) {
      const nowMs = Date.now();
      if (!force && nowMs - lastRefreshMs < REFRESH_TTL_MS) {
        return;
      }
      lastRefreshMs = nowMs;
      loadSubtransOverview();
      loadAudits();
    },
    refreshAudits: loadAudits,
    setSystemBusy(isBusy, reason = "") {
      const nextBusy = Boolean(isBusy);
      const nextReason = String(reason || "");
      if (systemBusy === nextBusy && systemBusyReason === nextReason) return;
      systemBusy = nextBusy;
      systemBusyReason = nextReason;
      syncSubtransLaunchButtons();
    },
  };
}
