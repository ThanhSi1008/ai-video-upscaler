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

/* =====================================================================
 * TAB 2: QUEUE & OVERNIGHT SCHEDULER
 * ===================================================================== */
export function initQueuePanel({ selectTabById, onJobLaunched }) {
  const listEl = document.getElementById("queue-inputs-list");
  const manualInput = document.getElementById("manual-queue-path");
  const btnAddManual = document.getElementById("btn-add-manual-path");
  const btnOpenFs = document.getElementById("btn-open-fs-queue");
  const btnClear = document.getElementById("btn-clear-queue");
  const form = document.getElementById("upscale-config-form");
  const btnCalc = document.getElementById("btn-calc-schedule");
  const btnPreview10s = document.getElementById("btn-run-preview-10s");
  const summaryPill = document.getElementById("schedule-summary-pill");
  const scheduleTable = document.getElementById("schedule-table");
  const panelQueue = document.getElementById("panel-queue");
  const tbody = document.getElementById("schedule-tbody");

  const optModel = document.getElementById("opt-model");
  const optDedup = document.getElementById("opt-dedup");
  const optBatch = document.getElementById("opt-batch-size");
  const optBatchError = document.getElementById("opt-batch-size-error");
  const optDeadline = document.getElementById("opt-deadline");
  const optDeadlineError = document.getElementById("opt-deadline-error");
  const optOutDir = document.getElementById("opt-output-dir");
  const optSubsQuality = document.getElementById("opt-subs-quality");
  const optZeroDisk = document.getElementById("opt-zero-disk");
  const optTensorrt = document.getElementById("opt-tensorrt");
  const optAllowPartial = document.getElementById("opt-allow-partial");
  const optSubsAfter = document.getElementById("opt-subs-after");
  const optAsyncSubs = document.getElementById("opt-async-subs");
  const optPreferWhisper = document.getElementById("opt-prefer-whisper");
  const optForce = document.getElementById("opt-force");
  const optDryRun = document.getElementById("opt-dry-run");

  let queueInputs = [];
  let calcTimer = null;
  let hasComputedInitial = false;
  let scheduleDirty = false;
  let scheduleSeq = 0;

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
    queueInputs.splice(index, 1);
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
      const li = el("li", { className: "selected-input-item" }, [
        el("code", { textContent: p }),
        removeBtn,
      ]);
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
      if (summaryPill) {
        summaryPill.textContent = "Chọn tập phim để xem dự báo tự động theo shows.yaml";
      }
      renderEmptySchedule("Chưa có dữ liệu dự báo. Hãy thêm ít nhất 1 file hoặc thư mục ở mục 1.");
      return;
    }

    if (!validateDeadlineInput()) {
      if (summaryPill) {
        summaryPill.textContent = "Định dạng hạn chót không hợp lệ (cần HH:MM, ví dụ 07:00).";
      }
      return;
    }

    const reqId = ++scheduleSeq;
    if (summaryPill) {
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
        zero_disk: Boolean(optZeroDisk?.checked ?? true),
        use_tensorrt: Boolean(optTensorrt?.checked ?? true),
        subs_after_upscale: Boolean(optSubsAfter?.checked ?? true),
        output_dir: optOutDir?.value.trim() || null,
        force: Boolean(optForce?.checked),
      };

      const data = await apiFetch("/api/queue/preview-schedule", {
        method: "POST",
        body: payload,
      });
      if (reqId !== scheduleSeq) return;

      if (summaryPill) {
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
            textContent: `${ep.n_unique.toLocaleString("vi-VN")} / ${ep.total_frames.toLocaleString("vi-VN")} (-${ep.dedup_ratio_percent}%)`,
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
        summaryPill.textContent = `Lỗi dự báo lịch trình: ${err.message}`;
      }
      renderEmptySchedule(`Không thể tính dự báo: ${err.message}`);
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

  const handleManualAdd = () => {
    if (!manualInput) return;
    const val = manualInput.value.trim();
    if (!val) return;
    addInputs([val]);
    manualInput.value = "";
  };

  btnAddManual?.addEventListener("click", handleManualAdd);
  manualInput?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      if (ev.isComposing || ev.keyCode === 229) return;
      ev.preventDefault();
      handleManualAdd();
    }
  });

  btnClear?.addEventListener("click", () => {
    queueInputs = [];
    persistInputs();
    renderInputsList();
    computeScheduleNow();
  });

  btnCalc?.addEventListener("click", computeScheduleNow);

  optDeadline?.addEventListener("input", () => {
    const val = optDeadline.value.trim();
    if (!val || optDeadline.checkValidity()) {
      setFieldError(optDeadline, optDeadlineError, null);
    }
  });
  optDeadline?.addEventListener("blur", () => {
    validateDeadlineInput();
  });
  optBatch?.addEventListener("input", () => setFieldError(optBatch, optBatchError, null));

  for (const ctrl of [
    optModel,
    optDedup,
    optDeadline,
    optOutDir,
    optZeroDisk,
    optTensorrt,
    optAllowPartial,
    optSubsAfter,
    optForce,
  ]) {
    ctrl?.addEventListener("change", schedulePreviewDebounced);
  }

  btnPreview10s?.addEventListener("click", async () => {
    if (queueInputs.length === 0) {
      showToast("Hãy chọn ít nhất 1 file video hoặc thư mục để chạy Xem trước 10s.", "warn");
      return;
    }
    if (btnPreview10s instanceof HTMLButtonElement) btnPreview10s.disabled = true;
    try {
      const res = await apiFetch("/api/jobs/preview", {
        method: "POST",
        body: {
          video_path: queueInputs[0],
          seconds: 10,
          model: optModel?.value || null,
          dedup_policy: optDedup?.value || null,
          zero_disk: Boolean(optZeroDisk?.checked ?? true),
          use_tensorrt: Boolean(optTensorrt?.checked ?? true),
          force: true,
          auto_open_player: false,
        },
      });
      showToast(`Đã khởi chạy Xem trước 10s (PID=${res.job.pid}).`, "success");
      selectTabById?.("tab-monitor");
      onJobLaunched?.(res.job);
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      if (btnPreview10s instanceof HTMLButtonElement) btnPreview10s.disabled = false;
    }
  });

  form?.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (queueInputs.length === 0) {
      showToast("Vui lòng chọn ít nhất 1 video hoặc thư mục nguồn trước khi bắt đầu.", "warn");
      return;
    }
    const rawBatch = optBatch?.value.trim() ?? "";
    const batchVal = rawBatch === "" ? 8 : Number(rawBatch);
    if (!Number.isFinite(batchVal) || !Number.isInteger(batchVal) || batchVal < 1 || batchVal > 32) {
      setFieldError(optBatch, optBatchError, "Batch Size phải là số nguyên từ 1 đến 32.");
      optBatch?.focus();
      showToast("Batch Size phải là số nguyên từ 1 đến 32.", "warn");
      return;
    }
    setFieldError(optBatch, optBatchError, null);
    if (!validateDeadlineInput()) {
      optDeadline?.focus();
      showToast("Định dạng giờ hạn chót phải là HH:MM (ví dụ 07:00).", "warn");
      return;
    }
    const submitBtn = ev.submitter instanceof HTMLButtonElement ? ev.submitter : null;
    if (submitBtn) submitBtn.disabled = true;
    try {
      const payload = {
        inputs: queueInputs,
        output_dir: optOutDir?.value.trim() || null,
        model: optModel?.value || null,
        dedup_policy: optDedup?.value || null,
        batch_size: batchVal,
        deadline: optDeadline?.value.trim() || null,
        allow_partial: Boolean(optAllowPartial?.checked),
        zero_disk: Boolean(optZeroDisk?.checked ?? true),
        use_tensorrt: Boolean(optTensorrt?.checked ?? true),
        subs_after_upscale: Boolean(optSubsAfter?.checked ?? true),
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
      if (submitBtn) submitBtn.disabled = false;
    }
  });

  renderInputsList();
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
    focusWithPath(pathStr) {
      if (pathStr) {
        addInputs([pathStr]);
      }
      selectTabById?.("tab-queue");
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

  let editorDirty = false;

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

  const runTestMatch = async () => {
    const sample = testInput?.value.trim() || "";
    if (!sample) return;
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
      const policies = Array.isArray(data.dedup_policies) ? data.dedup_policies : [];

      // Populate Queue dropdowns
      if (optModelSelect instanceof HTMLSelectElement) {
        const cur = optModelSelect.value;
        clearChildren(optModelSelect);
        optModelSelect.append(
          el("option", {
            value: "",
            textContent: "Tự động phân giải theo shows.yaml (Khuyên dùng)",
          })
        );
        for (const m of models) {
          optModelSelect.append(
            el("option", {
              value: m.id,
              textContent: `${m.id} — ~${m.ref_median_fps} fps (${m.native_upscale}x)`,
            })
          );
        }
        setSelectValueSafely(optModelSelect, cur, "");
      }

      if (optDedupSelect instanceof HTMLSelectElement) {
        const curPol = optDedupSelect.value;
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

  btnTestMatch?.addEventListener("click", runTestMatch);
  testInput?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      if (ev.isComposing || ev.keyCode === 229) return;
      ev.preventDefault();
      runTestMatch();
    }
  });

  btnReloadShows?.addEventListener("click", () => {
    loadModelsAndShows(true).then(() => showToast("Đã nạp lại shows.yaml từ đĩa.", "info", 2600));
  });

  btnSaveShows?.addEventListener("click", async () => {
    if (!editor) return;
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

  loadModelsAndShows(true);

  return {
    refresh: () => loadModelsAndShows(false),
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
  const subAutoAlign = document.getElementById("sub-auto-align");
  const subPreferWhisper = document.getElementById("sub-prefer-whisper");
  const subUpgrade = document.getElementById("sub-upgrade");
  const subNoEmbed = document.getElementById("sub-no-embed");
  const subDryRun = document.getElementById("sub-dry-run");
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

  let auditLoadSeq = 0;

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
        kpiKeys.textContent = `${data.configured_keys_count} / 3 khe`;
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
        if (btnRunPending instanceof HTMLButtonElement) {
          btnRunPending.disabled = count === 0;
        }
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
            const tone = r.status === "ok" ? "badge-ok" : "badge-warn";
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
        if (!d.path) continue;
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
        ].filter((d) => Boolean(d.path));

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

  subForm?.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const target = subInputPath?.value.trim() || "";
    if (!target) {
      setSubFieldError(subInputPath, subInputError, "Vui lòng chọn tệp video hoặc phụ đề cần dịch.");
      subInputPath?.focus();
      showToast("Vui lòng chọn tệp video hoặc phụ đề cần dịch.", "warn");
      return;
    }
    setSubFieldError(subInputPath, subInputError, null);
    const rawOffset = subOffset?.value.trim() ?? "";
    const offsetMs = rawOffset === "" ? 0 : Number(rawOffset);
    if (!Number.isFinite(offsetMs) || !Number.isInteger(offsetMs)) {
      setSubFieldError(subOffset, subOffsetError, "Sub Offset (ms) phải là số nguyên hợp lệ.");
      subOffset?.focus();
      showToast("Sub Offset (ms) phải là số nguyên hợp lệ.", "warn");
      return;
    }
    setSubFieldError(subOffset, subOffsetError, null);
    const submitBtn = ev.submitter instanceof HTMLButtonElement ? ev.submitter : null;
    if (submitBtn) submitBtn.disabled = true;
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
      if (submitBtn) submitBtn.disabled = false;
    }
  });

  btnRunPending?.addEventListener("click", async () => {
    if (btnRunPending instanceof HTMLButtonElement) btnRunPending.disabled = true;
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
      if (btnRunPending instanceof HTMLButtonElement) btnRunPending.disabled = false;
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

  lastRefreshMs = Date.now();
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
  };
}
