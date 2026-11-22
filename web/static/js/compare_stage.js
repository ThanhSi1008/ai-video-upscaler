/**
 * web/static/js/compare_stage.js
 * Tab 3 — Interactive 1080p Source vs 4K AI Split-Compare Lightbox (BT.709):
 * - Draggable / keyboard-accessible split curtain (0%..100%).
 * - Split-view 200%..400% pixel-inspection Magnifier Loupe on hover.
 * - Auto-discovers candidate 4K / Preview pairs from out/ and opens mpv / native player.
 */

import { openFsPicker } from "./fs_picker.js";
import { apiFetch, clearChildren, el, showToast } from "./ui.js";

function loadHtmlImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Không thể tải ảnh khung hình: ${url}`));
    img.src = `${url}?t=${Date.now()}`;
  });
}

export function initCompareStage({ selectTabById } = {}) {
  const candidateSelect = document.getElementById("compare-candidate-select");
  const srcInput = document.getElementById("compare-src-input");
  const upInput = document.getElementById("compare-up-input");
  const tsInput = document.getElementById("compare-ts-input");
  const offsetInput = document.getElementById("compare-offset-input");
  const btnRefresh = document.getElementById("btn-refresh-candidates");
  const btnOpenMpv = document.getElementById("btn-open-4k-mpv");
  const btnBrowseSrc = document.getElementById("btn-browse-compare-src");
  const btnBrowseUp = document.getElementById("btn-browse-compare-up");
  const btnExtract = document.getElementById("btn-extract-compare");

  const wipeRange = document.getElementById("compare-wipe-range");
  const wipeLabel = document.getElementById("compare-wipe-label");
  const loupeToggle = document.getElementById("compare-loupe-toggle");
  const zoomSelect = document.getElementById("compare-zoom-select");
  const stage = document.getElementById("compare-stage");
  const canvas = document.getElementById("compare-canvas");
  const placeholder = document.getElementById("compare-placeholder");
  const metaCaption = document.getElementById("compare-meta-caption");

  const tsErrorEl = document.getElementById("compare-ts-error");
  const offsetErrorEl = document.getElementById("compare-offset-error");

  let candidatesCache = [];
  let srcImg = null;
  let upImg = null;
  let wipePct = 50;
  let pointerPos = null;
  let draggingSplit = false;
  let renderRafId = 0;
  let extractSeq = 0;
  let imageBounds = { x: 0, y: 0, w: 0, h: 0 };

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

  const renderCanvas = () => {
    const comparePanel = document.getElementById("panel-compare");
    if (comparePanel && comparePanel.hasAttribute("hidden")) {
      return;
    }
    if (!(canvas instanceof HTMLCanvasElement) || !srcImg || !upImg) return;
    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(320, Math.floor(rect.width || 1280));
    const cssH = Math.max(180, Math.floor(rect.height || 720));
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const pxW = Math.round(cssW * dpr);
    const pxH = Math.round(cssH * dpr);

    if (canvas.width !== pxW || canvas.height !== pxH) {
      canvas.width = pxW;
      canvas.height = pxH;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#070b14";
    ctx.fillRect(0, 0, cssW, cssH);

    // Compute aspect-fit rectangle inside stage
    const imgAspect = upImg.naturalWidth / Math.max(1, upImg.naturalHeight);
    const stageAspect = cssW / Math.max(1, cssH);
    let drawW = cssW;
    let drawH = cssH;
    let drawX = 0;
    let drawY = 0;

    if (imgAspect > stageAspect) {
      drawH = cssW / imgAspect;
      drawY = (cssH - drawH) / 2;
    } else {
      drawW = cssH * imgAspect;
      drawX = (cssW - drawW) / 2;
    }
    imageBounds = { x: drawX, y: drawY, w: drawW, h: drawH };

    const splitX = drawX + (wipePct / 100) * drawW;

    // 1. Draw full 4K AI frame (right side visible)
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(upImg, drawX, drawY, drawW, drawH);

    // 2. Clip left side and draw 1080p source frame
    ctx.save();
    ctx.beginPath();
    ctx.rect(drawX, drawY, Math.max(0, splitX - drawX), drawH);
    ctx.clip();
    ctx.drawImage(srcImg, drawX, drawY, drawW, drawH);
    ctx.restore();

    // 3. Draw vertical split curtain line + handle
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(splitX, drawY);
    ctx.lineTo(splitX, drawY + drawH);
    ctx.stroke();

    const handleY = drawY + drawH / 2;
    ctx.fillStyle = "rgba(9, 14, 26, 0.88)";
    ctx.beginPath();
    ctx.arc(splitX, handleY, 16, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#e8eef8";
    ctx.font = "700 12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("⇆", splitX, handleY);

    // 4. Corner labels
    const drawCornerTag = (label, x, alignRight = false) => {
      ctx.font = "700 12px system-ui, sans-serif";
      const padX = 10;
      const boxW = ctx.measureText(label).width + padX * 2;
      const boxH = 26;
      const boxX = alignRight ? x - boxW : x;
      const boxY = drawY + 12;
      ctx.fillStyle = "rgba(7, 11, 20, 0.78)";
      ctx.strokeStyle = "rgba(148, 163, 184, 0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") {
        ctx.roundRect(boxX, boxY, boxW, boxH, 6);
      } else {
        ctx.rect(boxX, boxY, boxW, boxH);
      }
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#f8fafc";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(label, boxX + padX, boxY + boxH / 2);
    };

    drawCornerTag(`1080p GỐC (${srcImg.naturalWidth}×${srcImg.naturalHeight})`, drawX + 12, false);
    drawCornerTag(`4K AI (${upImg.naturalWidth}×${upImg.naturalHeight})`, drawX + drawW - 12, true);

    // 5. Split-View Pixel Magnifier Loupe on hover
    const loupeActive =
      pointerPos &&
      !draggingSplit &&
      loupeToggle instanceof HTMLInputElement &&
      loupeToggle.checked &&
      pointerPos.x >= drawX &&
      pointerPos.x <= drawX + drawW &&
      pointerPos.y >= drawY &&
      pointerPos.y <= drawY + drawH;

    if (loupeActive) {
      const zoom = Math.max(2, Number(zoomSelect?.value || 3));
      const radius = Math.min(105, Math.max(72, Math.min(cssW, cssH) * 0.19));
      const normU = (pointerPos.x - drawX) / Math.max(1, drawW);
      const normV = (pointerPos.y - drawY) / Math.max(1, drawH);

      // Offset loupe center slightly if near edges so it stays inside canvas
      const cx = Math.max(radius + 8, Math.min(cssW - radius - 8, pointerPos.x));
      const cy = Math.max(radius + 8, Math.min(cssH - radius - 8, pointerPos.y));

      // Sample region in 4K coordinate space so zoom=2..4 inspects true 4K pixel detail
      const sampleWUp = (radius * 2) / zoom;
      const sampleHUp = (radius * 2) / zoom;
      const sxUp = normU * upImg.naturalWidth - sampleWUp / 2;
      const syUp = normV * upImg.naturalHeight - sampleHUp / 2;

      const scaleRatioX = srcImg.naturalWidth / Math.max(1, upImg.naturalWidth);
      const scaleRatioY = srcImg.naturalHeight / Math.max(1, upImg.naturalHeight);
      const sampleWSrc = sampleWUp * scaleRatioX;
      const sampleHSrc = sampleHUp * scaleRatioY;
      const sxSrc = normU * srcImg.naturalWidth - sampleWSrc / 2;
      const sySrc = normV * srcImg.naturalHeight - sampleHSrc / 2;

      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.clip();

      ctx.fillStyle = "#050811";
      ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
      ctx.imageSmoothingEnabled = false;

      // Left half of loupe = 1080p source pixels
      ctx.save();
      ctx.beginPath();
      ctx.rect(cx - radius, cy - radius, radius, radius * 2);
      ctx.clip();
      ctx.drawImage(
        srcImg,
        sxSrc,
        sySrc,
        sampleWSrc,
        sampleHSrc,
        cx - radius,
        cy - radius,
        radius * 2,
        radius * 2
      );
      ctx.restore();

      // Right half of loupe = 4K AI pixels
      ctx.save();
      ctx.beginPath();
      ctx.rect(cx, cy - radius, radius, radius * 2);
      ctx.clip();
      ctx.drawImage(
        upImg,
        sxUp,
        syUp,
        sampleWUp,
        sampleHUp,
        cx - radius,
        cy - radius,
        radius * 2,
        radius * 2
      );
      ctx.restore();

      // Divider inside loupe
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, cy - radius);
      ctx.lineTo(cx, cy + radius);
      ctx.stroke();

      ctx.fillStyle = "rgba(7, 11, 20, 0.82)";
      ctx.fillRect(cx - radius, cy + radius - 22, radius * 2, 22);
      ctx.fillStyle = "#e2e8f0";
      ctx.font = "700 10px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`1080p  |  4K (${zoom}00%)`, cx, cy + radius - 11);
      ctx.restore();

      // Outer ring of loupe
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();
  };

  const scheduleRender = () => {
    if (renderRafId) return;
    renderRafId = window.requestAnimationFrame(() => {
      renderRafId = 0;
      renderCanvas();
    });
  };

  const setWipe = (pct, { coalesce = false } = {}) => {
    wipePct = Math.max(0, Math.min(100, Number(pct)));
    if (wipeRange instanceof HTMLInputElement) {
      wipeRange.value = String(Math.round(wipePct));
    }
    if (wipeLabel) {
      wipeLabel.textContent = `${Math.round(wipePct)}%`;
    }
    stage?.style.setProperty("--wipe-pct", `${wipePct}%`);
    if (coalesce) {
      scheduleRender();
    } else {
      renderCanvas();
    }
  };

  const loadCandidates = async () => {
    if (!(candidateSelect instanceof HTMLSelectElement)) return;
    try {
      const data = await apiFetch("/api/preview/candidates");
      candidatesCache = Array.isArray(data.candidates) ? data.candidates : [];
      const prevVal = candidateSelect.value;
      clearChildren(candidateSelect);
      candidateSelect.append(
        el("option", {
          value: "",
          textContent:
            candidatesCache.length > 0
              ? `-- Chọn nhanh trong ${candidatesCache.length} bản 4K / Preview tại out/ --`
              : "-- Chưa có file _4K nào trong out/ --",
        })
      );
      for (const c of candidatesCache) {
        const tag = c.is_preview ? `[Preview ${c.preview_seconds}s]` : "[Full 4K]";
        candidateSelect.append(
          el("option", {
            value: c.upscaled_path,
            textContent: `${tag} ${c.name} (${c.size_mb} MB)`,
          })
        );
      }
      if (prevVal && candidatesCache.some((x) => x.upscaled_path === prevVal)) {
        candidateSelect.value = prevVal;
      }
    } catch {
      // Non-fatal on startup
    }
  };

  const extractAndCompare = async () => {
    const srcPath = srcInput?.value.trim() || "";
    const upPath = upInput?.value.trim() || "";
    if (!srcPath || !upPath) {
      showToast("Vui lòng chọn đủ đường dẫn Video gốc (1080p) và Video thành phẩm (4K).", "warn");
      return;
    }

    const rawTs = tsInput?.value.trim() ?? "";
    const tsSec = rawTs === "" ? 2.0 : Number(rawTs);
    if (!Number.isFinite(tsSec) || tsSec < 0) {
      setFieldError(tsInput, tsErrorEl, "Mốc thời gian phải là số giây hợp lệ (>= 0).");
      tsInput?.focus();
      showToast("Mốc thời gian phải là số giây hợp lệ (>= 0).", "warn");
      return;
    }
    setFieldError(tsInput, tsErrorEl, null);

    const rawOffset = offsetInput?.value.trim() || "";
    const offsetVal = rawOffset === "" ? null : Number(rawOffset);
    if (offsetVal !== null && !Number.isFinite(offsetVal)) {
      setFieldError(offsetInput, offsetErrorEl, "Độ lệch gốc phải là số giây hữu hạn hợp lệ.");
      offsetInput?.focus();
      showToast("Độ lệch gốc (offset) phải là số giây hữu hạn hợp lệ.", "warn");
      return;
    }
    setFieldError(offsetInput, offsetErrorEl, null);

    const reqId = ++extractSeq;
    const isCurrentExtract = () => reqId === extractSeq;

    if (btnExtract instanceof HTMLButtonElement) {
      btnExtract.disabled = true;
      btnExtract.setAttribute("aria-busy", "true");
    }
    if (metaCaption) {
      metaCaption.textContent = "Đang trích xuất cặp khung hình chuẩn ma trận màu BT.709…";
    }

    try {
      const res = await apiFetch("/api/preview/compare-frames", {
        method: "POST",
        body: {
          source_path: srcPath,
          upscaled_path: upPath,
          timestamp_sec: tsSec,
          source_offset_sec: offsetVal,
        },
      });
      if (!isCurrentExtract()) return;

      const [loadedSrc, loadedUp] = await Promise.all([
        loadHtmlImage(res.source_frame_url),
        loadHtmlImage(res.upscaled_frame_url),
      ]);
      if (!isCurrentExtract()) return;

      srcImg = loadedSrc;
      upImg = loadedUp;
      if (placeholder) {
        placeholder.hidden = true;
      }
      if (metaCaption) {
        metaCaption.textContent =
          `Đã khớp: Gốc ${res.source_meta.width}×${res.source_meta.height} @ ${res.source_timestamp_sec}s ` +
          `⇄ Thành phẩm ${res.upscaled_meta.width}×${res.upscaled_meta.height} @ ${res.upscaled_timestamp_sec}s ` +
          `(Bù lệch gốc: ${res.source_offset_sec}s • Chuẩn màu BT.709 PC Full-Range)`;
      }
      renderCanvas();
    } catch (err) {
      if (!isCurrentExtract()) return;
      showToast(err.message, "error");
      if (metaCaption) {
        metaCaption.textContent = `Không thể trích xuất khung hình: ${err.message}`;
      }
    } finally {
      if (isCurrentExtract() && btnExtract instanceof HTMLButtonElement) {
        btnExtract.disabled = false;
        btnExtract.removeAttribute("aria-busy");
      }
    }
  };

  candidateSelect?.addEventListener("change", () => {
    const chosenUp = candidateSelect.value;
    const item = candidatesCache.find((c) => c.upscaled_path === chosenUp);
    if (!item) return;
    if (upInput) upInput.value = item.upscaled_path;
    if (srcInput) srcInput.value = item.source_path || "";
    if (offsetInput) offsetInput.value = "";
    if (srcInput?.value && upInput?.value) {
      extractAndCompare();
    } else {
      ++extractSeq;
      if (btnExtract instanceof HTMLButtonElement) {
        btnExtract.disabled = false;
        btnExtract.removeAttribute("aria-busy");
      }
      if (metaCaption) {
        metaCaption.textContent =
          "Tệp 4K này chưa tự khớp được video gốc 1080p — hãy chọn video gốc thủ công rồi nhấn Trích xuất.";
      }
    }
  });

  btnRefresh?.addEventListener("click", () => {
    loadCandidates().then(() => showToast("Đã cập nhật danh sách video trong out/.", "info", 2800));
  });

  btnBrowseSrc?.addEventListener("click", async () => {
    const picked = await openFsPicker({
      title: "Chọn Video Gốc (1080p)",
      multi: false,
      allowFolders: false,
      allow4k: false,
    });
    if (picked.length > 0 && srcInput) {
      srcInput.value = picked[0];
    }
  });

  btnBrowseUp?.addEventListener("click", async () => {
    const picked = await openFsPicker({
      title: "Chọn Video Thành phẩm 4K",
      multi: false,
      allowFolders: false,
      allow4k: true,
      initialPath: "out",
    });
    if (picked.length > 0 && upInput) {
      upInput.value = picked[0];
    }
  });

  btnOpenMpv?.addEventListener("click", async () => {
    const target = upInput?.value.trim() || candidateSelect?.value || "";
    if (!target) {
      showToast("Chưa có đường dẫn file 4K để mở trình phát.", "warn");
      return;
    }
    try {
      const res = await apiFetch("/api/open-media", {
        method: "POST",
        body: { path: target, prefer_mpv: true },
      });
      showToast(`Đã mở trình phát (${res.launcher}) cho bản 4K.`, "success", 3200);
    } catch (err) {
      showToast(err.message, "error");
    }
  });

  tsInput?.addEventListener("input", () => setFieldError(tsInput, tsErrorEl, null));
  offsetInput?.addEventListener("input", () => setFieldError(offsetInput, offsetErrorEl, null));

  btnExtract?.addEventListener("click", extractAndCompare);
  wipeRange?.addEventListener("input", () => setWipe(wipeRange.value));
  loupeToggle?.addEventListener("change", renderCanvas);
  zoomSelect?.addEventListener("change", renderCanvas);

  // Pointer interactions on compare stage (drag = move curtain, hover = move loupe)
  if (stage && canvas instanceof HTMLCanvasElement) {
    const updateFromEvent = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      pointerPos = { x, y };
      if (draggingSplit && rect.width > 0) {
        const effW = imageBounds.w > 0 ? imageBounds.w : rect.width;
        const effX = imageBounds.w > 0 ? x - imageBounds.x : x;
        setWipe((effX / Math.max(1, effW)) * 100, { coalesce: true });
      } else {
        scheduleRender();
      }
    };

    stage.addEventListener("pointerdown", (ev) => {
      if (!srcImg || !upImg) return;
      if (ev.pointerType === "mouse" && ev.button !== 0) return;
      draggingSplit = true;
      try {
        stage.setPointerCapture?.(ev.pointerId);
      } catch {
        // Synthetic or non-capturable pointer
      }
      updateFromEvent(ev);
    });
    stage.addEventListener("pointermove", (ev) => {
      if (!srcImg || !upImg) return;
      updateFromEvent(ev);
    });
    const endDrag = (ev) => {
      draggingSplit = false;
      try {
        if (stage.hasPointerCapture?.(ev.pointerId)) {
          stage.releasePointerCapture(ev.pointerId);
        }
      } catch {
        // Ignore if pointer was not captured
      }
      scheduleRender();
    };
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);
    stage.addEventListener("pointerleave", () => {
      draggingSplit = false;
      pointerPos = null;
      scheduleRender();
    });

    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => scheduleRender()).observe(stage);
    }
  }

  loadCandidates();

  return {
    refreshCandidates: loadCandidates,
    redraw: renderCanvas,
    async openPair(sourcePath, upscaledPath) {
      if (srcInput) srcInput.value = sourcePath || "";
      if (upInput) upInput.value = upscaledPath || "";
      if (offsetInput) offsetInput.value = "";
      selectTabById?.("tab-compare");
      await loadCandidates();
      if (sourcePath && upscaledPath) {
        await extractAndCompare();
      }
    },
  };
}
