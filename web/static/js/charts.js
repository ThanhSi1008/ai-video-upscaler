// -*- coding: utf-8 -*-
/**
 * web/static/js/charts.js
 * High-DPI Canvas 2D 120-second sliding telemetry charts:
 * - Zero external chart libraries, crisp devicePixelRatio scaling, ResizeObserver responsive.
 * - Reads active theme colors directly from computed CSS tokens (Light / Dark / Forced Colors).
 */

export class TelemetryChartsController {
  constructor() {
    this.maxPoints = 120;
    this.history = [];
    this.fpsHistory = [];
    this.canvases = {
      gpu: document.getElementById("chart-gpu"),
      mem: document.getElementById("chart-mem"),
      power: document.getElementById("chart-power"),
      io: document.getElementById("chart-io"),
    };
    this.readouts = {
      gpu: document.getElementById("chart-val-gpu"),
      mem: document.getElementById("chart-val-mem"),
      power: document.getElementById("chart-val-power"),
      io: document.getElementById("chart-val-io"),
    };

    this._rafId = 0;
    this._pendingLiveData = null;

    if (typeof ResizeObserver === "function") {
      const gridEl =
        document.querySelector(".charts-grid") ||
        Object.values(this.canvases).find((c) => c instanceof HTMLCanvasElement);
      if (gridEl) {
        this._ro = new ResizeObserver(() => this.scheduleRedraw());
        this._ro.observe(gridEl);
      }
    }
  }

  scheduleRedraw(liveData = null) {
    if (liveData !== null) {
      this._pendingLiveData = liveData;
    }
    if (this._rafId) return;
    this._rafId = window.requestAnimationFrame(() => {
      this._rafId = 0;
      const ld = this._pendingLiveData;
      this._pendingLiveData = null;
      this.redrawAll(ld);
    });
  }

  setSnapshot(historySamples, liveData) {
    const arr = Array.isArray(historySamples) ? historySamples.slice(-this.maxPoints) : [];
    const prevFps = Array.isArray(this.fpsHistory) ? this.fpsHistory.slice() : [];
    this.history = arr;
    const baseFps = Number(liveData?.steady_fps || liveData?.latest_fps || 0);
    if (arr.length === 0) {
      this.fpsHistory = [];
    } else if (prevFps.length > 0) {
      // Bảo toàn chuỗi FPS lịch sử qua các lần reconnect SSE / đồng bộ snapshot
      const priorSlots = arr.length - 1;
      const kept = priorSlots > 0 ? prevFps.slice(-priorSlots) : [];
      const padCount = Math.max(0, priorSlots - kept.length);
      this.fpsHistory = [
        ...arr.slice(0, padCount).map((s) => Number(s?.fps_est ?? s?.fps ?? 0) || 0),
        ...kept,
        baseFps,
      ];
    } else {
      this.fpsHistory = arr.map((s, idx) =>
        idx === arr.length - 1 ? baseFps : Number(s?.fps_est ?? s?.fps ?? 0) || 0
      );
    }
    this.redrawAll(liveData);
  }

  pushTick(sample, liveData) {
    if (!sample || typeof sample !== "object") return;
    this.history.push(sample);
    if (this.history.length > this.maxPoints) {
      this.history.shift();
    }
    const fpsVal = Number(liveData?.steady_fps || liveData?.latest_fps || 0);
    this.fpsHistory.push(fpsVal);
    if (this.fpsHistory.length > this.maxPoints) {
      this.fpsHistory.shift();
    }
    this.redrawAll(liveData);
  }

  _palette() {
    if (window.matchMedia?.("(forced-colors: active)")?.matches) {
      const bodyCs = getComputedStyle(document.body);
      const readoutEl = this.readouts.gpu || document.querySelector(".chart-readout");
      const readoutCs = readoutEl ? getComputedStyle(readoutEl) : bodyCs;
      const mutedEl = document.querySelector(".muted-meta");
      const mutedCs = mutedEl ? getComputedStyle(mutedEl) : bodyCs;
      const cardEl = document.querySelector(".chart-card");
      const cardCs = cardEl ? getComputedStyle(cardEl) : bodyCs;
      const fg = bodyCs.color || "CanvasText";
      const accent = readoutCs.color || "Highlight";
      return {
        accent,
        success: fg,
        warning: accent,
        danger: fg,
        muted: mutedCs.color || "GrayText",
        grid: cardCs.borderTopColor || "ButtonBorder",
        bg: cardCs.backgroundColor || "Canvas",
      };
    }

    const cs = getComputedStyle(document.documentElement);
    const metaContent = (
      document.querySelector('meta[name="color-scheme"]')?.getAttribute("content") || ""
    )
      .trim()
      .toLowerCase();
    const prefersDark = Boolean(window.matchMedia?.("(prefers-color-scheme: dark)")?.matches);
    const isDark = metaContent === "dark" || (metaContent !== "light" && prefersDark);
    const suffix = isDark ? "dark" : "light";
    const pick = (base, fallback) => {
      const concrete = cs.getPropertyValue(`${base}-${suffix}`).trim();
      if (concrete && !concrete.startsWith("light-dark(")) {
        return concrete;
      }
      const raw = cs.getPropertyValue(base).trim();
      if (raw && !raw.startsWith("light-dark(")) {
        return raw;
      }
      return fallback;
    };

    return {
      accent: pick("--accent", isDark ? "#38bdf8" : "#0058b8"),
      success: pick("--success", isDark ? "#34d399" : "#0f766e"),
      warning: pick("--warning", isDark ? "#fbbf24" : "#b45309"),
      danger: pick("--danger", isDark ? "#f87171" : "#b91c1c"),
      muted: pick("--fg-muted", isDark ? "#9fb0cc" : "#475569"),
      grid: pick("--border-subtle", isDark ? "#273759" : "#cbd5e1"),
      bg: pick("--bg-elevated", isDark ? "#18233c" : "#eaf0f8"),
    };
  }

  redrawAll(liveData = null) {
    const monitorPanel = document.getElementById("panel-monitor");
    if (monitorPanel && monitorPanel.hasAttribute("hidden")) {
      return;
    }

    const latest = this.history.length > 0 ? this.history[this.history.length - 1] : null;
    const pal = this._palette();

    if (latest) {
      if (this.readouts.gpu) {
        this.readouts.gpu.textContent = `${Number(latest.gpu_util_pct || 0).toFixed(0)}% GPU / ${Number(latest.enc_util_pct || 0).toFixed(0)}% NVENC`;
      }
      if (this.readouts.mem) {
        this.readouts.mem.textContent = `${Math.round(Number(latest.vram_used_mib || 0))} MiB / ${Number(latest.ram_used_gb || 0).toFixed(1)} GB`;
      }
      if (this.readouts.power) {
        this.readouts.power.textContent = `${Number(latest.power_w || 0).toFixed(1)} W / ${Number(latest.temp_c || 0).toFixed(0)} °C`;
      }
      const curFps =
        liveData !== null
          ? Number(liveData?.steady_fps || liveData?.latest_fps || 0)
          : this.fpsHistory[this.fpsHistory.length - 1] || 0;
      const diskRate = Number(latest.ssd_read_mb_s || 0) + Number(latest.ssd_write_mb_s || 0);
      if (this.readouts.io) {
        this.readouts.io.textContent = `${curFps.toFixed(2)} fps / ${diskRate.toFixed(1)} MB/s`;
      }
    }

    // 1. GPU Compute % & NVENC %
    this._drawDualChart(this.canvases.gpu, {
      seriesA: this.history.map((s) => Number(s.gpu_util_pct || 0)),
      seriesB: this.history.map((s) => Number(s.enc_util_pct || 0)),
      maxA: 100,
      maxB: 100,
      colorA: pal.accent,
      colorB: pal.success,
      labelA: "GPU %",
      labelB: "NVENC %",
      yTopLabel: "100%",
      pal,
    });

    // 2. VRAM % & System RAM %
    this._drawDualChart(this.canvases.mem, {
      seriesA: this.history.map((s) => {
        const tot = Math.max(1, Number(s.vram_total_mib || 12288));
        return (Number(s.vram_used_mib || 0) / tot) * 100;
      }),
      seriesB: this.history.map((s) => {
        const tot = Math.max(1, Number(s.ram_total_gb || 32));
        return (Number(s.ram_used_gb || 0) / tot) * 100;
      }),
      maxA: 100,
      maxB: 100,
      colorA: pal.accent,
      colorB: pal.warning,
      labelA: "VRAM %",
      labelB: "RAM %",
      yTopLabel: "100%",
      pal,
    });

    // 3. Power (W, 0..185W) & Temp (°C, 0..95°C)
    this._drawDualChart(this.canvases.power, {
      seriesA: this.history.map((s) => Number(s.power_w || 0)),
      seriesB: this.history.map((s) => Number(s.temp_c || 0)),
      maxA: 185,
      maxB: 95,
      colorA: pal.warning,
      colorB: pal.danger,
      labelA: "Power (W)",
      labelB: "Temp (°C)",
      yTopLabel: "185W / 95°C",
      pal,
    });

    // 4. AI FPS & NVMe R+W MB/s
    const ioSeries = this.history.map(
      (s) => Number(s.ssd_read_mb_s || 0) + Number(s.ssd_write_mb_s || 0)
    );
    const maxFps = Math.max(25, ...this.fpsHistory, 1);
    const maxIo = Math.max(100, ...ioSeries, 1);
    this._drawDualChart(this.canvases.io, {
      seriesA: this.fpsHistory,
      seriesB: ioSeries,
      maxA: maxFps,
      maxB: maxIo,
      colorA: pal.success,
      colorB: pal.accent,
      labelA: "AI FPS",
      labelB: "NVMe MB/s",
      yTopLabel: `${Math.round(maxFps)} fps`,
      pal,
    });
  }

  _drawDualChart(canvas, cfg) {
    if (!(canvas instanceof HTMLCanvasElement)) return;
    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(240, Math.floor(rect.width || canvas.width || 520));
    const cssH = Math.max(120, Math.floor(rect.height || canvas.height || 170));
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
    ctx.clearRect(0, 0, cssW, cssH);

    const padLeft = 10;
    const padRight = 10;
    const padTop = 22;
    const padBottom = 18;
    const plotW = cssW - padLeft - padRight;
    const plotH = cssH - padTop - padBottom;

    // Horizontal grid lines (25%, 50%, 75%)
    ctx.strokeStyle = cfg.pal.grid;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    for (const frac of [0.25, 0.5, 0.75]) {
      const y = padTop + plotH * frac;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    const drawSeries = (values, maxVal, color, fillAlpha, dashed = false) => {
      if (!values || values.length === 0) return;
      const denom = Math.max(1, this.maxPoints - 1);
      const startOffset = this.maxPoints - values.length;

      const points = values.map((v, i) => {
        const clamped = Math.max(0, Math.min(maxVal, Number(v || 0)));
        const x = padLeft + ((startOffset + i) / denom) * plotW;
        const y = padTop + plotH - (clamped / Math.max(0.001, maxVal)) * plotH;
        return { x, y };
      });

      if (fillAlpha > 0 && points.length >= 2) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(points[0].x, padTop + plotH);
        for (const pt of points) {
          ctx.lineTo(pt.x, pt.y);
        }
        ctx.lineTo(points[points.length - 1].x, padTop + plotH);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.globalAlpha = fillAlpha;
        ctx.fill();
        ctx.restore();
      }

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      if (dashed) {
        ctx.setLineDash([5, 3]);
      }
      ctx.beginPath();
      points.forEach((pt, idx) => {
        if (idx === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.stroke();

      // Latest sample dot
      const lastPt = points[points.length - 1];
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(lastPt.x, lastPt.y, 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    };

    drawSeries(cfg.seriesA, cfg.maxA, cfg.colorA, 0.14, false);
    drawSeries(cfg.seriesB, cfg.maxB, cfg.colorB, 0.0, true);

    // Compact legend + axis markers
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.fillStyle = cfg.colorA;
    ctx.fillText(`● ${cfg.labelA}`, padLeft + 2, 14);
    const wA = ctx.measureText(`● ${cfg.labelA}`).width;
    ctx.fillStyle = cfg.colorB;
    ctx.fillText(`┄ ${cfg.labelB}`, padLeft + wA + 14, 14);

    ctx.fillStyle = cfg.pal.muted;
    ctx.textAlign = "right";
    ctx.fillText(cfg.yTopLabel, cssW - padRight, 14);
    ctx.fillText("Hiện tại (0s)", cssW - padRight, cssH - 4);
    ctx.textAlign = "left";
    ctx.fillText("-120s", padLeft, cssH - 4);

    ctx.restore();
  }
}
