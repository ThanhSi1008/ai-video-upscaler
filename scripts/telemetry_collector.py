#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/telemetry_collector.py
Continuous hardware telemetry sampler for ai-video-upscaler benchmarking.

Samples 17 continuous metrics at fixed intervals (default 1.0s):
- timestamp (ISO-8601)
- elapsed_s (float seconds since start)
- pstate (NVIDIA P-State: P0..P8 or N/A)
- gpu_util_pct (GPU core compute utilization %)
- mem_util_pct (GPU memory controller utilization %)
- enc_util_pct (NVENC hardware video encoder utilization %)
- dec_util_pct (NVDEC hardware video decoder utilization %)
- mem_clock_mhz (GPU memory clock in MHz)
- core_clock_mhz (GPU graphics core clock in MHz)
- power_w (GPU power draw in Watts)
- temp_c (GPU temperature in Celsius)
- vram_used_mib (Dedicated VRAM used in MiB)
- vram_free_mib (Dedicated VRAM free in MiB)
- cpu_pct (Host CPU utilization %)
- ram_used_gb (Host system RAM used in GB)
- ssd_read_mb_s (Disk read throughput in MB/s)
- ssd_write_mb_s (Disk write throughput in MB/s)

Thread-safe, non-blocking, exception-safe.
Gracefully handles absent or mocked GPU query environments.
"""

from __future__ import annotations

import csv
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

TELEMETRY_HEADER: List[str] = [
    "timestamp",
    "elapsed_s",
    "pstate",
    "gpu_util_pct",
    "mem_util_pct",
    "enc_util_pct",
    "dec_util_pct",
    "mem_clock_mhz",
    "core_clock_mhz",
    "power_w",
    "temp_c",
    "vram_used_mib",
    "vram_free_mib",
    "cpu_pct",
    "ram_used_gb",
    "ssd_read_mb_s",
    "ssd_write_mb_s",
]


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert strings or numbers to float."""
    if val is None:
        return default
    try:
        cleaned = str(val).strip()
        if not cleaned or cleaned.upper() in {"N/A", "[NOT SUPPORTED]", "UNKNOWN", "ERR"}:
            return default
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def default_gpu_query() -> Dict[str, Any]:
    """
    Query NVIDIA GPU hardware state using nvidia-smi.
    Falls back gracefully to safe default values when nvidia-smi is unavailable.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=pstate,utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder,clocks.current.memory,clocks.current.graphics,power.draw,temperature.gpu,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        # Use STARTUPINFO on Windows to prevent console flashing if detached
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
            startupinfo=startupinfo,
        )
        line = res.stdout.strip().splitlines()
        if not line:
            raise ValueError("Empty output from nvidia-smi")
        parts = [p.strip() for p in line[0].split(",")]
        if len(parts) < 11:
            raise ValueError(f"Incomplete fields from nvidia-smi: {parts}")

        return {
            "pstate": parts[0] if parts[0] else "N/A",
            "gpu_util_pct": _safe_float(parts[1]),
            "mem_util_pct": _safe_float(parts[2]),
            "enc_util_pct": _safe_float(parts[3]),
            "dec_util_pct": _safe_float(parts[4]),
            "mem_clock_mhz": _safe_float(parts[5]),
            "core_clock_mhz": _safe_float(parts[6]),
            "power_w": _safe_float(parts[7]),
            "temp_c": _safe_float(parts[8]),
            "vram_used_mib": _safe_float(parts[9]),
            "vram_free_mib": _safe_float(parts[10]),
        }
    except Exception:
        return {
            "pstate": "N/A",
            "gpu_util_pct": 0.0,
            "mem_util_pct": 0.0,
            "enc_util_pct": 0.0,
            "dec_util_pct": 0.0,
            "mem_clock_mhz": 0.0,
            "core_clock_mhz": 0.0,
            "power_w": 0.0,
            "temp_c": 0.0,
            "vram_used_mib": 0.0,
            "vram_free_mib": 0.0,
        }


class TelemetryCollector:
    """
    Background thread continuous hardware telemetry collector.
    Collects 17 metrics at specified intervals and streams to CSV.
    """

    def __init__(
        self,
        output_csv_path: str | Path,
        interval_sec: float = 1.0,
        gpu_query_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.output_csv_path = Path(output_csv_path).resolve()
        self.interval_sec = max(0.1, float(interval_sec))
        self.gpu_query_fn = gpu_query_fn or default_gpu_query

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._samples: List[Dict[str, Any]] = []
        self._start_time: float = 0.0
        self._csv_file = None
        self._csv_writer = None
        self._summary: Dict[str, Any] = {}
        self._micro_timings: Optional[Dict[str, Any]] = None

        # Delta tracking for disk I/O
        self._last_disk_time: float = 0.0
        self._last_read_bytes: int = 0
        self._last_write_bytes: int = 0

    def record_micro_timings(self, timings: Dict[str, Any]) -> None:
        """Record GPU micro-timing dictionary (e.g. from CUDA Events in inference pipeline)."""
        with self._lock:
            self._micro_timings = dict(timings)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def samples(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._samples)

    @property
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._summary)

    def start(self) -> TelemetryCollector:
        """Start the background sampling thread."""
        with self._lock:
            if self.is_running:
                return self

            self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self.output_csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=TELEMETRY_HEADER)
            self._csv_writer.writeheader()
            self._csv_file.flush()

            self._samples.clear()
            self._summary.clear()
            self._stop_event.clear()
            self._start_time = time.perf_counter()

            # Initialize host baseline counters
            self._init_host_counters()

            self._thread = threading.Thread(
                target=self._run,
                name="TelemetryCollectorThread",
                daemon=True,
            )
            self._thread.start()
            return self

    def _init_host_counters(self) -> None:
        """Prime cpu_percent and disk_io_counters baseline."""
        try:
            import psutil
            psutil.cpu_percent(interval=None)
            io = psutil.disk_io_counters()
            if io is not None:
                self._last_read_bytes = io.read_bytes
                self._last_write_bytes = io.write_bytes
                self._last_disk_time = time.perf_counter()
        except Exception:
            pass

    def _sample_host_metrics(self) -> Dict[str, float]:
        """Sample host CPU %, system RAM, and disk read/write throughput."""
        cpu_pct = 0.0
        ram_used_gb = 0.0
        ssd_read_mb_s = 0.0
        ssd_write_mb_s = 0.0

        try:
            import psutil
            cpu_pct = round(float(psutil.cpu_percent(interval=None)), 1)
            mem = psutil.virtual_memory()
            ram_used_gb = round(float(mem.used) / (1024 ** 3), 2)

            now = time.perf_counter()
            io = psutil.disk_io_counters()
            if io is not None and self._last_disk_time > 0:
                dt = now - self._last_disk_time
                if dt > 0.05:
                    d_read = max(0, io.read_bytes - self._last_read_bytes)
                    d_write = max(0, io.write_bytes - self._last_write_bytes)
                    ssd_read_mb_s = round(d_read / (dt * 1024 * 1024), 2)
                    ssd_write_mb_s = round(d_write / (dt * 1024 * 1024), 2)
                    self._last_read_bytes = io.read_bytes
                    self._last_write_bytes = io.write_bytes
                    self._last_disk_time = now
            elif io is not None:
                self._last_read_bytes = io.read_bytes
                self._last_write_bytes = io.write_bytes
                self._last_disk_time = now
        except Exception:
            pass

        return {
            "cpu_pct": cpu_pct,
            "ram_used_gb": ram_used_gb,
            "ssd_read_mb_s": ssd_read_mb_s,
            "ssd_write_mb_s": ssd_write_mb_s,
        }

    def _collect_one_sample(self) -> Dict[str, Any]:
        """Collect a single 17-metric hardware snapshot."""
        now_dt = datetime.now()
        elapsed = round(max(0.0, time.perf_counter() - self._start_time), 2)
        sample: Dict[str, Any] = {
            "timestamp": now_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": elapsed,
        }

        # GPU metrics
        try:
            gpu_data = self.gpu_query_fn()
        except Exception:
            gpu_data = default_gpu_query()

        for key in (
            "pstate", "gpu_util_pct", "mem_util_pct", "enc_util_pct", "dec_util_pct",
            "mem_clock_mhz", "core_clock_mhz", "power_w", "temp_c", "vram_used_mib", "vram_free_mib"
        ):
            sample[key] = gpu_data.get(key, 0.0 if key != "pstate" else "N/A")

        # Host metrics
        host_data = self._sample_host_metrics()
        sample.update(host_data)
        return sample

    def _run(self) -> None:
        """Main sampling loop."""
        # Initial sample immediately upon start
        first_sample = self._collect_one_sample()
        self._record_sample(first_sample)

        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self.interval_sec):
                break
            try:
                sample = self._collect_one_sample()
                self._record_sample(sample)
            except Exception:
                # Never let thread crash
                pass

    def _record_sample(self, sample: Dict[str, Any]) -> None:
        """Write sample to CSV and in-memory list under lock."""
        with self._lock:
            self._samples.append(sample)
            if self._csv_writer and self._csv_file:
                try:
                    self._csv_writer.writerow(sample)
                    self._csv_file.flush()
                except Exception:
                    pass

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and compute aggregated summary."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        with self._lock:
            if self._csv_file:
                try:
                    self._csv_file.flush()
                    self._csv_file.close()
                except Exception:
                    pass
                self._csv_file = None
                self._csv_writer = None

            self._summary = self._compute_summary()
            return dict(self._summary)

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute aggregated statistical summary over all collected samples."""
        if not self._samples:
            return {
                "samples_count": 0,
                "duration_s": 0.0,
                "peak_vram_mib": 0.0,
                "avg_vram_mib": 0.0,
                "avg_gpu_util": 0.0,
                "peak_gpu_util": 0.0,
                "avg_mem_util": 0.0,
                "avg_enc_util": 0.0,
                "peak_enc_util": 0.0,
                "avg_dec_util": 0.0,
                "avg_cpu_pct": 0.0,
                "peak_cpu_pct": 0.0,
                "avg_ram_used_gb": 0.0,
                "peak_ram_used_gb": 0.0,
                "avg_power_w": 0.0,
                "peak_power_w": 0.0,
                "peak_temp_c": 0.0,
                "avg_ssd_read_mb_s": 0.0,
                "avg_ssd_write_mb_s": 0.0,
                "csv_path": str(self.output_csv_path),
            }

        count = len(self._samples)
        last_elapsed = self._samples[-1].get("elapsed_s", 0.0)

        def _mean(k: str) -> float:
            vals = [_safe_float(s.get(k, 0.0)) for s in self._samples]
            return round(sum(vals) / max(1, len(vals)), 2)

        def _max(k: str) -> float:
            vals = [_safe_float(s.get(k, 0.0)) for s in self._samples]
            return round(max(vals) if vals else 0.0, 2)

        res = {
            "samples_count": count,
            "duration_s": round(float(last_elapsed), 2),
            "peak_vram_mib": _max("vram_used_mib"),
            "avg_vram_mib": _mean("vram_used_mib"),
            "avg_gpu_util": _mean("gpu_util_pct"),
            "peak_gpu_util": _max("gpu_util_pct"),
            "avg_mem_util": _mean("mem_util_pct"),
            "avg_enc_util": _mean("enc_util_pct"),
            "peak_enc_util": _max("enc_util_pct"),
            "avg_dec_util": _mean("dec_util_pct"),
            "avg_cpu_pct": _mean("cpu_pct"),
            "peak_cpu_pct": _max("cpu_pct"),
            "avg_ram_used_gb": _mean("ram_used_gb"),
            "peak_ram_used_gb": _max("ram_used_gb"),
            "avg_power_w": _mean("power_w"),
            "peak_power_w": _max("power_w"),
            "peak_temp_c": _max("temp_c"),
            "avg_ssd_read_mb_s": _mean("ssd_read_mb_s"),
            "avg_ssd_write_mb_s": _mean("ssd_write_mb_s"),
            "csv_path": str(self.output_csv_path),
        }
        if self._micro_timings:
            res["micro_timings"] = dict(self._micro_timings)
        return res

    def __enter__(self) -> TelemetryCollector:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
