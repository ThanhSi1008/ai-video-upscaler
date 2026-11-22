# Audit, Output Sanity Guard & Safe Runtime Architecture

> **Production Quality Verification, Disk Backpressure & Hardware Locks**
> **Primary Contract:** [GEMINI.md](../GEMINI.md) | **Historical Measurements:** [engineering-log.md](archive/reports/engineering-log.md)

## 1. Core Modules

| Module | Responsibility |
| :--- | :--- |
| [`src/output_sanity.py`](../src/output_sanity.py) | Output Sanity Guard: $|\Delta Y| \le 6/255$ and Pearson $r \ge 0.98$ (supports disk images and `270x480` GPU thumbnails) |
| [`src/mux_audit.py`](../src/mux_audit.py) | Parallel `ThreadPoolExecutor` Mux Audit: frame count, 10-bit & BT.709 Limited VUI (`video_vui_audit`), 1-pass `-f streamhash` audio MD5, subtitles, fonts, interleaving |
| [`src/cleanup.py`](../src/cleanup.py) | Auto cleanup of `.temp_<basename>` after Mux `PASS` and artifact organization into `audit/` |
| [`src/preflight.py`](../src/preflight.py) | Alder Lake P-Core QoS lock, Win32 `SetThreadExecutionState` AntiSleep, AC power & disk checks |
| [`src/config.py`](../src/config.py) | PID-validated `.gpu_active.lock` and `atexit` cleanup |
| [`src/night_report.py`](../src/night_report.py) | Overnight telemetry and dedup summary reporter |

## 2. Output Sanity Guard (`src/output_sanity.py`)

Protects against black/white/corrupted model outputs before encoding an entire episode:
1. **Trigger Schedule:** Runs on chunk `0` and every `10` chunks thereafter.
2. **Zero-Reinference in Zero-Disk Mode:** `DoubleBufferedPipeUpscaler` downsamples the Batch-0 4K tensor on GPU via `F.avg_pool2d` (`480x270`) and transfers asynchronously on `stream_d2h`.
3. **Hard Thresholds:** $|\Delta Y| \le 6.0 / 255$ and Pearson $r(Y_{4K}, Y_{1080p}) \ge 0.98$.
4. **Failure Action:** Aborts chunk encoding, writes failure diagnostics via `write_failed_sanity_audit`, and raises `OutputSanityError` before muxing.

## 3. Parallel Mux Audit (`src/mux_audit.py`)

Executed by `verify_mux_integrity` right after `mux_segments_directly`:
1. **Exact Frame Count:** Reuses probed source metadata and fails on any $\pm 1$ `nb_frames` mismatch.
2. **1-Pass Multi-Stream Audio MD5 (`get_all_audio_streams_md5`):** Runs `-map 0:a -c copy -f streamhash -hash md5 -` to verify every audio track in one sequential container pass.
3. **Parallel Packet Interleave Check (`check_window_interleaving`):** Probes 5 time windows concurrently to confirm `-max_interleave_delta 0` prevented audio clump dropouts.
4. **Metadata Parity:** Confirms 100% parity on subtitle streams, chapters, and embedded TTF/OTF fonts; only `status == "PASS"` triggers `os.replace` from `*.mux_tmp.mkv` and cleanup.
5. **10-Bit & BT.709 Limited VUI Audit (`video_vui_audit`):** `probe_video` (PyAV + `ffprobe` parity) and `verify_mux_integrity` enforce 10-bit pixel format (`yuv420p10le`/`p010le`) and BT.709 Limited VUI tags (`color_range=tv`, `color_space=bt709`, `color_primaries=bt709`, `color_transfer=bt709`) on upscaled/HEVC outputs (preserving backward-compatible passthrough for legacy unit mocks omitting color keys and un-reencoded non-HEVC `-c:v copy` fixtures). Disk-mode FFmpeg `scale` filtergraphs (`encode_chunk_hevc`, `encode_dedup_chunk_hevc`) append `setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709` so FFmpeg 8.x preserves all four VUI tags.

## 4. Storage Cleanup & `ENOSPC` Backpressure

- **Auto Cleanup (`src/cleanup.py`):** Deletes `.temp_<basename>` on Mux `PASS` (unless `--keep-temp`), moves `.audit.json`, `.audit.md`, and `.cast.yaml` into `audit/`, and honors `--dry-run` on `--clean-temp` / `--cleanup`.
- **`ENOSPC` Recovery (`src/pipeline.py`):** Standard disk mode reserves $n_{\text{unique}} \times \max(\text{recent}, P_{95}, 16.0\text{ MB}) \times 1.2$; Zero-Disk reserves $\max(256\text{ MB}, n_{\text{unique}} \times 0.5\text{ MB})$. On disk exhaustion, drains `EncodeWorker` / retries up to 2 times, or raises `FailedResumableError` so completed `up_*.mp4` chunks remain safe for queue retry.

## 5. PID GPU Lock, P-Core QoS & Anti-Sleep

1. **PID-Verified GPU Lock (`.gpu_active.lock`):** `acquire_gpu_lock()` checks live PID status via `psutil.pid_exists`, reclaiming orphaned locks while blocking concurrent GPU/Whisper collisions.
2. **Alder Lake P-Core Lock (`configure_alder_lake_pcore_qos`):** Disables Windows EcoQoS throttling (`PROCESS_POWER_THROTTLING_EXECUTION_SPEED`) and sets `ABOVE_NORMAL_PRIORITY_CLASS` so background/minimized terminals stay on P-Cores.
3. **Win32 Anti-Sleep (`AntiSleepContext`):** Sets `ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED` with guaranteed `atexit` restoration on `Ctrl+C`.
