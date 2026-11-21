# AI Video Upscaler 4K & AI Subtitle Translator (Windows 11 & NVIDIA RTX)

Local **4K Anime Super-Resolution (2x)** and **Cloud Gemini Anime Subtitle Translation (Subtrans V2)** pipeline with a **Real-Time Web Studio (`web_gui.bat`)** designed for Windows 11.

## Target Workstation Specification
- **OS:** Windows 11 64-bit (`HAGS = 2`, `Ultimate Performance`, PCIe ASPM Off)
- **CPU:** Intel Core i5-12600K (10C/16T, `HighQoS` P-Core pinning, AVX2 OpenCV & SIMD)
- **GPU:** NVIDIA GeForce RTX 3060 12GB (`GA106`, VRAM P0 `7500 MHz`, Ampere Tensor Cores `HMMA.16816.F16`, TensorRT 10.16 FP16 & Triton JIT)
- **Hardware Encoder:** `hevc_nvenc` Main 10 (`p010le`, Preset `p6`, `CQ 18`, `maxrate 45M`, Multipass `fullres`, Lookahead `32`, Spatial AQ `strength 6` + Temporal AQ, `bf 4`, `b_ref_mode middle`, BT.709)
- **Anti-Sleep:** Win32 native `SetThreadExecutionState`

---

## 1. 4K Anime Video Upscaler (`upscale.py` / `upscale.bat` / `upscale_turbo.bat`)

- **SR Models & Pareto Routing:** `realesr-animevideov3` (`SRVGGNetCompact` Catmull-Rom collapsed 2x `_aa2x`, default fallback), Native 2x models (`AnimeJaNaiCompact2x`, `SPAN`) via Spandrel `0.4.2` mapped per show in `shows.yaml` (`sharp1-ultracompact` for Blu-ray, `ultracompact` for Web-DL), and explicit scene-adaptive `--model auto-pareto` (`compact` 16c + `ultracompact` 8c).
- **10-Bit End-to-End Zero-Disk Pipe (`--zero-disk`, Default):** Streams semi-planar `p010le` / `nv12` -> `@triton.jit` BT.709 YUV2RGB (`left` chroma) -> TensorRT 10.16 FP16 -> `@triton.jit` `p010le` (`left` chroma + IGN dither) -> in-process PyAV `hevc_nvenc` (`24s` super-chunks, zero intermediate PNGs). Fallback `--no-zero-disk` (`--disk`) uses rolling chunk cleanup.
- **True 3-Stage Overlap & Rolling Preflight:** `RollingDedupPlanner` (`max_workers=2`) launches GPU inference right after Chunk `0..1` while scanning remaining segments in parallel; decodes chunk `N+1` and flushes NVENC for chunk `N-1` concurrently with GPU inference on chunk `N`.
- **Perceptual Dedup V2 (`dis-auto`) & Cyclic Cel (`A->B->A`):** 1-pass `mpdecimate + DISOpticalFlow` with `8x8` dual-phase lip-flap guard, 3-DOF textured Lucas-Kanade pan/zoom solver, `raw_actions_rle` `< 1 ms` GOP re-partitioning, and bounded `<= 64`-frame Belady `OPT` cel reuse.
- **1-Pass Concat-Mux & Parallel Audit:** `mux_segments_directly` preserves original multi-channel audio, softsubs, fonts, and chapters with `-c copy -max_interleave_delta 0` (episode-locked `--nvenc-multipass` guarantees uniform HEVC VPS/SPS/PPS headers). Batch-0 GPU Output Sanity (`|ΔY| <= 6/255`, Pearson `r >= 0.98`) and parallel `-f streamhash` MD5 audit run before cleanup.
- **Resume & GPU Exclusivity:** Resumes safely from `up_*.mp4` checkpoints under `.temp_<basename>/` and protects VRAM with PID-validated `.gpu_active.lock`.

---

## 2. System Requirements & Setup

- Windows 10/11 64-bit, NVIDIA RTX 20/30/40/50 GPU (12GB+ recommended), 16GB+ RAM, FFmpeg/FFprobe with `hevc_nvenc`.

```powershell
# 1. Clone and create virtual environment
git clone https://github.com/ThanhSi1008/ai-video-upscaler.git
cd ai-video-upscaler
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install PyTorch (CUDA 12.4+) & project dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 3. (Recommended once as Admin) Enable HAGS=2 & unlock CUDA P0 clocks (7500 MHz)
.\scripts\unlock_rtx3060_admin.bat

# 4. Verify agent memory & run isolated offline tests
.\.venv\Scripts\python.exe scripts/agent_context.py --check
.\.venv\Scripts\python.exe scripts/verify_offline.py
```

---

## 3. CLI & Web Studio Usage

```powershell
# Quick 10s preview at 30% runtime
.\.venv\Scripts\python.exe upscale.py samples/sample_30s.mkv --preview 10

# Upscale 1 episode (Zero-Disk TensorRT default)
.\upscale.bat path/to/episode.mkv
.\upscale_turbo.bat path/to/episode.mkv
.\.venv\Scripts\python.exe upscale.py path/to/episode.mkv -o out/episode_4K.mkv

# Overnight batch queue with background GOP preflight & deadline
.\.venv\Scripts\python.exe upscale.py --queue ep1.mkv ep2.mkv -o out/ --deadline 07:00 --allow-partial

# Launch real-time Web Studio (http://127.0.0.1:8765)
.\web_gui.bat
.\.venv\Scripts\python.exe web_gui.py --port 8765
```

### Key `upscale.py` Options
- `-o`, `--output`: Output file or directory (default: `out/<name>_4K.mkv`).
- `-m`, `--model`: SR model ID from `models/models.yaml` or `auto-pareto` (auto-selected via `shows.yaml` if omitted).
- `--engine` (`cuda`|`cpu`), `--precision` (`fp16`|`fp32`), `--batch-size` (`8` default), `--nvenc-multipass` (`auto`|`fullres`|`qres`, default `auto`).
- `--zero-disk` (default `True`) / `--no-zero-disk` (`--disk`), `--no-tensorrt`.
- `--dedup-policy` (`default`|`dis-auto`|`sensitive-a`|`sensitive-b`|`motion`|`drift`|`dis`|`sakuga`|`dis-sakuga`|`off`), `--dis-thresh` (`0.30`), `--drift-delta` (`0.75`), `--no-dedup`.
- `--queue`, `--deadline HH:MM`, `--allow-partial`, `--force`, `--no-resume`, `--keep-temp`, `--clean-temp` / `--cleanup`.
- `--subs-after-upscale` / `--no-subs`, `--async-subs`, `--subs-only`, `--subs-pending`, `--subs-upgrade`, `--subs-quality` (`normal`|`fast`|`high`).
- `--prefer-whisper`, `--auto-align-en` / `--no-auto-align-en`, `--auto-offset` / `--no-auto-offset`, `--sub-offset-ms`, `--no-embed`, `--target-4k`.
- `--dry-run`, `--benchmark-mode`, `--preview [seconds]`, `--list-models`, `--no-anti-sleep`, `--no-ramdisk`.

---

## 4. Empirical Hardware Baseline (RTX 3060 12GB, `GA106`)

> Historical baseline logs are archived in [docs/archive/reports/engineering-log.md](docs/archive/reports/engineering-log.md).

| Specification | Legacy PyTorch (`batch=1`) | Legacy PyTorch PNG (`batch=6`) | Current (`TensorRT 10 + Zero-Disk`, `batch=8`) |
| :--- | :--- | :--- | :--- |
| **Clocks & QoS** | i5-12600K, RTX 3060 (P2) | i5-12600K, RTX 3060 (P2) | i5-12600K (`HighQoS` P-Core), RTX 3060 (`HAGS=2`, P0 `7500 MHz`) |
| **Engine** | PyTorch CUDA FP16 (`b=1`) | PyTorch CUDA FP16 (`b=6`) | **TensorRT 10.16 FP16** (`opt=3`, `_aa2x` collapsed 2x, 2-slot `_out_ring`) |
| **Throughput** | `1.37 fps` GPU-only | `4.8–5.2 fps` GPU-only (~`10–12+ fps` eff.) | **`10.91 fps`** (`animevideov3` `_aa2x`), **`18.06–24.44 fps`** (`sharp1-uc` + `dis-auto`), **`16.33 fps`** (`auto-pareto`) |
| **Frame I/O** | PIL `Image.save` (~45 ms/f) | OpenCV `cv2.imwrite` AVX2 (~14 ms/f) | **10-bit Zero-Disk Pipe** (`p010le/nv12 -> Triton YUV2RGB -> Triton p010le -> PyAV NVENC`) |
| **Encoder** | `hevc_nvenc` Main10 (`p6`, `CQ 19`) | `hevc_nvenc` Main10 (`p6`, `qres`, LA `24`) | `hevc_nvenc` Main10 (`p6`, **`CQ 18`**, `maxrate 45M`, **`auto` (`fullres`\|`qres`)**, LA **`32`**, Dual AQ) |
| **Mux & Audit** | 2-step mux (`merged_4k.mp4`) | 2-step mux + sequential audit | **1-Pass `mux_segments_directly`** + **1-Pass `streamhash` MD5** |

---

## 5. AI Anime Subtitle Translator — Subtrans V2 (`subtrans/`)

Cloud-only anime subtitle translation powered by Google GenAI Gemini SDK:

```powershell
.\subtrans.bat check                   # Check Gemini API connectivity & keys (consumes API call)
.\subtrans.bat models                  # List configured Gemini models & limits
.\subtrans.bat eval                    # Run offline deterministic eval & pronoun suites
.\subtrans.bat run path/to/ep.ja.srt --en path/to/ep.en.ass -o out/
```

- **3-Key Round-Robin & Quota Protection:** Rotates 3 keys across `gemini-3.5-flash-lite` (Draft) and `gemini-3.8-flash` (High / Cascade Repair), handling `429 PerMinute` failover and per-`(model, key_hash)` daily PT tracking.
- **Context Caching & CoT Schema:** Static `> 4,096`-token system prefix (Implicit & `--explicit-cache`), SQLite Contextual Cache `v2.2`, Stage A `.cast.yaml` + `HysteresisCastManager`, `ja_morph_tagger.py` (`fugashi`/UniDic), and think-before-translate `p1`/`p2` schema.
- **Linter V2, Cascade & Hybrid RAG:** Enforces CPS, CJK/Katakana bounds, Romaji vocative retention, `RULE_GLOSSARY_TERM`, and `RULE_PRONOUN_CONSISTENCY`; repairs hard cues via `gemini-3.8-flash` and retrieves few-shot examples from Hybrid BM25 + SQLite FTS5 2/3-gram RAG.
- **Faster-Whisper 5.1 FC & DTW Auto-Align:** Extracts 5.1 FC center speech, transcribes with `faster-whisper` (`large-v3-turbo`) + Silero VAD + Glossary prompt, and snaps timestamps to studio EN via Monotonic DTW (1:2 pause splitting + Piecewise 2-Segment Auto-Offset).

---

## 6. Repository Structure

```text
ai-video-upscaler/
├── GEMINI.md                   # Single Source of Truth instructions for Antigravity
├── upscale.py / .bat           # 4K Upscaler CLI & Windows launchers (upscale_turbo.bat)
├── subtrans.py / .bat          # Subtrans V2 CLI & launcher
├── web_gui.py / .bat           # Local Real-Time Web Studio (FastAPI + SSE 1Hz)
├── shows.yaml                  # Per-series SR model & dedup_policy rules
├── pytest.ini / ruff.toml      # Test markers & Ruff linter config
├── .antigravityignore          # Workspace indexing & token/contamination exclusions
├── .agents/                    # Lazy skills (skills/), guard hooks & optional MCP server
├── src/                        # Video core (TRT 10, Zero-Disk, Triton, Dedup, NVENC, Mux Audit)
├── subtrans/                   # Subtitle translation engine, eval harness & subtrans/tests/
├── web/                        # Web Studio backend (app, sse_hub, job_manager) & static UI
├── scripts/                    # Tooling (agent_context, export_review_bundle, verify_offline), benchmarks & scripts/oneoff/
├── glossaries/                 # Per-series anime glossaries & base.yaml pointer
├── models/                     # Video SR model registry (models/models.yaml)
├── state/                      # Runtime mutable state (state/pending_subs.json, gitignored)
├── tests/                      # Unit tests & tests/e2e/ acceptance suites
├── benchmarks/                 # GPU, Dedup/DIS & pronoun evaluation harnesses
└── docs/                       # Canonical architecture docs, docs/agents/ & docs/archive/
```
