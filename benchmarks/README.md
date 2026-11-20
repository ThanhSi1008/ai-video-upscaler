# Benchmarks & Evaluations

Performance benchmarks, pronoun stability evaluators, and visual fidelity / blind A-B tools tuned for Windows 11 & NVIDIA RTX.

## Directory Layout

### 1. `benchmarks/upscale/`
Video Super-Resolution & deduplication harnesses:
- `benchmark_models_comparison.py`: Compare throughput and VRAM across CUDA/TRT SR models.
- `profile_tail_latency.py`: Measure P95/P99 per-frame tail latency on NVIDIA RTX.
- `audit_local_metrics.py` & `audit_small_block_dedup.py`: Measure deduplication compression ratio and motion sensitivity.
- `chaos_test_resume.py`: Test chunk checkpoint recovery under interruption.
- `ab_test/`: Blind A/B visual evaluation assets (`KEY_ANSWER.json` excluded in `.antigravityignore`).

### 2. `benchmarks/subtrans/`
Subtitle translation pronoun & regex evaluators:
- `eval_golden_pronouns.py`: Evaluate pronoun consistency on the *Mushoku Tensei* golden set.
- `eval_golden_pronouns_multigenre.py`: Cross-genre pronoun evaluation (Fantasy, Action, Drama, Romance).
- `validate_pronouns_regex.py`: Validate pronoun & honorific detection regexes.

### 3. `benchmarks/research/`
Algorithm calibration & micro-benchmark scripts:
- `calibrate_b1_b2.py`: Motion-detection threshold calibration.
- `benchmark_dis_synthetic_b1_b2.py`: Synthetic DIS optical-flow policy benchmarks.
- `measure_a1_a2_windows.py`: Windows CUDA FP16 throughput & precision checks.
- `measure_a2_animejanai.py`: Numerical parity verification against Spandrel CPU reference.
- `make_blind_ab.py`: Generate blind A/B comparison clips across SR models.
