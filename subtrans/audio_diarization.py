"""CPU-only 5.1 FC timbre and F0 pitch turn detector for unlabeled SRT dialogue."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from subtrans.models import Cue

logger = logging.getLogger("subtrans.audio_diarization")

SAMPLE_RATE_HZ = 8000
WIN_SAMPLES = 256   # 32 ms @ 8 kHz
HOP_SAMPLES = 128   # 16 ms @ 8 kHz
MIN_CUE_DURATION_MS = 320
MAX_TURN_GAP_MS = 3000

# 12 dải tần quasi-Mel bao phủ vùng cộng hưởng thanh quản & formant F1-F3 (90 Hz -> 3700 Hz)
_BAND_EDGES_HZ = (90, 160, 240, 340, 460, 600, 780, 1000, 1300, 1700, 2200, 2850, 3700)


@lru_cache(maxsize=8)
def _get_dsp_constants(sample_rate: int) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    """Caches Hann window and 12 quasi-Mel frequency band masks per sample rate."""
    hann = np.hanning(WIN_SAMPLES).astype(np.float32)
    hann.setflags(write=False)
    freqs = np.fft.rfftfreq(WIN_SAMPLES, d=1.0 / float(sample_rate))
    masks = []
    for b_idx in range(len(_BAND_EDGES_HZ) - 1):
        f_lo, f_hi = _BAND_EDGES_HZ[b_idx], _BAND_EDGES_HZ[b_idx + 1]
        m = (freqs >= f_lo) & (freqs < f_hi)
        m.setflags(write=False)
        masks.append(m)
    return hann, tuple(masks)


@dataclass(frozen=True)
class VoiceSignature:
    """Đặc trưng âm học chuẩn hóa năng lượng của một câu thoại đơn lẻ."""
    timbre: np.ndarray          # Vector 12 chiều đã trừ trung bình (CMS) & chuẩn hóa L2
    f0_hz: Optional[float]      # Cao độ cơ bản trung vị (75 Hz - 400 Hz) nếu đủ khung hữu thanh
    voiced_frames: int          # Số khung hữu thanh tuần hoàn
    active_frames: int          # Tổng số khung vượt ngưỡng năng lượng thoại


def compute_voice_signature(
    pcm_f32: np.ndarray,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> Optional[VoiceSignature]:
    """
    Trích xuất đặc trưng âm sắc (12-band CMS timbre) và cao độ cơ bản F0 từ mảng PCM float32 [-1, 1].
    Trả về None nếu đoạn âm thanh quá ngắn, toàn khoảng lặng hoặc nhiễu không đủ độ tin cậy.
    """
    if pcm_f32 is None or pcm_f32.ndim != 1:
        return None
    min_samples = int(sample_rate * (MIN_CUE_DURATION_MS / 1000.0))
    if pcm_f32.size < max(WIN_SAMPLES + HOP_SAMPLES, min_samples):
        return None

    # Chia khung trượt 32ms, bước nhảy 16ms bằng sliding_window_view không sao chép bộ nhớ thừa
    frames_view = sliding_window_view(pcm_f32, WIN_SAMPLES)[::HOP_SAMPLES]
    n_frames = frames_view.shape[0]
    if n_frames < 5:
        return None

    # Khử DC offset từng khung (tạo mảng float32 1 lần duy nhất) và lọc khung thoại thực sự
    frames = frames_view.astype(np.float32, copy=False)
    frames = frames - np.mean(frames, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    p85 = float(np.percentile(rms, 85))
    energy_floor = max(0.010, 0.30 * p85)
    active_mask = rms >= energy_floor
    active_count = int(np.count_nonzero(active_mask))
    if active_count < 5:
        return None

    active_frames = frames[active_mask]
    hann, band_masks = _get_dsp_constants(sample_rate)

    # 1. Phổ Formant / Timbre (Pre-emphasis 0.95 + RFFT + 12 dải tần log-energy + CMS)
    pre = np.empty_like(active_frames)
    pre[:, 0] = active_frames[:, 0]
    pre[:, 1:] = active_frames[:, 1:] - 0.95 * active_frames[:, :-1]
    spec_power = np.abs(np.fft.rfft(pre * hann[None, :], n=WIN_SAMPLES)) ** 2

    band_energies = np.zeros(len(band_masks), dtype=np.float32)
    for b_idx, b_mask in enumerate(band_masks):
        if np.any(b_mask):
            band_energies[b_idx] = float(np.mean(np.log(np.mean(spec_power[:, b_mask], axis=1) + 1e-10)))

    # Cepstral Mean Subtraction (CMS) -> bất biến với âm lượng to/nhỏ của câu nói
    timbre = band_energies - float(np.mean(band_energies))
    norm = float(np.linalg.norm(timbre))
    if norm < 1e-6:
        return None
    timbre = timbre / norm

    # 2. Ước lượng cao độ cơ bản F0 qua hàm tự tương quan chuẩn hóa (75 Hz - 400 Hz)
    lag_min = max(2, int(round(sample_rate / 400.0)))
    lag_max = min(WIN_SAMPLES - 2, int(round(sample_rate / 75.0)))
    win_raw = active_frames * hann[None, :]
    psd = np.abs(np.fft.rfft(win_raw, n=512)) ** 2
    acf = np.fft.irfft(psd, n=512)[:, : lag_max + 1]
    acf0 = np.maximum(acf[:, 0:1], 1e-8)
    norm_acf = acf / acf0

    search_region = norm_acf[:, lag_min : lag_max + 1]
    peak_rel_idx = np.argmax(search_region, axis=1)
    peak_vals = search_region[np.arange(active_count), peak_rel_idx]
    voiced_mask = peak_vals >= 0.32
    voiced_count = int(np.count_nonzero(voiced_mask))

    f0_median: Optional[float] = None
    if voiced_count >= 3:
        voiced_lags = (peak_rel_idx[voiced_mask] + lag_min).astype(np.float32)
        f0_vals = float(sample_rate) / voiced_lags
        f0_median = float(np.median(f0_vals))

    return VoiceSignature(
        timbre=timbre,
        f0_hz=f0_median,
        voiced_frames=voiced_count,
        active_frames=active_count,
    )


def classify_turn_transition(
    prev_sig: Optional[VoiceSignature],
    curr_sig: Optional[VoiceSignature],
    gap_ms: int,
) -> Optional[str]:
    """
    Phân loại chuyển lượt thoại giữa 2 câu liên tiếp trong cùng cảnh (`0 <= gap_ms <= 3000`):
    - 'speaker_changed': Độ lệch phổ âm sắc hoặc bước nhảy cao độ F0 vượt ngưỡng rõ rệt.
    - 'same_speaker': Âm sắc và cao độ F0 đồng nhất cao qua khoảng nghỉ ngắn (`<= 1500ms`).
    - None: Không đủ độ tin cậy âm học (BGM lớn, thoại chồng, hoặc khoảng nghỉ chuyển cảnh).
    """
    if prev_sig is None or curr_sig is None:
        return None
    if gap_ms < -150 or gap_ms > MAX_TURN_GAP_MS:
        return None

    timbre_dist = 1.0 - float(np.clip(np.dot(prev_sig.timbre, curr_sig.timbre), -1.0, 1.0))
    octave_diff: Optional[float] = None
    if prev_sig.f0_hz and curr_sig.f0_hz and prev_sig.f0_hz > 0 and curr_sig.f0_hz > 0:
        octave_diff = abs(math.log2(curr_sig.f0_hz / prev_sig.f0_hz))

    # Ngưỡng báo đổi giọng bảo thủ (chống báo giả khi cùng 1 người lên giọng cuối câu)
    if octave_diff is not None and octave_diff >= 0.42:
        return "speaker_changed"
    if timbre_dist >= 0.34:
        return "speaker_changed"
    if octave_diff is not None and octave_diff >= 0.24 and timbre_dist >= 0.20:
        return "speaker_changed"

    # Ngưỡng xác nhận giữ nguyên người nói (câu nối tiếp sát nhau, cùng F0 và cùng phổ âm sắc)
    if gap_ms <= 1500 and timbre_dist <= 0.08:
        if octave_diff is None or octave_diff <= 0.14:
            return "same_speaker"

    return None


def annotate_cues_from_pcm(
    cues: List[Cue],
    pcm_f32: np.ndarray,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> Dict[str, int]:
    """
    Gắn trường `cue.audio_turn_hint` cho danh sách `Cue` từ mảng âm thanh PCM mono đã giải mã.
    Chỉ gắn nhãn cho các cue chưa có `cue.speaker` tường minh (SRT trần).
    """
    stats = {"analyzed": 0, "speaker_changed": 0, "same_speaker": 0}
    if not cues or pcm_f32 is None or pcm_f32.size == 0:
        return stats

    active_cues = [
        c for c in cues
        if not c.is_skipped and not getattr(c, "is_dropped_sound", False)
    ]
    if len(active_cues) < 2:
        return stats

    total_samples = pcm_f32.size
    sigs: List[Optional[VoiceSignature]] = []
    for c in active_cues:
        # Bỏ qua câu thoại song song gộp sẵn nhiều người ('- A\n- B')
        raw_ja = (c.ja or "").strip()
        if "\n-" in raw_ja or raw_ja.startswith("- ") or raw_ja.startswith("-\u3000"):
            sigs.append(None)
            continue
        s_idx = max(0, min(total_samples, int(c.start * sample_rate / 1000)))
        e_idx = max(s_idx, min(total_samples, int(c.end * sample_rate / 1000)))
        sig = compute_voice_signature(pcm_f32[s_idx:e_idx], sample_rate=sample_rate)
        if sig is not None:
            stats["analyzed"] += 1
        sigs.append(sig)

    for idx in range(1, len(active_cues)):
        curr_c = active_cues[idx]
        prev_c = active_cues[idx - 1]
        if curr_c.speaker and curr_c.speaker.strip().lower() not in ("", "unknown"):
            continue
        gap_ms = curr_c.start - prev_c.end
        hint = classify_turn_transition(sigs[idx - 1], sigs[idx], gap_ms=gap_ms)
        if hint:
            curr_c.audio_turn_hint = hint
            stats[hint] += 1

    return stats


def _extract_center_dialogue_pcm_in_memory(
    media_path: Path,
    sample_rate: int = SAMPLE_RATE_HZ,
    timeout_sec: int = 45,
) -> Optional[np.ndarray]:
    """
    Dùng FFmpeg giải mã trực tiếp kênh thoại trung tâm 5.1 FC (hoặc stereo) sang PCM 16-bit 8kHz
    ngay trong bộ nhớ RAM (~23 MB cho tập anime 24 phút, 0 file tạm trên SSD, 0 VRAM GPU).
    """
    if not media_path or not media_path.exists():
        return None

    chosen_stream_map = "0:a:0"
    channels = 2
    effective_timeout = int(timeout_sec)
    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,channels:stream_tags=language,title:format=duration",
            "-select_streams", "a",
            "-of", "json",
            str(media_path),
        ]
        pr = subprocess.run(probe_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        if pr.returncode == 0 and pr.stdout:
            probe_data = json.loads(pr.stdout)
            streams = probe_data.get("streams", [])
            chosen = None
            for s in streams:
                lang = (s.get("tags", {}).get("language") or "").lower()
                title = (s.get("tags", {}).get("title") or "").lower()
                if lang in ("jpn", "ja") or "japanese" in title or "jpn" in title:
                    chosen = s
                    break
            if not chosen and streams:
                chosen = streams[0]
            if chosen is not None:
                chosen_stream_map = f"0:{chosen['index']}"
                channels = int(chosen.get("channels", 2) or 2)
            try:
                dur_sec = float((probe_data.get("format") or {}).get("duration") or 0.0)
                if dur_sec > 0:
                    effective_timeout = min(600, max(effective_timeout, int(math.ceil(dur_sec / 20.0)) + 30))
            except (TypeError, ValueError):
                pass
    except Exception as e:
        logger.debug(f"Audio diarization ffprobe skipped: {e}")

    if channels >= 6:
        af_filter = "pan=mono|c0=0.75*FC+0.125*FL+0.125*FR,highpass=f=80,lowpass=f=3800"
    else:
        af_filter = "highpass=f=80,lowpass=f=3800"

    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", str(media_path),
        "-map", chosen_stream_map,
        "-vn", "-sn", "-dn",
        "-af", af_filter,
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=effective_timeout)
        if res.returncode != 0 or not res.stdout or len(res.stdout) < sample_rate * 2:
            return None
        usable_bytes = (len(res.stdout) // 2) * 2
        pcm_i16 = np.frombuffer(memoryview(res.stdout)[:usable_bytes], dtype=np.int16)
        return pcm_i16.astype(np.float32) / 32768.0
    except Exception as e:
        logger.debug(f"Audio diarization ffmpeg pipe skipped: {e}")
        return None


def maybe_annotate_unlabeled_srt_speaker_turns(
    cues: List[Cue],
    video_path: Optional[Path],
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Chỉ kích hoạt trích xuất âm sắc 5.1 FC khi:
    1. Có `video_path` thực tế trên đĩa và không ở chế độ `dry_run`.
    2. Hơn 60% số câu thoại đang khuyết `cue.speaker` (phụ đề `.srt` trần không có Actor/CC).
    """
    empty_stats = {"analyzed": 0, "speaker_changed": 0, "same_speaker": 0}
    if dry_run or not video_path or not Path(video_path).exists() or not cues:
        return empty_stats

    active_cues = [
        c for c in cues
        if not c.is_skipped and not getattr(c, "is_dropped_sound", False)
    ]
    if len(active_cues) < 2:
        return empty_stats

    labeled_count = sum(
        1 for c in active_cues
        if c.speaker and c.speaker.strip().lower() not in ("", "unknown")
    )
    if labeled_count / float(len(active_cues)) >= 0.40:
        # Đã có nhãn nhân vật từ .ass Actor hoặc Closed Captions -> không tốn CPU phân tích audio
        return empty_stats

    pcm = _extract_center_dialogue_pcm_in_memory(Path(video_path))
    if pcm is None:
        return empty_stats

    return annotate_cues_from_pcm(cues, pcm, sample_rate=SAMPLE_RATE_HZ)
