"""Subtitle discovery, container extraction, and Whisper 5.1 FC STT fallback."""

import os
import sys
import json
import subprocess
from typing import Optional, Tuple, List


def _build_whisper_initial_prompt(video_path: str) -> str:
    """Build a Japanese character/term Whisper initial_prompt from the matched anime glossary."""
    try:
        from subtrans.glossary import find_matching_glossary, load_glossary_file
        from subtrans.ja_morph_tagger import strip_ja_furigana

        from pathlib import Path
        gloss_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "glossaries"
        matched = find_matching_glossary(video_path, gloss_dir)
        if not matched:
            return ""
        g = load_glossary_file(matched)
        tokens: List[str] = []
        for c in getattr(g, "characters", [])[:25]:
            nja = strip_ja_furigana((c.name_ja or "").strip())
            if nja and nja not in tokens:
                tokens.append(nja)
        for t in getattr(g, "terms", [])[:35]:
            tja = strip_ja_furigana((t.ja or "").strip())
            if len(tja) >= 2 and tja not in tokens:
                tokens.append(tja)
        if not tokens:
            return ""
        return "用語・登場人物: " + "、".join(tokens[:45])
    except Exception:
        return ""


def transcribe_japanese_audio(video_path: str, output_srt: str, model_size: str = "large-v3-turbo") -> bool:
    """Extract center-weighted Japanese dialogue and transcribe via faster-whisper/openai-whisper."""
    try:
        import pysubs2
        import torch

        env_model = os.environ.get("WHISPER_MODEL")
        if env_model:
            model_size = env_model.strip()

        print(f"\n🎙️ [WHISPER STT] Không tìm thấy track phụ đề tiếng Nhật. Đang trích xuất audio và nhận diện giọng nói (Whisper {model_size})...")
        audio_tmp = os.path.splitext(output_srt)[0] + ".temp.wav"

        chosen_stream_map = "0:a:0"
        channels = 2
        try:
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=index,channels:stream_tags=language,title",
                "-select_streams", "a",
                "-of", "json",
                video_path
            ]
            r = subprocess.run(probe_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0:
                audio_streams = json.loads(r.stdout).get("streams", [])
                chosen_s = None
                for s in audio_streams:
                    lang = s.get("tags", {}).get("language", "").lower()
                    title = s.get("tags", {}).get("title", "").lower()
                    if lang in ("jpn", "ja") or "japanese" in title or "jpn" in title:
                        chosen_s = s
                        break
                if not chosen_s and audio_streams:
                    chosen_s = audio_streams[0]
                if chosen_s:
                    chosen_stream_map = f"0:{chosen_s['index']}"
                    channels = int(chosen_s.get("channels", 2))
        except Exception as probe_err:
            print(f"⚠️ [WHISPER STT] Lỗi probe audio streams: {probe_err}")

        if channels >= 6:
            af_filter = "pan=mono|c0=0.7*FC+0.15*FL+0.15*FR,highpass=f=80,lowpass=f=8000,afftdn=nf=-25"
        else:
            af_filter = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25"

        ext_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-map", chosen_stream_map,
            "-vn", "-af", af_filter,
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            audio_tmp
        ]
        res = subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0 or not os.path.exists(audio_tmp) or os.path.getsize(audio_tmp) == 0:
            ext_cmd_basic = [
                "ffmpeg", "-y", "-i", video_path,
                "-map", chosen_stream_map,
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                audio_tmp
            ]
            res = subprocess.run(ext_cmd_basic, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode != 0 or not os.path.exists(audio_tmp) or os.path.getsize(audio_tmp) == 0:
                return False

        import src.config as _app_config

        force_cpu = os.environ.get("UPSCALER_FORCE_WHISPER_CPU", "").strip().lower() in ("1", "true", "yes")
        gpu_busy_pid = _app_config.is_gpu_lock_held_by_other()
        if force_cpu or gpu_busy_pid is not None:
            reason = (
                "UPSCALER_FORCE_WHISPER_CPU=1 (--async-subs)"
                if force_cpu
                else f"GPU đang bận bởi tiến trình Upscale (PID={gpu_busy_pid})"
            )
            print(f"ℹ️ [WHISPER STT] {reason} -> Chuyển Whisper sang CPU (int8) để bảo toàn VRAM.")
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        lock_acquired = False
        if device == "cuda" and not getattr(_app_config, "_GPU_LOCK_OWNED", False):
            try:
                _app_config.acquire_gpu_lock()
                lock_acquired = True
            except RuntimeError as lock_err:
                print(f"ℹ️ [WHISPER STT] {lock_err} -> Fallback sang CPU (int8).")
                device = "cpu"

        glossary_prompt = _build_whisper_initial_prompt(video_path)
        raw_segments = []
        fw_model = None
        model = None

        try:
            # Prefer openai-whisper mock branch whenever a test injects sys.modules["whisper"].
            whisper_mod_in_sys = sys.modules.get("whisper")
            use_faster_whisper = whisper_mod_in_sys is None
            if use_faster_whisper:
                try:
                    from faster_whisper import WhisperModel  # type: ignore

                    compute_type = "float16" if device == "cuda" else "int8"
                    fw_model = WhisperModel(model_size, device=device, compute_type=compute_type)
                    fw_kwargs = {
                        "language": "ja",
                        "task": "transcribe",
                        "condition_on_previous_text": False,
                        "no_speech_threshold": 0.6,
                        "log_prob_threshold": -1.0,
                        "compression_ratio_threshold": 2.0,
                        "vad_filter": True,
                        "vad_parameters": {"min_silence_duration_ms": 350},
                    }
                    if glossary_prompt:
                        fw_kwargs["initial_prompt"] = glossary_prompt
                    seg_iter, _ = fw_model.transcribe(audio_tmp, **fw_kwargs)
                    for s in seg_iter:
                        raw_segments.append({
                            "start": float(s.start),
                            "end": float(s.end),
                            "text": s.text,
                            "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0) or 0.0),
                            "compression_ratio": float(getattr(s, "compression_ratio", 0.0) or 0.0),
                            "avg_logprob": float(getattr(s, "avg_logprob", 0.0) or 0.0),
                        })
                except ImportError:
                    use_faster_whisper = False

            if not use_faster_whisper:
                import whisper

                model = whisper.load_model(model_size, device=device)
                ow_kwargs = {
                    "language": "ja",
                    "task": "transcribe",
                    "condition_on_previous_text": False,
                    "no_speech_threshold": 0.6,
                    "logprob_threshold": -1.0,
                    "compression_ratio_threshold": 2.0,
                    "fp16": (device == "cuda"),
                }
                if glossary_prompt:
                    ow_kwargs["initial_prompt"] = glossary_prompt
                result = model.transcribe(audio_tmp, **ow_kwargs)
                raw_segments = result.get("segments", [])
        finally:
            fw_model = None
            model = None
            if device == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if lock_acquired:
                _app_config.release_gpu_lock()
            try:
                if os.path.exists(audio_tmp):
                    os.remove(audio_tmp)
            except Exception:
                pass

        subs = pysubs2.SSAFile()
        for seg in raw_segments:
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            text = seg["text"].strip()
            if not text:
                continue

            # Filter segments with likely background noise or Whisper hallucination
            no_speech = seg.get("no_speech_prob", 0.0)
            comp_ratio = seg.get("compression_ratio", 0.0)
            avg_logprob = seg.get("avg_logprob", 0.0)
            dur_ms = end_ms - start_ms

            if no_speech > 0.65:
                continue
            if comp_ratio > 2.2:
                continue
            if avg_logprob < -1.2 and dur_ms < 800:
                continue

            subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text))

        if len(subs) == 0:
            return False

        subs.save(output_srt)
        if os.path.exists(output_srt):
            print(f"✅ [WHISPER STT] Đã tạo thành công {len(subs)} câu thoại tiếng Nhật dự phòng: {os.path.basename(output_srt)}")
            return True
        return False
    except Exception as e:
        try:
            print(f"⚠️ [WHISPER STT] Lỗi nhận diện giọng nói: {e}")
        except Exception:
            pass
        return False


def find_or_extract_subtitles(
    video_path: str,
    prefer_whisper_when_no_ja: bool = False,
    dry_run: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Locate sidecar JA/EN subtitles, extract text streams via FFmpeg, or fall back to Whisper STT."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    vid_dir = os.path.dirname(os.path.abspath(video_path))

    ja_sub = None
    en_sub = None

    # 1. Probe existing sidecar subtitle files alongside the video
    candidates_ja = [
        os.path.join(vid_dir, f"{base_name}.ja.ass"),
        os.path.join(vid_dir, f"{base_name}_ja.ass"),
        os.path.join(vid_dir, f"{base_name}_jpn.ass"),
        os.path.join(vid_dir, f"{base_name}.ass"),
        os.path.join(vid_dir, f"{base_name}.ja.srt"),
        os.path.join(vid_dir, f"{base_name}.ja[cc].srt"),
        os.path.join(vid_dir, f"{base_name}_ja.srt"),
        os.path.join(vid_dir, f"{base_name}_jpn.srt"),
        os.path.join(vid_dir, f"{base_name}.srt"),
        f"subtrans/samples/{base_name}_jpn.ass",
        f"subtrans/samples/{base_name}.ass",
        f"subtrans/samples/{base_name}.ja[cc].srt",
    ]
    for c in candidates_ja:
        if os.path.exists(c):
            ja_sub = c
            break

    candidates_en = [
        os.path.join(vid_dir, f"{base_name}.en.ass"),
        os.path.join(vid_dir, f"{base_name}_en.ass"),
        os.path.join(vid_dir, f"{base_name}_eng.ass"),
        os.path.join(vid_dir, f"{base_name}.en.srt"),
        os.path.join(vid_dir, f"{base_name}_en.srt"),
        os.path.join(vid_dir, f"{base_name}_eng.srt"),
        f"subtrans/samples/{base_name}_eng.ass",
        f"subtrans/samples/{base_name}.en.ass",
    ]
    for c in candidates_en:
        if os.path.exists(c):
            en_sub = c
            break

    # 2. Probe container subtitle tracks with ffprobe and extract missing streams
    safe_base = base_name.rstrip('. ')
    tmp_dir = os.path.join(vid_dir, f".temp_{safe_base}")
    bitmap_sub_codecs = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub", "pgssub", "vobsub"}

    def _extract_text_stream(stream_obj: dict, stem_prefix: str) -> Optional[str]:
        codec = (stream_obj.get("codec_name") or "").lower()
        if codec in bitmap_sub_codecs:
            return None
        s_idx = stream_obj["index"]
        if codec in ("ass", "ssa"):
            out_file = os.path.join(tmp_dir, f"{stem_prefix}.ass")
            codec_args = ["-c:s", "copy"]
        elif codec in ("subrip", "srt"):
            out_file = os.path.join(tmp_dir, f"{stem_prefix}.srt")
            codec_args = ["-c:s", "copy"]
        else:
            # Normalize mov_text/webvtt/text codecs to plain SRT
            out_file = os.path.join(tmp_dir, f"{stem_prefix}.srt")
            codec_args = ["-c:s", "srt"]
        ext_cmd = ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{s_idx}", *codec_args, out_file]
        sub_res = subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if sub_res.returncode == 0 and os.path.exists(out_file) and os.path.getsize(out_file) > 20:
            return out_file
        return None

    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_name:stream_tags=language,title:stream_disposition=default",
            "-select_streams", "s",
            "-of", "json",
            video_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        raw_sub_info = json.loads(r.stdout).get("streams", [])
        # Skip bitmap tracks (PGS/VOBSUB) so they do not block text extraction or Whisper fallback
        sub_info = [
            s for s in raw_sub_info
            if (s.get("codec_name") or "").lower() not in bitmap_sub_codecs
        ]

        if sub_info and not dry_run:
            os.makedirs(tmp_dir, exist_ok=True)

            # 2a. Extract EN stream if no sidecar EN file was found
            if not en_sub:
                chosen_en_stream = None
                # Rank full dialogue tracks ahead of Signs & Songs tracks
                dialogue_candidates = []
                other_en_candidates = []
                for s in sub_info:
                    lang = s.get("tags", {}).get("language", "").lower()
                    title = s.get("tags", {}).get("title", "").lower()
                    is_def = s.get("disposition", {}).get("default", 0) == 1
                    is_en = lang in ("eng", "en") or "english" in title or "eng" in title
                    if is_en:
                        is_signs = any(k in title for k in ("s&s", "signs", "songs", "[signs]", "s/s"))
                        if not is_signs:
                            score = (10 if is_def else 0) + (5 if "dialog" in title else 0) + (3 if "full" in title else 0)
                            dialogue_candidates.append((score, s))
                        else:
                            other_en_candidates.append(s)

                if dialogue_candidates:
                    dialogue_candidates.sort(key=lambda x: x[0], reverse=True)
                    chosen_en_stream = dialogue_candidates[0][1]
                elif other_en_candidates:
                    chosen_en_stream = other_en_candidates[0]
                elif len(sub_info) > 0:
                    first_lang = str(sub_info[0].get("tags", {}).get("language", "") or "").strip().lower()
                    if not first_lang or first_lang == "und" or first_lang.startswith("en"):
                        chosen_en_stream = sub_info[0]

                if chosen_en_stream is not None:
                    en_sub = _extract_text_stream(chosen_en_stream, "extracted_en")

            # 2b. Extract JA stream if no sidecar JA file was found
            if not ja_sub:
                chosen_ja_stream = None
                for s in sub_info:
                    lang = s.get("tags", {}).get("language", "").lower()
                    title = s.get("tags", {}).get("title", "").lower()
                    if lang in ("jpn", "ja") or "japanese" in title or "jpn" in title:
                        chosen_ja_stream = s
                        break
                if chosen_ja_stream is not None:
                    ja_sub = _extract_text_stream(chosen_ja_stream, "extracted_ja")
    except Exception:
        pass

    if not prefer_whisper_when_no_ja:
        env_val = (
            os.environ.get("SUBTRANS_WHISPER_WITH_EN", "").strip().lower()
            or os.environ.get("SUBTRANS_AUTO_WHISPER", "").strip().lower()
        )
        if env_val in ("1", "true", "yes"):
            prefer_whisper_when_no_ja = True

    if dry_run:
        if not ja_sub and not prefer_whisper_when_no_ja and en_sub and os.path.exists(en_sub) and os.path.getsize(en_sub) > 100:
            ja_sub = en_sub
        return ja_sub, en_sub

    # 3. Fall back when no text JA subtitle track is available
    if not ja_sub:
        whisper_ja = os.path.join(tmp_dir, f"{base_name}.whisper_ja.srt")
        if prefer_whisper_when_no_ja:
            # Run Whisper STT first to capture JA honorifics/pronouns for downstream alignment
            if os.path.exists(whisper_ja) and os.path.getsize(whisper_ja) > 100:
                ja_sub = whisper_ja
            else:
                try:
                    os.makedirs(tmp_dir, exist_ok=True)
                    ok = transcribe_japanese_audio(video_path, whisper_ja)
                    if ok and os.path.exists(whisper_ja) and os.path.getsize(whisper_ja) > 100:
                        ja_sub = whisper_ja
                except Exception as e_whisper:
                    try:
                        print(f"⚠️ [WHISPER FALLBACK] Không thể tạo phụ đề JA từ giọng nói: {e_whisper}")
                    except Exception:
                        pass

            # Only fall back to EN timing if Whisper fails
            if not ja_sub and en_sub and os.path.exists(en_sub) and os.path.getsize(en_sub) > 100:
                ja_sub = en_sub
        else:
            # Default: prefer frame-accurate studio EN track over Whisper when JA text is missing
            if en_sub and os.path.exists(en_sub) and os.path.getsize(en_sub) > 100:
                ja_sub = en_sub
            else:
                # Run Whisper STT when neither JA nor EN subtitle track exists
                if os.path.exists(whisper_ja) and os.path.getsize(whisper_ja) > 100:
                    ja_sub = whisper_ja
                else:
                    try:
                        os.makedirs(tmp_dir, exist_ok=True)
                        ok = transcribe_japanese_audio(video_path, whisper_ja)
                        if ok and os.path.exists(whisper_ja) and os.path.getsize(whisper_ja) > 100:
                            ja_sub = whisper_ja
                    except Exception as e_whisper:
                        try:
                            print(f"⚠️ [WHISPER FALLBACK] Không thể tạo phụ đề JA từ giọng nói: {e_whisper}")
                        except Exception:
                            pass

    return ja_sub, en_sub
