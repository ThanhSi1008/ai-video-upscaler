# -*- coding: utf-8 -*-
"""
src/subtitles.py
Quản lý phụ đề đầu vào cho quy trình upscale:
- Tìm kiếm hoặc trích xuất phụ đề JA và EN từ file rời hoặc container video (.mkv, .mp4).
- Cơ chế Whisper Speech-to-Text dự phòng tạo phụ đề .ja.srt khi video gốc thiếu track tiếng Nhật.
"""

import os
import sys
import json
import subprocess
from typing import Optional, Tuple, List


def _build_whisper_initial_prompt(video_path: str) -> str:
    """
    Trích xuất danh sách tên nhân vật (Kanji/Katakana) và thuật ngữ đặc thù từ Glossary của bộ anime
    làm initial_prompt cho Whisper STT, giúp nhận diện chính xác tên riêng thay vì từ đồng âm.
    """
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
    """
    Trích xuất audio tiếng Nhật và chạy Whisper Speech-to-Text để tạo file phụ đề .ja.srt dự phòng.
    Ưu tiên tự động dùng `faster-whisper` (CTranslate2 + Silero VAD) nếu đã cài đặt và không bị mock,
    fallback mượt về `openai-whisper` để bảo toàn 100% tương thích ngược.
    Mặc định sử dụng model 'large-v3-turbo' tối ưu cho anime, hỗ trợ override qua biến môi trường WHISPER_MODEL.
    """
    try:
        import pysubs2
        import torch

        # Cho phép ghi đè mô hình qua biến môi trường WHISPER_MODEL
        env_model = os.environ.get("WHISPER_MODEL")
        if env_model:
            model_size = env_model.strip()

        print(f"\n🎙️ [WHISPER STT] Không tìm thấy track phụ đề tiếng Nhật. Đang trích xuất audio và nhận diện giọng nói (Whisper {model_size})...")
        audio_tmp = os.path.splitext(output_srt)[0] + ".temp.wav"

        # 1. Thăm dò stream âm thanh qua ffprobe: ưu tiên track tiếng Nhật và kiểm tra số kênh
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

        # 2. Áp dụng bộ lọc âm thanh tối ưu dải tần giọng nói / tách kênh thoại trung tâm FC
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
            # Fallback an toàn nếu -af thất bại: chạy lại lệnh cơ bản
            ext_cmd_basic = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                audio_tmp
            ]
            res = subprocess.run(ext_cmd_basic, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode != 0 or not os.path.exists(audio_tmp) or os.path.getsize(audio_tmp) == 0:
                return False

        from src.config import is_gpu_lock_held_by_other
        gpu_busy_pid = is_gpu_lock_held_by_other()
        if gpu_busy_pid is not None:
            print(f"ℹ️ [WHISPER STT] GPU đang bận bởi tiến trình Upscale (PID={gpu_busy_pid}) -> Chuyển Whisper sang CPU để bảo toàn VRAM.")
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        glossary_prompt = _build_whisper_initial_prompt(video_path)
        raw_segments = []
        fw_model = None
        model = None

        try:
            # Nếu môi trường test đang inject mock `whisper` vào sys.modules, dùng trực tiếp nhánh `whisper`
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

            # Lọc các segment có dấu hiệu là tạp âm hoặc ảo giác
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


def find_or_extract_subtitles(video_path: str, prefer_whisper_when_no_ja: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Tìm hoặc trích xuất phụ đề tiếng Nhật (ja) và tiếng Anh (en) cho video:
    1. Quét các file phụ đề sẵn có cùng thư mục video hoặc trong subtrans/samples.
    2. Nếu thiếu, dùng ffprobe/ffmpeg trích xuất track tương ứng từ container video.
    3. Nếu hoàn toàn không có track JA, chạy Whisper STT dự phòng.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    vid_dir = os.path.dirname(os.path.abspath(video_path))

    ja_sub = None
    en_sub = None

    # 1. Tìm các file phụ đề sẵn có cùng thư mục
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

    # 2. Kiểm tra container video qua ffprobe để trích xuất nếu thiếu file rời
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
            # mov_text, webvtt, text, ... -> chuẩn hóa sang SRT văn bản thuần
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
        # Loại bỏ các luồng phụ đề ảnh bitmap (PGS/VOBSUB trên Blu-ray) để không chặn trích xuất text hoặc Whisper fallback
        sub_info = [
            s for s in raw_sub_info
            if (s.get("codec_name") or "").lower() not in bitmap_sub_codecs
        ]

        if sub_info:
            os.makedirs(tmp_dir, exist_ok=True)

            # 2a. Trích xuất stream EN từ container video nếu chưa có file EN rời
            if not en_sub:
                chosen_en_stream = None
                # Phân loại và ưu tiên track hội thoại (Dialogue / Full Subtitles), tránh Signs & Songs
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
                            # Điểm ưu tiên: default=1 được 10 điểm, có chữ 'dialog' được 5 điểm, 'full' được 3 điểm
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
                    chosen_en_stream = sub_info[0]

                if chosen_en_stream is not None:
                    en_sub = _extract_text_stream(chosen_en_stream, "extracted_en")

            # 2b. Trích xuất stream JA từ container video nếu chưa có file JA rời
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

    # 3. Cơ chế dự phòng: Nếu không có track phụ đề tiếng Nhật dạng text
    if not ja_sub:
        whisper_ja = os.path.join(tmp_dir, f"{base_name}.whisper_ja.srt")
        if prefer_whisper_when_no_ja:
            # Ưu tiên chạy Whisper STT trước để giữ trọn vẹn ngữ cảnh đại từ xưng hô / kính ngữ tiếng Nhật
            # và sau đó có thể kết hợp với snap_whisper_cues_to_en
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

            # Chỉ fallback về ja_sub = en_sub nếu Whisper thất bại
            if not ja_sub and en_sub and os.path.exists(en_sub) and os.path.getsize(en_sub) > 100:
                ja_sub = en_sub
        else:
            # Mặc định: Ưu tiên 3a: Nếu có sẵn track tiếng Anh phòng thu (EN), ưu tiên dùng trực tiếp mốc thời gian chuẩn phòng thu
            # để đảm bảo đồng bộ từng khung hình (frame-perfect, 0ms sai lệch) và triệt tiêu câu rác Whisper
            if en_sub and os.path.exists(en_sub) and os.path.getsize(en_sub) > 100:
                ja_sub = en_sub
            else:
                # Ưu tiên 3b: Chạy Whisper STT nhận diện giọng nói tiếng Nhật nếu hoàn toàn không có track nào
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
