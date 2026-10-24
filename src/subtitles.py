# -*- coding: utf-8 -*-
"""
src/subtitles.py
Quản lý phụ đề đầu vào cho quy trình upscale:
- Tìm kiếm hoặc trích xuất phụ đề JA và EN từ file rời hoặc container video (.mkv, .mp4).
- Cơ chế Whisper Speech-to-Text dự phòng tạo phụ đề .ja.srt khi video gốc thiếu track tiếng Nhật.
"""

import os
import json
import subprocess
from typing import Optional, Tuple


def transcribe_japanese_audio(video_path: str, output_srt: str, model_size: str = "base") -> bool:
    """
    Trích xuất audio tiếng Nhật và chạy Whisper Speech-to-Text để tạo file phụ đề .ja.srt dự phòng.
    Giúp tận dụng triệt để mọi nguồn video không có sẵn track phụ đề tiếng Nhật.
    """
    try:
        import whisper
        import pysubs2
        import torch

        print(f"\n🎙️ [WHISPER STT] Không tìm thấy track phụ đề tiếng Nhật. Đang trích xuất audio và nhận diện giọng nói (Whisper {model_size})...")
        audio_tmp = os.path.splitext(output_srt)[0] + ".temp.wav"
        ext_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            audio_tmp
        ]
        res = subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0 or not os.path.exists(audio_tmp):
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(audio_tmp, language="ja", task="transcribe")

        subs = pysubs2.SSAFile()
        for seg in result.get("segments", []):
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            text = seg["text"].strip()
            if text:
                subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text))

        subs.save(output_srt)
        try:
            if os.path.exists(audio_tmp):
                os.remove(audio_tmp)
        except Exception:
            pass

        if os.path.exists(output_srt) and len(subs) > 0:
            print(f"✅ [WHISPER STT] Đã tạo thành công {len(subs)} câu thoại tiếng Nhật dự phòng: {os.path.basename(output_srt)}")
            return True
        return False
    except Exception as e:
        print(f"⚠️ [WHISPER STT] Lỗi nhận diện giọng nói: {e}")
        return False


def find_or_extract_subtitles(video_path: str) -> Tuple[Optional[str], Optional[str]]:
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
    tmp_dir = os.path.join(vid_dir, f".temp_{base_name}")
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index:stream_tags=language,title",
            "-select_streams", "s",
            "-of", "json",
            video_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        sub_info = json.loads(r.stdout).get("streams", [])

        if sub_info:
            os.makedirs(tmp_dir, exist_ok=True)

            # 2a. Trích xuất stream EN từ container video nếu chưa có file EN rời
            if not en_sub:
                en_idx = None
                for s in sub_info:
                    lang = s.get("tags", {}).get("language", "").lower()
                    title = s.get("tags", {}).get("title", "").lower()
                    if lang in ("eng", "en") or "english" in title or "eng" in title:
                        en_idx = s["index"]
                        break
                if en_idx is None and len(sub_info) > 0:
                    en_idx = sub_info[0]["index"]

                if en_idx is not None:
                    extracted_en = os.path.join(tmp_dir, "extracted_en.ass")
                    ext_cmd = ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{en_idx}", "-c:s", "copy", extracted_en]
                    sub_res = subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if sub_res.returncode == 0 and os.path.exists(extracted_en) and os.path.getsize(extracted_en) > 100:
                        en_sub = extracted_en

            # 2b. Trích xuất stream JA từ container video nếu chưa có file JA rời
            if not ja_sub:
                ja_idx = None
                for s in sub_info:
                    lang = s.get("tags", {}).get("language", "").lower()
                    title = s.get("tags", {}).get("title", "").lower()
                    if lang in ("jpn", "ja") or "japanese" in title or "jpn" in title:
                        ja_idx = s["index"]
                        break
                if ja_idx is not None:
                    extracted_ja = os.path.join(tmp_dir, "extracted_ja.ass")
                    ext_cmd = ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{ja_idx}", "-c:s", "copy", extracted_ja]
                    sub_res = subprocess.run(ext_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if sub_res.returncode == 0 and os.path.exists(extracted_ja) and os.path.getsize(extracted_ja) > 100:
                        ja_sub = extracted_ja
    except Exception:
        pass

    # 3. Cơ chế dự phòng: Nếu không có track phụ đề tiếng Nhật dạng text
    if not ja_sub:
        # Ưu tiên 3a: Nếu có sẵn track tiếng Anh phòng thu (EN), ưu tiên dùng trực tiếp mốc thời gian chuẩn phòng thu
        # để đảm bảo đồng bộ từng khung hình (frame-perfect, 0ms sai lệch) và triệt tiêu câu rác Whisper
        if en_sub and os.path.exists(en_sub) and os.path.getsize(en_sub) > 100:
            ja_sub = en_sub
        else:
            # Ưu tiên 3b: Chạy Whisper STT nhận diện giọng nói tiếng Nhật nếu hoàn toàn không có track nào
            whisper_ja = os.path.join(tmp_dir, f"{base_name}.whisper_ja.srt")
            if os.path.exists(whisper_ja) and os.path.getsize(whisper_ja) > 100:
                ja_sub = whisper_ja
            else:
                try:
                    os.makedirs(tmp_dir, exist_ok=True)
                    ok = transcribe_japanese_audio(video_path, whisper_ja)
                    if ok and os.path.exists(whisper_ja):
                        ja_sub = whisper_ja
                except Exception as e_whisper:
                    print(f"⚠️ [WHISPER FALLBACK] Không thể tạo phụ đề JA từ giọng nói: {e_whisper}")

    return ja_sub, en_sub
