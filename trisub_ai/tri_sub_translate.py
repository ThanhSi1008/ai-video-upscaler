#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TriSubTranslate: Công cụ dịch phụ đề Anime chuẩn chỉnh đa ngữ (Anh - Nhật - Trung -> Tiếng Việt)
Hỗ trợ định dạng: .srt, .vtt, .ass
Hỗ trợ API: Google Gemini, DeepSeek, OpenAI, hoặc Chế độ Xuất Prompt thủ công.
Tự động nhúng phụ đề vào file video MKV qua ffmpeg nếu có yêu cầu.
"""

import sys
import os
import re
import json
import argparse
import subprocess
import tempfile
import urllib.request
import urllib.error

try:
    import requests
except ImportError:
    requests = None


def clean_json_text(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def http_post_json(url, payload, headers=None, timeout=60):
    headers = headers or {}
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    if requests is not None:
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if res.status_code == 200:
                return res.json()
        except Exception:
            pass

    # Fallback to standard library urllib.request (100% độc lập, không cần thư viện ngoài)
    data_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8')
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8', errors='ignore')
        raise RuntimeError(f"HTTP {e.code} Error: {err_body}")

    sec = float(s) + float('0.' + str(frac))
    return int((int(h) * 3600 + int(m) * 60 + sec) * 1000)


def ms_to_srt_time(ms):
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_subtitle(file_path):
    """Đọc và trích xuất danh sách câu thoại kèm timestamp (ms) từ file srt, vtt hoặc ass"""
    if not file_path or not os.path.isfile(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    ext = file_path.lower().split('.')[-1]
    entries = []

    if ext in ['srt', 'vtt']:
        pattern = re.compile(
            r'(?:(\d+)\s*\n)?'
            r'(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{3})'
            r'(?:[^\n]*\n)'
            r'([\s\S]*?)(?=\n\s*(?:\d+\s*\n)?\d{1,2}:\d{2}:\d{2}[,\.]\d{3}\s*-->|\Z)'
        )
        for m in pattern.finditer(content):
            _, h1, m1, s1, ms1, h2, m2, s2, ms2, text = m.groups()
            start = to_ms(h1, m1, s1, ms1)
            end = to_ms(h2, m2, s2, ms2)
            clean = re.sub(r'<[^>]+>', '', text)
            clean = re.sub(r'\{[^\}]+\}', '', clean).strip()
            if clean:
                entries.append({
                    'start': start,
                    'end': end,
                    'time_str': f"{int(h1):02d}:{m1}:{s1},{ms1} --> {int(h2):02d}:{m2}:{s2},{ms2}",
                    'text': clean
                })

    elif ext == 'ass':
        for line in content.splitlines():
            if line.startswith('Dialogue:'):
                parts = line.split(',', 9)
                if len(parts) == 10:
                    start_str, end_str, text = parts[1].strip(), parts[2].strip(), parts[9].strip()
                    m1 = re.match(r'(\d+):(\d{2}):(\d{2})\.(\d{2})', start_str)
                    m2 = re.match(r'(\d+):(\d{2}):(\d{2})\.(\d{2})', end_str)
                    if m1 and m2:
                        start = to_ms(m1.group(1), m1.group(2), m1.group(3), m1.group(4) + '0')
                        end = to_ms(m2.group(1), m2.group(2), m2.group(3), m2.group(4) + '0')
                        clean = re.sub(r'\{[^\}]+\}', '', text)
                        clean = clean.replace('\\N', '\n').replace('\\n', '\n').strip()
                        if clean:
                            entries.append({
                                'start': start,
                                'end': end,
                                'time_str': f"{ms_to_srt_time(start)} --> {ms_to_srt_time(end)}",
                                'text': clean
                            })

    return entries


def get_sub_tracks(video_path):
    """Lấy danh sách các track phụ đề có trong file video kèm thông tin ngôn ngữ và codec"""
    if not video_path or not os.path.isfile(video_path):
        return []
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "s",
            "-show_entries", "stream=index,codec_name:stream_tags=language,title",
            "-of", "json", video_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            data = json.loads(res.stdout)
            streams = data.get("streams", [])
            tracks = []
            for s in streams:
                tags = s.get("tags") or {}
                tracks.append({
                    "stream_index": s.get("index"),
                    "codec": s.get("codec_name", ""),
                    "lang": tags.get("language", "und").lower(),
                    "title": tags.get("title", "")
                })
            return tracks
    except Exception:
        pass
    return []


def extract_sub_from_video(video_path, lang_code="eng", output_srt=None):
    """Trích xuất tự động track phụ đề từ video MKV/MP4 ra file srt chuẩn UTF-8"""
    if not video_path or not os.path.isfile(video_path):
        return None
    if not output_srt:
        base, _ = os.path.splitext(video_path)
        output_srt = f"{base}.extracted_{lang_code}.srt"

    # 1. Tìm chính xác track phù hợp qua ffprobe
    tracks = get_sub_tracks(video_path)
    target_stream = None

    for t in tracks:
        t_lang = t["lang"].lower()
        t_title = t["title"].lower()
        if t_lang in [lang_code.lower(), "en", "eng"] or "english" in t_title:
            target_stream = t["stream_index"]
            break

    # 2. Thực hiện trích xuất bằng ffmpeg
    if target_stream is not None:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-map", f"0:{target_stream}",
            "-c:s", "srt",
            output_srt
        ]
    else:
        # Fallback thử theo metadata language hoặc lấy track sub đầu tiên
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-map", f"0:m:language:{lang_code}?",
            "-map", "0:s:0?",
            "-c:s", "srt",
            output_srt
        ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.isfile(output_srt) or os.path.getsize(output_srt) == 0:
        cmd_fallback = ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", "-c:s", "srt", output_srt]
        subprocess.run(cmd_fallback, capture_output=True, text=True)

    return output_srt if (os.path.isfile(output_srt) and os.path.getsize(output_srt) > 0) else None


def align_subtitles(base_entries, ja_entries, zh_entries):

    """Khớp các câu từ tiếng Nhật và tiếng Trung vào từng câu của file phụ đề gốc (tiếng Anh/Base)"""
    aligned = []
    for idx, base in enumerate(base_entries, 1):
        s_ms, e_ms = base['start'], base['end']

        matching_ja = [
            j['text'].replace('\n', ' ')
            for j in ja_entries
            if (max(s_ms, j['start']) < min(e_ms, j['end']) or abs(s_ms - j['start']) < 1500)
        ]
        matching_zh = [
            z['text'].replace('\n', ' ')
            for z in zh_entries
            if (max(s_ms, z['start']) < min(e_ms, z['end']) or abs(s_ms - z['start']) < 1500)
        ]

        aligned.append({
            'id': idx,
            'time': base['time_str'],
            'base': base['text'],
            'ja': ' / '.join(matching_ja) if matching_ja else '',
            'zh': ' / '.join(matching_zh) if matching_zh else ''
        })
    return aligned


def generate_prompt_for_batch(batch):
    """Tạo câu lệnh Prompt chuyên dụng cho mô hình AI dịch chuẩn phong cách Anime"""
    system_instruction = (
        "Bạn là một dịch giả Anime/Manga/Light Novel tiếng Việt chuyên nghiệp hàng đầu.\n"
        "Nhiệm vụ: Dịch danh sách các câu phụ đề sau sang TIẾNG VIỆT tự nhiên, mượt mà và đúng phong cách anime.\n"
        "\n"
        "NGUYÊN TẮC DỊCH ĐA NGÔN NGỮ QUAN TRỌNG:\n"
        "1. XƯƠNG CỐT TỪ TIẾNG ANH (Base): Dựa vào tiếng Anh để xác định rõ CHỦ NGỮ, VỊ NGỮ và NGỮ CẢNH (ai nói với ai, bối cảnh hành động) để không dịch sai chủ ngữ.\n"
        "2. TỪ VỰNG HÁN - VIỆT TỪ TIẾNG TRUNG (ZH): Với các danh xưng, địa danh, ma pháp, chức vị, thuật ngữ kỳ ảo, kiếm hiệp, hãy dùng âm Hán - Việt tương ứng từ tiếng Trung (ví dụ: Thần Long, Kiếm Thánh, Đại Tội Giám Mục, Phàm Ăn, Ma Nữ Đố Kỵ, Trở Về Từ Cõi Chết... KHÔNG dịch kiểu Tây hóa ngô nghê).\n"
        "3. NGÔI XƯNG & CẢM XÚC TỪ TIẾNG NHẬT (JA): Bắt đúng kính ngữ và đại từ nhân xưng theo vai vế anime (anh, em, cậu, tớ, tôi, sư phụ, bệ hạ, Betty, nhóc... KHÔNG dịch cứng nhắc thành 'tôi - bạn').\n"
        "\n"
        "ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:\n"
        "Trả về định dạng JSON hợp lệ duy nhất, ánh xạ ID sang câu dịch tiếng Việt. KHÔNG viết thêm bất kỳ lời dẫn giải nào.\n"
        "Ví dụ:\n"
        '{\n  "1": "Dấu tay của... ai thế này?",\n  "2": "Khoan đã..."\n}\n'
    )

    items_to_send = []
    for item in batch:
        line_info = {
            "id": item['id'],
            "EN": item['base']
        }
        if item['ja']:
            line_info["JA_Context"] = item['ja']
        if item['zh']:
            line_info["ZH_Context"] = item['zh']
        items_to_send.append(line_info)

    user_prompt = "Dưới đây là danh sách các câu cần dịch:\n" + json.dumps(items_to_send, ensure_ascii=False, indent=2)
    return system_instruction, user_prompt


def call_gemini(api_key, sys_prompt, user_prompt, model="gemini-2.5-flash"):
    """Gọi trực tiếp Google Gemini API qua HTTP request (không phụ thuộc SDK)"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {
            "parts": [{"text": sys_prompt}]
        },
        "contents": [{
            "parts": [{"text": user_prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }
    data = http_post_json(url, payload, timeout=60)
    text = data['candidates'][0]['content']['parts'][0]['text']
    return json.loads(clean_json_text(text))


def call_openai_compatible(api_key, endpoint, model, sys_prompt, user_prompt):
    """Gọi API tương thích OpenAI (DeepSeek, Groq, OpenRouter, OpenAI, Ollama)"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    data = http_post_json(endpoint, payload, headers=headers, timeout=90)
    content = data['choices'][0]['message']['content']
    return json.loads(clean_json_text(content))


def translate_subtitles_for_video(
    video_path,
    output_srt=None,
    api_key=None,
    api_type="gemini",
    model_name=None,
    endpoint="https://api.deepseek.com/chat/completions",
    progress_callback=None
):
    """
    Trích xuất phụ đề tự động từ video MKV/MP4, đối chiếu 3 thứ tiếng nếu có,
    và dịch chuẩn anime sang Tiếng Việt bằng AI.
    """
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

    if not api_key:
        if api_type == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
        else:
            api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "❌ Thiếu API Key! Vui lòng nhập Google Gemini API Key (miễn phí từ aistudio.google.com) "
            "hoặc DeepSeek / OpenAI API Key để dịch phụ đề."
        )

    scratch_dir = tempfile.gettempdir()
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if not output_srt:
        output_srt = os.path.join(scratch_dir, f"{base_name}.vi.srt")

    if progress_callback:
        progress_callback(0.05, desc="📝 [TriSub AI] Đang quét và trích xuất track phụ đề từ video...")

    # 1. Trích xuất phụ đề
    en_path = None
    ja_path = None
    zh_path = None

    if video_path.lower().endswith(('.srt', '.ass', '.vtt')):
        en_path = video_path
    else:
        tracks = get_sub_tracks(video_path)
        for t in tracks:
            lang = t.get("lang", "").lower()
            title = t.get("title", "").lower()
            idx = t.get("stream_index")

            if not en_path and (lang in ["eng", "en"] or "english" in title or "eng" in title):
                out_f = os.path.join(scratch_dir, f"{base_name}_en.srt")
                cmd = ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{idx}", "-c:s", "srt", out_f]
                if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 and os.path.exists(out_f) and os.path.getsize(out_f) > 50:
                    en_path = out_f

            if not ja_path and (lang in ["jpn", "ja", "jap"] or "japanese" in title or "nhật" in title):
                out_f = os.path.join(scratch_dir, f"{base_name}_ja.srt")
                cmd = ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{idx}", "-c:s", "srt", out_f]
                if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 and os.path.exists(out_f) and os.path.getsize(out_f) > 50:
                    ja_path = out_f

            if not zh_path and (lang in ["chi", "zho", "zh"] or "chinese" in title or "trung" in title):
                out_f = os.path.join(scratch_dir, f"{base_name}_zh.srt")
                cmd = ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{idx}", "-c:s", "srt", out_f]
                if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 and os.path.exists(out_f) and os.path.getsize(out_f) > 50:
                    zh_path = out_f

        # Nếu không tìm thấy tag cụ thể, fallback lấy track sub đầu tiên
        if not en_path and tracks:
            first_idx = tracks[0].get("stream_index", "s:0")
            out_f = os.path.join(scratch_dir, f"{base_name}_track0.srt")
            cmd = ["ffmpeg", "-y", "-i", video_path, "-map", f"0:{first_idx}", "-c:s", "srt", out_f]
            if subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 and os.path.exists(out_f) and os.path.getsize(out_f) > 50:
                en_path = out_f

    if not en_path:
        raise RuntimeError(f"❌ Không tìm thấy track phụ đề mềm nào trong file '{video_path}'!")

    # 2. Đọc và phân tích phụ đề
    base_entries = parse_subtitle(en_path)
    if not base_entries:
        raise ValueError("❌ Không trích xuất được câu thoại nào từ file phụ đề gốc!")

    ja_entries = parse_subtitle(ja_path) if ja_path else []
    zh_entries = parse_subtitle(zh_path) if zh_path else []

    if progress_callback:
        progress_callback(0.12, desc=f"📝 [TriSub AI] Đã nạp {len(base_entries)} câu thoại. Đang đồng bộ mốc thời gian...")

    aligned = align_subtitles(base_entries, ja_entries, zh_entries)
    
    # 3. Dịch batch bằng AI
    batch_size = 60
    total_batches = (len(aligned) + batch_size - 1) // batch_size
    translated_dict = {}

    for b_idx in range(total_batches):
        batch = aligned[b_idx * batch_size : (b_idx + 1) * batch_size]
        pct = 0.15 + (b_idx / total_batches) * 0.8
        if progress_callback:
            progress_callback(pct, desc=f"🤖 [TriSub AI] Đang dịch đợt {b_idx + 1}/{total_batches} (Câu {batch[0]['id']} -> {batch[-1]['id']})...")

        sys_prompt, user_prompt = generate_prompt_for_batch(batch)

        try:
            if api_type == "gemini":
                m = model_name or "gemini-2.5-flash"
                res_dict = call_gemini(api_key, sys_prompt, user_prompt, m)
            elif api_type == "openai":
                m = model_name or "gpt-4o-mini"
                ep = "https://api.openai.com/v1/chat/completions" if endpoint == "https://api.deepseek.com/chat/completions" else endpoint
                res_dict = call_openai_compatible(api_key, ep, m, sys_prompt, user_prompt)
            else:
                m = model_name or "deepseek-chat"
                res_dict = call_openai_compatible(api_key, endpoint, m, sys_prompt, user_prompt)

            for k, v in res_dict.items():
                translated_dict[str(k)] = v
        except Exception as e_batch:
            print(f"⚠️ Lỗi đợt dịch {b_idx + 1}: {e_batch}")
            for item in batch:
                translated_dict[str(item['id'])] = item['base']

    # 4. Xuất file SRT
    srt_lines = []
    for item in aligned:
        str_id = str(item['id'])
        text = translated_dict.get(str_id, item['base'])
        srt_lines.append(f"{item['id']}\n{item['time']}\n{text}\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_srt)), exist_ok=True)
    with open(output_srt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_lines))

    if progress_callback:
        progress_callback(0.98, desc="✅ [TriSub AI] Đã tạo file phụ đề Tiếng Việt hoàn chỉnh!")

    return output_srt



def get_sub_stream_count(video_path):
    """Đếm số lượng track phụ đề hiện có trong file video"""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "s",
            "-show_entries", "stream=index", "-of", "csv=p=0", video_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            lines = [l for l in res.stdout.strip().splitlines() if l.strip()]
            return len(lines)
    except Exception:
        pass
    return 0


def remux_to_mkv(video_path, srt_path, output_mkv=None):
    """Nhúng phụ đề tiếng Việt vào file MKV qua ffmpeg làm track mặc định"""
    if not output_mkv:
        base, _ = os.path.splitext(video_path)
        output_mkv = f"{base}.vi_subbed.mkv"

    print(f"\n[*] Đang nhúng phụ đề vào file MKV qua ffmpeg...")
    sub_count = get_sub_stream_count(video_path)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-sub_charenc", "UTF-8", "-i", srt_path,
        "-map", "0", "-map", "1:0",
        "-c", "copy",
        f"-metadata:s:s:{sub_count}", "language=vie",
        f"-metadata:s:s:{sub_count}", "title=Tiếng Việt (Chuẩn đối chiếu)",
        "-disposition:s", "0",
        f"-disposition:s:{sub_count}", "default",
        output_mkv
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"[✓] Đã tạo file video mới kèm phụ đề thành công: {output_mkv}")
        return True
    except Exception as e:
        print(f"[!] Lỗi khi nhúng bằng ffmpeg: {e}")
        return False



def main():
    parser = argparse.ArgumentParser(
        description="TriSubTranslate: Dịch phụ đề Anime chuẩn chỉnh đối chiếu 3 thứ tiếng (Anh - Nhật - Trung)"
    )
    parser.add_argument("--en", "--base", help="File phụ đề tiếng Anh (hoặc file gốc dùng để căn timestamp)")
    parser.add_argument("--ja", help="File phụ đề tiếng Nhật (đối chiếu ngôi xưng và cảm xúc)")
    parser.add_argument("--zh", help="File phụ đề tiếng Trung (đối chiếu thuật ngữ Hán - Việt)")
    parser.add_argument("-o", "--output", help="Đường dẫn file .srt tiếng Việt xuất ra")
    parser.add_argument("--mkv", help="Đường dẫn file video MKV để tự động nhúng phụ đề sau khi dịch")

    parser.add_argument("--remux-sub", help="Chỉ nhúng file phụ đề (.srt/.ass) này vào file --mkv mà không cần dịch lại")

    # Tùy chọn AI API
    parser.add_argument("--gemini-key", help="API Key của Google Gemini (miễn phí từ aistudio.google.com)")
    parser.add_argument("--openai-key", help="API Key của DeepSeek/OpenAI/Groq")
    parser.add_argument("--endpoint", default="https://api.deepseek.com/chat/completions", help="Endpoint API (mặc định: DeepSeek)")
    parser.add_argument("--model", help="Tên model (vd: gemini-2.5-flash hoặc deepseek-chat)")

    # Chế độ thủ công
    parser.add_argument("--export-prompt", help="Xuất danh sách prompt ra file text để copy dịch trên ChatGPT/Claude")
    parser.add_argument("--import-json", help="Nhập file JSON kết quả dịch thủ công để tạo file .srt")

    parser.add_argument("--gui", action="store_true", help="Khởi động giao diện Web trực quan trong trình duyệt")

    args = parser.parse_args()

    # Khởi động Giao diện Web nếu được yêu cầu
    if args.gui:
        import gui_server
        gui_server.start_server()
        return

    # Xử lý nhanh trường hợp chỉ muốn nhúng phụ đề có sẵn vào video
    if args.remux_sub and args.mkv:
        remux_to_mkv(args.mkv, args.remux_sub, args.output)
        return


    # Chế độ tương tác hỏi nếu chạy không có tham số
    if len(sys.argv) == 1:
        print("=" * 60)
        print("   TRI-SUB TRANSLATE (ANH - NHẬT - TRUNG -> TIẾNG VIỆT)")
        print("=" * 60)
        args.en = input("[?] Đường dẫn file phụ đề gốc (Tiếng Anh .srt/.ass): ").strip().strip('"\'')
        args.ja = input("[?] Đường dẫn file Tiếng Nhật (Enter nếu không có): ").strip().strip('"\'') or None
        args.zh = input("[?] Đường dẫn file Tiếng Trung (Enter nếu không có): ").strip().strip('"\'') or None
        args.output = input("[?] Tên file .srt xuất ra (Enter mặc định: vietsub.srt): ").strip() or "vietsub.srt"
        args.mkv = input("[?] File MKV để nhúng phụ đề luôn (Enter nếu không cần): ").strip().strip('"\'') or None

    if not args.en and args.mkv:
        print(f"[*] Không có file Tiếng Anh rời, đang tự động trích xuất phụ đề Tiếng Anh từ video: {os.path.basename(args.mkv)}...")
        extracted = extract_sub_from_video(args.mkv, "eng")
        if extracted:
            args.en = extracted
            print(f"[✓] Đã trích xuất thành công phụ đề Tiếng Anh từ video MKV!")
        else:
            print("[!] Không thể tự động trích xuất track Tiếng Anh từ file video.")

    if not args.en:
        print("[!] Lỗi: Bạn cần cung cấp ít nhất file phụ đề gốc (--en) hoặc file video có sẵn phụ đề (--mkv).")
        sys.exit(1)



    print(f"\n[*] Đang đọc dữ liệu phụ đề...")
    base_entries = parse_subtitle(args.en)
    print(f" [+] File gốc ({args.en}): {len(base_entries)} câu.")

    ja_entries = parse_subtitle(args.ja) if args.ja else []
    if ja_entries:
        print(f" [+] File Tiếng Nhật ({args.ja}): {len(ja_entries)} câu.")

    zh_entries = parse_subtitle(args.zh) if args.zh else []
    if zh_entries:
        print(f" [+] File Tiếng Trung ({args.zh}): {len(zh_entries)} câu.")

    print(f"[*] Đang đồng bộ đối chiếu các ngôn ngữ theo timestamp...")
    aligned = align_subtitles(base_entries, ja_entries, zh_entries)
    print(f"[✓] Đã đồng bộ hoàn tất {len(aligned)} câu thoại.")

    # 1. Chế độ xuất Prompt thủ công
    if args.export_prompt:
        sys_prompt, user_prompt = generate_prompt_for_batch(aligned)
        with open(args.export_prompt, 'w', encoding='utf-8') as f:
            f.write(f"=== SYSTEM PROMPT ===\n{sys_prompt}\n\n=== USER DATA ===\n{user_prompt}\n")
        print(f"[✓] Đã xuất file prompt tại: {args.export_prompt}")
        print("    Bạn có thể mở file này, copy toàn bộ nội dung dán vào Claude/ChatGPT/Gemini Web để dịch.")
        return

    # 2. Chế độ nhập kết quả JSON thủ công
    if args.import_json:
        with open(args.import_json, 'r', encoding='utf-8') as f:
            trans_dict = json.load(f)
        output_srt = args.output or "output.vi.srt"
        srt_lines = []
        for item in aligned:
            str_id = str(item['id'])
            text = trans_dict.get(str_id, trans_dict.get(item['id'], item['base']))
            srt_lines.append(f"{item['id']}\n{item['time']}\n{text}\n")
        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_lines))
        print(f"[✓] Đã tạo thành công file: {output_srt}")
        if args.mkv:
            remux_to_mkv(args.mkv, output_srt)
        return

    # 3. Dịch tự động qua API
    api_key_gemini = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    api_key_openai = args.openai_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")

    if not api_key_gemini and not api_key_openai:
        print("\n[!] Không tìm thấy API Key.")
        print("    Tùy chọn 1: Cung cấp API Key qua tham số `--gemini-key AIza...` hoặc `--openai-key sk-...`")
        print(f"    Tùy chọn 2: Dùng cờ `--export-prompt prompt.txt` để xuất dữ liệu dán vào web AI miễn phí.")
        choice = input("\n[?] Bạn có muốn xuất file prompt ra để dán vào Web AI (Claude/ChatGPT) không? (y/n): ").strip().lower()
        if choice == 'y':
            prompt_path = "prompt_dich_vietsub.txt"
            sys_prompt, user_prompt = generate_prompt_for_batch(aligned)
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(f"=== SYSTEM PROMPT ===\n{sys_prompt}\n\n=== USER PROMPT ===\n{user_prompt}\n")
            print(f"[✓] Đã xuất file: {prompt_path}")
            print(f"    Sau khi web trả về JSON, lưu vào `ket_qua.json` và chạy lệnh:")
            print(f"    python3 {sys.argv[0]} --en \"{args.en}\" --import-json ket_qua.json -o \"{args.output or 'vietsub.srt'}\"")
        return

    batch_size = 60
    total_batches = (len(aligned) + batch_size - 1) // batch_size
    translated_dict = {}

    print(f"\n[*] Bắt đầu dịch tự động qua AI (Tổng {len(aligned)} câu, chia làm {total_batches} đợt)...")

    for b_idx in range(total_batches):
        batch = aligned[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(f"  -> Đang dịch đợt {b_idx + 1}/{total_batches} (Câu {batch[0]['id']} đến {batch[-1]['id']})...")
        sys_prompt, user_prompt = generate_prompt_for_batch(batch)

        try:
            if api_key_gemini:
                model_name = args.model or "gemini-2.5-flash"
                res_dict = call_gemini(api_key_gemini, sys_prompt, user_prompt, model_name)
            else:
                model_name = args.model or "deepseek-chat"
                res_dict = call_openai_compatible(api_key_openai, args.endpoint, model_name, sys_prompt, user_prompt)

            for k, v in res_dict.items():
                translated_dict[str(k)] = v
        except Exception as e:
            print(f"  [!] Lỗi khi dịch đợt {b_idx + 1}: {e}")
            for item in batch:
                translated_dict[str(item['id'])] = item['base']

    # Xuất file .srt
    output_srt = args.output or "output.vi.srt"
    srt_lines = []
    for item in aligned:
        str_id = str(item['id'])
        text = translated_dict.get(str_id, item['base'])
        srt_lines.append(f"{item['id']}\n{item['time']}\n{text}\n")

    with open(output_srt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_lines))

    print(f"\n[✓] Hoàn tất! File phụ đề tiếng Việt đã tạo tại: {output_srt}")

    if args.mkv:
        remux_to_mkv(args.mkv, output_srt)


if __name__ == '__main__':
    main()
