#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TriSub AI - Web GUI Server
Giao diện trực quan đẹp mắt để chọn 3 file phụ đề (Anh - Nhật - Trung) và file video MKV để dịch và nhúng tự động.
"""

import sys
import os
import json
import threading
import subprocess
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Import các hàm xử lý từ tri_sub_translate
import requests
from tri_sub_translate import (
    parse_subtitle, align_subtitles, generate_prompt_for_batch,
    call_gemini, call_openai_compatible, remux_to_mkv, get_sub_stream_count,
    extract_sub_from_video, get_sub_tracks
)

PORT = 5055

# Trạng thái tiến trình toàn cục
job_state = {
    "status": "idle",       # idle | running | completed | error
    "progress": 0,          # 0 - 100
    "current_step": "",
    "logs": [],
    "output_srt": "",
    "output_mkv": "",
    "error_message": ""
}
job_lock = threading.Lock()


def log_message(msg):
    with job_lock:
        job_state["logs"].append(msg)
        job_state["current_step"] = msg
    print(f"[UI] {msg}")


def pick_file_macos(prompt="Chọn file", file_types=None):
    """Mở hộp thoại chọn file gốc của macOS qua AppleScript"""
    type_filter = ""
    if file_types:
        quoted = ", ".join([f'"{t}"' for t in file_types])
        type_filter = f'of type {{{quoted}}}'

    apple_script = f'''
    tell application "System Events"
        activate
        try
            set theFile to choose file with prompt "{prompt}" {type_filter} default location (path to home folder)
            return POSIX path of theFile
        on error number -128
            return ""
        end try
    end tell
    '''
    try:
        p = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
        return p.stdout.strip()
    except Exception as e:
        return ""


def run_translation_job(config):
    global job_state
    try:
        with job_lock:
            job_state["status"] = "running"
            job_state["progress"] = 5
            job_state["logs"] = []
            job_state["error_message"] = ""
            job_state["output_srt"] = ""
            job_state["output_mkv"] = ""

        en_path = config.get("en_path", "")
        ja_path = config.get("ja_path", "")
        zh_path = config.get("zh_path", "")
        mkv_path = config.get("mkv_path", "")
        output_srt = config.get("output_srt", "") or "vietsub_translated.srt"
        api_type = config.get("api_type", "gemini")
        api_key = config.get("api_key", "").strip()
        model_name = config.get("model_name", "")
        endpoint = config.get("endpoint", "") or "https://api.deepseek.com/chat/completions"

        # Tự động trích xuất phụ đề Tiếng Anh từ Video MKV nếu không có file rời
        if not en_path:
            if mkv_path and os.path.isfile(mkv_path):
                log_message(f"Không có file Tiếng Anh rời. Đang tự động trích xuất track Tiếng Anh từ video {os.path.basename(mkv_path)}...")
                extracted = extract_sub_from_video(mkv_path, "eng")
                if extracted and os.path.isfile(extracted):
                    en_path = extracted
                    log_message(f"✓ Đã trích xuất thành công track Tiếng Anh: {os.path.basename(extracted)}")
                else:
                    raise ValueError(f"Không thể trích xuất phụ đề Tiếng Anh từ video {os.path.basename(mkv_path)}. Vui lòng chọn file phụ đề Tiếng Anh rời!")
            else:
                raise ValueError("Vui lòng chọn File Tiếng Anh HOẶC chọn file Video MKV có chứa phụ đề!")

        log_message(f"Đang phân tích file phụ đề gốc ({os.path.basename(en_path)})...")
        base_entries = parse_subtitle(en_path)
        if not base_entries:
            raise ValueError("Không tìm thấy câu thoại nào trong file phụ đề gốc!")

        ja_entries = parse_subtitle(ja_path) if ja_path else []
        zh_entries = parse_subtitle(zh_path) if zh_path else []

        log_message(f"Đã đọc: {len(base_entries)} câu gốc, {len(ja_entries)} câu Nhật, {len(zh_entries)} câu Trung.")
        log_message("Đang đồng bộ và đối chiếu dữ liệu 3 ngôn ngữ theo mốc thời gian...")
        
        aligned = align_subtitles(base_entries, ja_entries, zh_entries)
        with job_lock:
            job_state["progress"] = 15

        batch_size = 60
        total_batches = (len(aligned) + batch_size - 1) // batch_size
        translated_dict = {}

        log_message(f"Bắt đầu dịch AI ({len(aligned)} câu, chia làm {total_batches} đợt)...")

        for b_idx in range(total_batches):
            batch = aligned[b_idx * batch_size : (b_idx + 1) * batch_size]
            log_message(f"Đang dịch đợt {b_idx + 1}/{total_batches} (Câu {batch[0]['id']} -> {batch[-1]['id']})...")
            sys_prompt, user_prompt = generate_prompt_for_batch(batch)

            if api_type == "gemini":
                m = model_name or "gemini-2.5-flash"
                res_dict = call_gemini(api_key, sys_prompt, user_prompt, m)
            else:
                m = model_name or "deepseek-chat"
                res_dict = call_openai_compatible(api_key, endpoint, m, sys_prompt, user_prompt)

            for k, v in res_dict.items():
                translated_dict[str(k)] = v

            pct = 15 + int(((b_idx + 1) / total_batches) * 70)
            with job_lock:
                job_state["progress"] = min(pct, 85)

        # Xuất file SRT
        log_message(f"Đang kết xuất file phụ đề tiếng Việt chuẩn: {output_srt}")
        srt_lines = []
        for item in aligned:
            str_id = str(item['id'])
            text = translated_dict.get(str_id, item['base'])
            srt_lines.append(f"{item['id']}\n{item['time']}\n{text}\n")

        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_lines))

        with job_lock:
            job_state["output_srt"] = os.path.abspath(output_srt)
            job_state["progress"] = 90

        # Nếu có yêu cầu nhúng vào file MKV
        if mkv_path and os.path.isfile(mkv_path):
            log_message(f"Đang nhúng phụ đề vào file MKV qua ffmpeg (Stream Copy)...")
            base, ext = os.path.splitext(mkv_path)
            output_mkv = f"{base}.vietsub.mkv"
            success = remux_to_mkv(mkv_path, output_srt, output_mkv)
            if success:
                with job_lock:
                    job_state["output_mkv"] = os.path.abspath(output_mkv)
                log_message(f"Nhúng video thành công! File xuất ra: {output_mkv}")
            else:
                log_message("Lỗi khi nhúng video bằng ffmpeg (vui lòng kiểm tra log terminal).")

        with job_lock:
            job_state["progress"] = 100
            job_state["status"] = "completed"
        log_message("Hoàn tất mọi thao tác thành công rực rỡ!")

    except Exception as e:
        with job_lock:
            job_state["status"] = "error"
            job_state["error_message"] = str(e)
        log_message(f"LỖI: {e}")


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TriSub AI - Anime Subtitle Localization Suite</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0d14;
            --bg-card: rgba(18, 24, 38, 0.7);
            --bg-card-hover: rgba(26, 35, 54, 0.85);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(99, 102, 241, 0.4);
            --primary: #6366f1;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --accent-cyan: #06b6d4;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --radius-lg: 16px;
            --radius-md: 10px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.1) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.05) 0%, transparent 60%);
            background-attachment: fixed;
            padding: 40px 20px;
        }

        .container {
            max-width: 1050px;
            margin: 0 auto;
        }

        /* Header */
        header {
            text-align: center;
            margin-bottom: 40px;
        }

        .badge {
            display: inline-block;
            padding: 6px 14px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
            color: #c7d2fe;
            margin-bottom: 14px;
            letter-spacing: 0.5px;
        }

        h1 {
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.1rem;
            color: var(--text-muted);
            max-width: 650px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* Grid Cards */
        .grid-2x2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 28px;
        }

        @media (max-width: 768px) {
            .grid-2x2 {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 24px;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
        }

        .card:hover {
            border-color: var(--border-glow);
            transform: translateY(-2px);
            box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        .card-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.05rem;
            font-weight: 600;
        }

        .flag-icon {
            font-size: 1.3rem;
        }

        .tag-required {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            padding: 3px 8px;
            border-radius: 6px;
            background: rgba(236, 72, 153, 0.15);
            color: #f472b6;
            border: 1px solid rgba(236, 72, 153, 0.3);
        }

        .tag-optional {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 6px;
            background: rgba(148, 163, 184, 0.1);
            color: #94a3b8;
        }

        .card-desc {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 16px;
            line-height: 1.45;
        }

        .input-row {
            display: flex;
            gap: 8px;
        }

        input[type="text"] {
            flex: 1;
            background: rgba(10, 14, 23, 0.75);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 14px;
            color: var(--text-main);
            font-size: 0.88rem;
            font-family: 'JetBrains Mono', monospace;
            outline: none;
            transition: border-color 0.2s;
        }

        input[type="text"]:focus {
            border-color: var(--primary);
        }

        input[type="text"]:disabled {
            background: rgba(255, 255, 255, 0.03);
            color: #94a3b8;
            cursor: not-allowed;
            border-color: rgba(255, 255, 255, 0.06);
        }

        .btn-browse {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-subtle);
            color: var(--text-main);
            padding: 10px 16px;
            border-radius: var(--radius-md);
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            white-space: nowrap;
            transition: all 0.2s;
        }

        .btn-browse:hover:not(:disabled) {
            background: rgba(99, 102, 241, 0.2);
            border-color: var(--primary);
        }

        .btn-browse:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            pointer-events: none;
        }

        /* Hướng dẫn tìm sub & API Key */
        .guide-box {
            margin-top: 14px;
            padding: 11px 14px;
            background: rgba(15, 23, 42, 0.7);
            border: 1px dashed rgba(255, 255, 255, 0.12);
            border-radius: var(--radius-md);
            font-size: 0.79rem;
            line-height: 1.55;
            color: #cbd5e1;
        }

        .guide-box a {
            color: var(--accent-cyan);
            text-decoration: underline;
            font-weight: 600;
        }

        .guide-badge {
            display: inline-block;
            color: #a5b4fc;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .guide-steps-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 16px;
        }

        @media (max-width: 850px) {
            .guide-steps-grid {
                grid-template-columns: 1fr;
            }
        }

        .guide-step-card {
            background: rgba(10, 14, 23, 0.65);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 14px;
            font-size: 0.8rem;
            line-height: 1.55;
        }

        .guide-step-card strong {
            color: #f8fafc;
        }

        .guide-step-card a {
            color: var(--accent-cyan);
            text-decoration: underline;
            font-weight: 600;
        }

        /* Config Card */

        .config-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-bottom: 28px;
        }

        .config-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .config-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 768px) {
            .config-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        select {
            background: rgba(10, 14, 23, 0.75);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 10px 14px;
            color: var(--text-main);
            font-size: 0.88rem;
            font-family: inherit;
            outline: none;
        }

        select option {
            background: #111827;
        }

        /* Actions Button */
        .btn-start {
            width: 100%;
            padding: 18px 28px;
            background: var(--primary-gradient);
            border: none;
            border-radius: var(--radius-lg);
            color: #ffffff;
            font-size: 1.15rem;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.4);
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            letter-spacing: 0.3px;
        }

        .btn-start:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px -5px rgba(168, 85, 247, 0.5);
        }

        .btn-start:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Progress Area */
        .progress-box {
            display: none;
            margin-top: 30px;
            background: rgba(15, 20, 31, 0.85);
            border: 1px solid var(--border-glow);
            border-radius: var(--radius-lg);
            padding: 24px;
            animation: fadeIn 0.3s ease;
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .progress-title {
            font-weight: 600;
            font-size: 0.95rem;
        }

        .progress-percent {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--accent-cyan);
        }

        .progress-bar-bg {
            height: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            overflow: hidden;
            margin-bottom: 16px;
        }

        .progress-bar-fill {
            height: 100%;
            width: 0%;
            background: var(--primary-gradient);
            border-radius: 9999px;
            transition: width 0.3s ease;
        }

        .log-terminal {
            background: #06090e;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-md);
            padding: 14px;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #94a3b8;
            line-height: 1.6;
        }

        .log-terminal div {
            margin-bottom: 4px;
        }

        .log-success { color: var(--accent-emerald); }
        .log-error { color: #f43f5e; font-weight: 600; }
        .log-info { color: var(--accent-cyan); }

        .result-box {
            display: none;
            margin-top: 20px;
            padding: 18px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: var(--radius-md);
            color: #6ee7b7;
            font-size: 0.95rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">ANIME SUBTITLE SUITE • MULTI-LANGUAGE CROSS-REF</div>
            <h1>TriSub AI</h1>
            <p class="subtitle">Dịch phụ đề Anime chuẩn mực sang tiếng Việt bằng phương pháp đối chiếu song song 3 thứ tiếng (Anh – Nhật – Trung) và tự động nhúng vào file video MKV.</p>
        </header>

        <!-- Form Chọn File -->
        <div class="grid-2x2">
            <!-- File Tiếng Anh -->
            <div class="card" id="card_en">
                <div class="card-header">
                    <div class="card-title">
                        <span class="flag-icon">🇬🇧</span>
                        <span>File Tiếng Anh (Gốc)</span>
                    </div>
                    <span id="en_tag" class="tag-required">Bắt buộc</span>
                </div>
                <p class="card-desc">Cung cấp bộ khung logic, xác định rõ chủ ngữ - vị ngữ và làm mốc căn timestamp.</p>
                <div class="input-row">
                    <input type="text" id="en_path" placeholder="Đường dẫn file .srt / .ass / .vtt">
                    <button class="btn-browse" id="btn_en_browse" onclick="pickFile('en', 'Chọn file phụ đề Tiếng Anh')">📂 Chọn file</button>
                </div>
                <div id="mkv_auto_notice" style="display: none; margin-top: 12px; padding: 12px 14px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; font-size: 0.85rem; color: #a7f3d0; line-height: 1.5;">
                    <div style="font-weight: 600; color: #34d399; margin-bottom: 2px;">
                        ✨ Đã tìm thấy phụ đề Tiếng Anh trong Video MKV (<span id="mkv_eng_detail">Track #2</span>)
                    </div>
                    <span>Hệ thống đã <strong>tự động khóa và sử dụng phụ đề từ video</strong>. Bạn không cần làm gì thêm ở ô này!</span>
                    <div style="margin-top: 6px;">
                        <button type="button" onclick="unlockEnInput()" style="background: none; border: none; color: #67e8f9; text-decoration: underline; cursor: pointer; font-size: 0.8rem; padding: 0;">Muốn dùng file .srt rời bên ngoài? Bấm mở khóa ô nhập</button>
                    </div>
                </div>
            </div>

            <!-- File Tiếng Nhật -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <span class="flag-icon">🇯🇵</span>
                        <span>File Tiếng Nhật (Ngữ điệu)</span>
                    </div>
                    <span class="tag-optional">Khuyên dùng</span>
                </div>
                <p class="card-desc">Bắt đúng đại từ xưng hô (anh/em, sư phụ, cậu/tớ), kính ngữ và cảm xúc gốc của nhân vật.</p>
                <div class="input-row">
                    <input type="text" id="ja_path" placeholder="Đường dẫn file .srt / .ass (tùy chọn)">
                    <button class="btn-browse" onclick="pickFile('ja', 'Chọn file phụ đề Tiếng Nhật')">📂 Chọn file</button>
                </div>
                <div class="guide-box">
                    <span class="guide-badge">🔍 Tìm tải phụ đề Tiếng Nhật chuẩn ở đâu?</span><br>
                    • <a href="https://kitsunekko.net/" target="_blank">Kitsunekko.net</a>: Kho phụ đề tiếng Nhật lớn nhất thế giới cho anime. Vào mục <em>Japanese Subtitles</em> gõ tên Anime (tiếng Anh hoặc Romaji) để tải file <code>.srt</code> chuẩn 100% kịch bản phòng thu.<br>
                    • <a href="https://jimaku.cc/" target="_blank">Jimaku.cc</a>: Tìm và tải phụ đề tiếng Nhật trích từ các dịch vụ streaming Nhật (Netflix JP, Amazon Prime JP).
                </div>
            </div>

            <!-- File Tiếng Trung -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <span class="flag-icon">🇨🇳</span>
                        <span>File Tiếng Trung (Hán - Việt)</span>
                    </div>
                    <span class="tag-optional">Khuyên dùng</span>
                </div>
                <p class="card-desc">Cung cấp kho thuật ngữ Hán - Việt chuẩn cho ma pháp, danh xưng, chức vị (ưu tiên Phồn Thể).</p>
                <div class="input-row">
                    <input type="text" id="zh_path" placeholder="Đường dẫn file .srt / .ass (tùy chọn)">
                    <button class="btn-browse" onclick="pickFile('zh', 'Chọn file phụ đề Tiếng Trung')">📂 Chọn file</button>
                </div>
                <div class="guide-box">
                    <span class="guide-badge">🔍 Tìm tải phụ đề Tiếng Trung (Phồn Thể) ở đâu?</span><br>
                    • <a href="https://subhd.tv/" target="_blank">SubHD.tv</a> hoặc <a href="http://www.zimuku.org/" target="_blank">Zimuku.org</a>: Kho phụ đề tiếng Trung lớn nhất. Tìm tên phim tiếng Trung, ưu tiên tải file có chữ <strong>[Baha]</strong>, <strong>[ANi]</strong>, hoặc <strong>TC / CHT / 繁體</strong> (Bản quyền Đài Loan Bahamut không bị kiểm duyệt).<br>
                    • <a href="https://github.com/ani-gamer" target="_blank">GitHub ANi Gamer</a>: Kho phim & sub rip từ Bahamut Đài Loan.
                </div>
            </div>

            <!-- File Video MKV -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <span class="flag-icon">🎬</span>
                        <span>File Video (.mkv)</span>
                    </div>
                    <span class="tag-optional">Tự động trích xuất & nhúng</span>
                </div>
                <p class="card-desc">Tự trích xuất sub Tiếng Anh có sẵn và tự động nhúng phụ đề Tiếng Việt sau khi dịch.</p>
                <div class="input-row">
                    <input type="text" id="mkv_path" placeholder="Đường dẫn file .mkv cần nhúng" oninput="checkMkvSubtitles(this.value)">
                    <button class="btn-browse" onclick="pickFile('mkv', 'Chọn file Video MKV')">📂 Chọn file</button>
                </div>
                <div id="mkv_track_list" style="display: none; margin-top: 10px; font-size: 0.82rem; color: var(--text-muted); line-height: 1.4;"></div>
            </div>
        </div>

        <!-- Cấu hình AI & Đầu ra -->
        <div class="config-card">
            <div class="config-title">🔑 Cấu Hình Mô Hình AI & Nhập API Key</div>
            
            <!-- Hộp gợi ý chọn AI -->
            <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 20px; font-size: 0.88rem; line-height: 1.5;">
                <span style="color: #a5b4fc; font-weight: 700;">💡 Nên chọn AI nào để dịch?</span><br>
                • <strong>Google Gemini (Khuyên dùng số 1):</strong> Hoàn toàn <strong>MIỄN PHÍ 100%</strong> qua Google AI Studio, dịch cực nhanh (~20s/tập), nhớ ngữ cảnh nhân vật xuyên suốt tập phim.<br>
                • <strong>DeepSeek:</strong> Xuất sắc về từ vựng Hán - Việt cổ phong, chi phí siêu rẻ (~50đ/tập) nhưng cần nạp tiền tài khoản.
            </div>

            <div class="config-grid">
                <div class="form-group">
                    <label class="form-label">Chọn Mô hình AI</label>
                    <select id="api_type" onchange="toggleApiFields()">
                        <option value="gemini">🌟 Google Gemini 2.5 Flash (Khuyên dùng - Miễn phí 100%)</option>
                        <option value="gemini-pro">🧠 Google Gemini 1.5 Pro (Văn phong sâu sắc - Miễn phí)</option>
                        <option value="deepseek">🇨🇳 DeepSeek V3 (Siêu đỉnh Hán tự - Rất rẻ)</option>
                        <option value="openai">🤖 OpenAI GPT-4o Mini (Ổn định)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label class="form-label">
                        API Key
                        <span id="key_status" style="margin-left: 8px; font-size: 0.8rem;"></span>
                    </label>
                    <div style="display: flex; gap: 8px;">
                        <input type="password" id="api_key" placeholder="Dán API Key (AIzaSy... hoặc sk-...)" style="flex: 1;">
                        <button type="button" class="btn-browse" onclick="toggleKeyVisibility()" id="btn_toggle_key" title="Ẩn/Hiện Key" style="padding: 10px 12px;">👁️</button>
                        <button type="button" class="btn-browse" onclick="testApiKey()" id="btn_test_key" style="padding: 10px 14px; background: rgba(99, 102, 241, 0.25); color: #c7d2fe; border-color: rgba(99, 102, 241, 0.5);">⚡ Kiểm tra</button>
                    </div>
                    <div id="key_hint" style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                        👉 Chưa có Key? <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: var(--accent-cyan); text-decoration: underline;">Nhấp vào đây để lấy API Key Gemini miễn phí trong 30s</a> (Chỉ cần đăng nhập Gmail).
                    </div>
                </div>

                <div class="form-group" id="endpoint_group" style="display: none;">
                    <label class="form-label">API Endpoint</label>
                    <input type="text" id="endpoint" value="https://api.deepseek.com/chat/completions">
                </div>

                <div class="form-group">
                    <label class="form-label">Tên file phụ đề xuất ra (.srt)</label>
                    <input type="text" id="output_srt" value="vietsub_translated.srt">
                </div>
            </div>

            <!-- Hướng dẫn chi tiết từng bước lấy API Key -->
            <div style="margin-top: 24px; border-top: 1px solid var(--border-subtle); padding-top: 18px;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                    <span>📖 Hướng Dẫn Từng Bước Lấy API Key Của Từng Loại AI:</span>
                </div>
                <div class="guide-steps-grid">
                    <div class="guide-step-card" style="border-color: rgba(99, 102, 241, 0.4);">
                        <strong style="color: #a5b4fc;">🌟 1. Google Gemini (Miễn phí 100%)</strong><br>
                        1. Truy cập <a href="https://aistudio.google.com/app/apikey" target="_blank">Google AI Studio</a>.<br>
                        2. Đăng nhập bằng tài khoản Gmail (không cần thẻ ngân hàng).<br>
                        3. Bấm nút xanh <strong>"Create API key"</strong>.<br>
                        4. Copy chuỗi mã (bắt đầu bằng <code>AIzaSy...</code>) rồi dán vào ô API Key.<br>
                        <em>✨ Miễn phí mãi mãi, tốc độ siêu nhanh (~20s/tập).</em>
                    </div>

                    <div class="guide-step-card">
                        <strong style="color: #38bdf8;">🇨🇳 2. DeepSeek (Siêu rẻ, đỉnh Hán tự)</strong><br>
                        1. Truy cập <a href="https://platform.deepseek.com/" target="_blank">DeepSeek Platform</a>.<br>
                        2. Đăng ký tài khoản (bằng Email).<br>
                        3. Vào mục <strong>API Keys</strong> ➔ Bấm <strong>"Create new API key"</strong>.<br>
                        4. Copy mã key (bắt đầu bằng <code>sk-...</code>) rồi dán vào ô API Key.<br>
                        <em>💰 Chi phí cực rẻ (~30 - 50đ / tập phim), cần nạp tiền tài khoản.</em>
                    </div>

                    <div class="guide-step-card">
                        <strong style="color: #34d399;">🤖 3. OpenAI (ChatGPT / GPT-4o)</strong><br>
                        1. Truy cập <a href="https://platform.openai.com/api-keys" target="_blank">OpenAI Dashboard</a>.<br>
                        2. Đăng nhập tài khoản OpenAI.<br>
                        3. Bấm <strong>"Create new secret key"</strong> ➔ Đặt tên cho key.<br>
                        4. Copy mã key (bắt đầu bằng <code>sk-...</code>) rồi dán vào ô API Key.<br>
                        <em>💳 Cần có thẻ tín dụng quốc tế (Visa/Mastercard) đã nạp tiền.</em>
                    </div>
                </div>
            </div>
        </div>



        <!-- Nút Kích Hoạt -->
        <button id="btn_run" class="btn-start" onclick="startTranslation()">
            <span>✨ Bắt Đầu Dịch Đối Chiếu & Nhúng Phụ Đề</span>
        </button>

        <!-- Khung hiển thị tiến trình -->
        <div class="progress-box" id="progress_box">
            <div class="progress-header">
                <span class="progress-title" id="progress_step">Đang khởi tạo tiến trình...</span>
                <span class="progress-percent" id="progress_pct">0%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" id="progress_fill"></div>
            </div>
            <div class="log-terminal" id="terminal_logs"></div>
            
            <div class="result-box" id="result_box">
                <p><strong>🎉 Đã dịch và xử lý thành công!</strong></p>
                <p id="result_srt" style="margin-top: 6px;"></p>
                <p id="result_mkv" style="margin-top: 4px;"></p>
            </div>
        </div>
    </div>

    <script>
        // Khôi phục API Key từ localStorage
        window.onload = function() {
            const savedKey = localStorage.getItem('trisub_api_key');
            if (savedKey) document.getElementById('api_key').value = savedKey;
        };

        function toggleKeyVisibility() {
            const input = document.getElementById('api_key');
            const btn = document.getElementById('btn_toggle_key');
            if (input.type === 'password') {
                input.type = 'text';
                btn.innerText = '🔒';
            } else {
                input.type = 'password';
                btn.innerText = '👁️';
            }
        }

        async function testApiKey() {
            const apiType = document.getElementById('api_type').value;
            const apiKey = document.getElementById('api_key').value.trim();
            const endpoint = document.getElementById('endpoint').value.trim();
            const statusEl = document.getElementById('key_status');
            const btn = document.getElementById('btn_test_key');

            if (!apiKey) {
                alert("Vui lòng nhập API Key trước khi kiểm tra!");
                return;
            }

            btn.disabled = true;
            btn.innerText = "⏳ Đang thử...";
            statusEl.innerHTML = "<span style='color: var(--accent-cyan);'>Đang kết nối...</span>";

            try {
                const res = await fetch('/api/test-key', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        api_type: (apiType.startsWith('gemini')) ? 'gemini' : apiType,
                        api_key: apiKey,
                        endpoint: endpoint,
                        model: (apiType === 'gemini-pro') ? 'gemini-1.5-pro' : (apiType === 'openai') ? 'gpt-4o-mini' : 'deepseek-chat'
                    })
                });
                const data = await res.json();
                if (data.ok) {
                    statusEl.innerHTML = "<span style='color: var(--accent-emerald); font-weight: 700;'>✅ " + data.message + "</span>";
                    localStorage.setItem('trisub_api_key', apiKey);
                } else {
                    statusEl.innerHTML = "<span style='color: #f43f5e; font-weight: 700;'>❌ " + data.message + "</span>";
                }
            } catch (err) {
                statusEl.innerHTML = "<span style='color: #f43f5e;'>❌ Lỗi: " + err.message + "</span>";
            } finally {
                btn.disabled = false;
                btn.innerText = "⚡ Kiểm tra";
            }
        }

        function toggleApiFields() {
            const apiType = document.getElementById('api_type').value;
            const endpointGroup = document.getElementById('endpoint_group');
            const keyHint = document.getElementById('key_hint');
            
            if (apiType === 'deepseek') {
                endpointGroup.style.display = 'flex';
                document.getElementById('endpoint').value = 'https://api.deepseek.com/chat/completions';
                keyHint.innerHTML = "👉 Lấy API Key tại: <a href='https://platform.deepseek.com/' target='_blank' style='color: var(--accent-cyan); text-decoration: underline;'>DeepSeek Platform</a> (Chi phí siêu rẻ).";
            } else if (apiType === 'openai') {
                endpointGroup.style.display = 'flex';
                document.getElementById('endpoint').value = 'https://api.openai.com/v1/chat/completions';
                keyHint.innerHTML = "👉 Lấy API Key tại: <a href='https://platform.openai.com/api-keys' target='_blank' style='color: var(--accent-cyan); text-decoration: underline;'>OpenAI Dashboard</a>.";
            } else {
                endpointGroup.style.display = 'none';
                keyHint.innerHTML = "👉 Chưa có Key? <a href='https://aistudio.google.com/app/apikey' target='_blank' style='color: var(--accent-cyan); text-decoration: underline;'>Nhấp vào đây để lấy API Key Gemini miễn phí trong 30s</a> (Chỉ cần đăng nhập Gmail).";
            }
        }


        async function pickFile(targetId, prompt) {
            try {
                const res = await fetch(`/api/browse?prompt=${encodeURIComponent(prompt)}`);
                const data = await res.json();
                if (data.path) {
                    document.getElementById(targetId + '_path').value = data.path;
                    if (targetId === 'mkv') {
                        checkMkvSubtitles(data.path);
                    }
                }
            } catch (err) {
                alert("Lỗi khi mở Finder: " + err.message);
            }
        }

        let isEnLockedByMkv = false;

        function unlockEnInput() {
            isEnLockedByMkv = false;
            const enInput = document.getElementById('en_path');
            const enBtn = document.getElementById('btn_en_browse');
            const notice = document.getElementById('mkv_auto_notice');
            const enTag = document.getElementById('en_tag');

            enInput.disabled = false;
            enInput.value = '';
            enInput.style.opacity = '1';
            enInput.style.cursor = 'text';
            enInput.style.backgroundColor = '';
            enInput.placeholder = 'Đường dẫn file .srt / .ass / .vtt';
            enBtn.disabled = false;
            enBtn.style.opacity = '1';
            enBtn.style.cursor = 'pointer';
            enTag.className = 'tag-required';
            enTag.style.background = '';
            enTag.style.color = '';
            enTag.style.borderColor = '';
            enTag.innerText = 'Bắt buộc';
            notice.style.display = 'none';
        }

        async function checkMkvSubtitles(path) {
            const notice = document.getElementById('mkv_auto_notice');
            const enTag = document.getElementById('en_tag');
            const enInput = document.getElementById('en_path');
            const enBtn = document.getElementById('btn_en_browse');
            const trackListDiv = document.getElementById('mkv_track_list');

            if (!path || !path.trim()) {
                unlockEnInput();
                trackListDiv.style.display = 'none';
                return;
            }

            try {
                const res = await fetch(`/api/inspect-mkv?path=${encodeURIComponent(path.trim())}`);
                const data = await res.json();
                if (data.valid && data.has_english) {
                    const t = data.english_track;
                    const desc = `Track #${t.stream_index} [${t.codec}/${t.lang}]${t.title ? ' - ' + t.title : ''}`;
                    document.getElementById('mkv_eng_detail').innerText = desc;
                    notice.style.display = 'block';
                    enTag.className = 'tag-optional';
                    enTag.style.background = 'rgba(16, 185, 129, 0.2)';
                    enTag.style.color = '#34d399';
                    enTag.style.borderColor = 'rgba(16, 185, 129, 0.4)';
                    enTag.innerText = '🔒 Đã tự động lấy từ MKV';

                    // Vô hiệu hóa ô nhập Tiếng Anh và nút chọn file
                    isEnLockedByMkv = true;
                    enInput.value = `[Tự động từ Video: ${desc}]`;
                    enInput.disabled = true;
                    enInput.style.opacity = '0.6';
                    enInput.style.cursor = 'not-allowed';
                    enInput.style.backgroundColor = 'rgba(255, 255, 255, 0.04)';
                    enBtn.disabled = true;
                    enBtn.style.opacity = '0.4';
                    enBtn.style.cursor = 'not-allowed';

                    if (data.tracks && data.tracks.length > 0) {
                        trackListDiv.style.display = 'block';
                        trackListDiv.innerHTML = `📼 Video có ${data.tracks.length} track phụ đề: ` + data.tracks.map(tr => 
                            `<span style="display:inline-block; padding:2px 7px; margin:2px; border-radius:4px; font-size:0.78rem; background:rgba(255,255,255,0.06); font-family:monospace; ${tr.lang === 'eng' ? 'color:#34d399; font-weight:bold; border:1px solid rgba(52,211,153,0.4);' : ''}">${tr.lang} (#${tr.stream_index})</span>`
                        ).join('');
                    }
                } else if (data.valid) {
                    unlockEnInput();
                    if (data.tracks && data.tracks.length > 0) {
                        trackListDiv.style.display = 'block';
                        trackListDiv.innerHTML = `📼 Video có ${data.tracks.length} track phụ đề (không có tiếng Anh): ` + data.tracks.map(tr => 
                            `<span style="display:inline-block; padding:2px 7px; margin:2px; border-radius:4px; font-size:0.78rem; background:rgba(255,255,255,0.06); font-family:monospace;">${tr.lang} (#${tr.stream_index})</span>`
                        ).join('');
                    }
                } else {
                    unlockEnInput();
                    trackListDiv.style.display = 'none';
                }
            } catch (e) {
                console.error("Lỗi khi kiểm tra MKV:", e);
            }
        }

        let pollInterval = null;

        async function startTranslation() {
            let en_path = document.getElementById('en_path').value.trim();
            if (isEnLockedByMkv || document.getElementById('en_path').disabled || en_path.startsWith("[")) {
                en_path = "";
            }
            const ja_path = document.getElementById('ja_path').value.trim();
            const zh_path = document.getElementById('zh_path').value.trim();
            const mkv_path = document.getElementById('mkv_path').value.trim();
            const api_type = document.getElementById('api_type').value;
            const api_key = document.getElementById('api_key').value.trim();
            const endpoint = document.getElementById('endpoint').value.trim();
            const output_srt = document.getElementById('output_srt').value.trim();

            if (!en_path && !mkv_path) {
                alert("Vui lòng chọn ít nhất File Tiếng Anh (Gốc) HOẶC chọn file Video MKV có chứa phụ đề!");
                return;
            }
            if (!api_key) {
                alert("Vui lòng nhập API Key để mô hình AI hoạt động!");
                return;
            }

            localStorage.setItem('trisub_api_key', api_key);

            document.getElementById('btn_run').disabled = true;
            document.getElementById('progress_box').style.display = 'block';
            document.getElementById('result_box').style.display = 'none';

            const payload = {
                en_path, ja_path, zh_path, mkv_path,
                api_type, api_key, endpoint, output_srt
            };

            try {
                const res = await fetch('/api/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.error) {
                    alert("Lỗi: " + data.error);
                    document.getElementById('btn_run').disabled = false;
                    return;
                }

                // Bắt đầu thăm dò tiến trình
                pollInterval = setInterval(pollProgress, 1000);
            } catch (err) {
                alert("Lỗi kết nối tới máy chủ nội bộ: " + err.message);
                document.getElementById('btn_run').disabled = false;
            }
        }

        async function pollProgress() {
            try {
                const res = await fetch('/api/progress');
                const data = await res.json();

                document.getElementById('progress_pct').innerText = data.progress + '%';
                document.getElementById('progress_fill').style.width = data.progress + '%';
                document.getElementById('progress_step').innerText = data.current_step || 'Đang xử lý...';

                const term = document.getElementById('terminal_logs');
                term.innerHTML = data.logs.map(log => {
                    let cls = '';
                    if (log.includes('thành công') || log.includes('Hoàn tất')) cls = 'log-success';
                    else if (log.includes('LỖI') || log.includes('lỗi')) cls = 'log-error';
                    else if (log.includes('Đang')) cls = 'log-info';
                    return `<div class="${cls}">[${new Date().toLocaleTimeString()}] ${log}</div>`;
                }).join('');
                term.scrollTop = term.scrollHeight;

                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    document.getElementById('btn_run').disabled = false;
                    const resBox = document.getElementById('result_box');
                    resBox.style.display = 'block';
                    document.getElementById('result_srt').innerHTML = `📄 <strong>File phụ đề:</strong> ${data.output_srt}`;
                    if (data.output_mkv) {
                        document.getElementById('result_mkv').innerHTML = `🎬 <strong>File video:</strong> ${data.output_mkv}`;
                    }
                } else if (data.status === 'error') {
                    clearInterval(pollInterval);
                    document.getElementById('btn_run').disabled = false;
                    alert("Đã xảy ra lỗi: " + data.error_message);
                }
            } catch (e) {
                console.error(e);
            }
        }
    </script>
</body>
</html>
'''


class SubtitleServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Giảm bớt log thừa thãi ra terminal
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))

        elif parsed.path == "/api/browse":
            query = parse_qs(parsed.query)
            prompt = query.get("prompt", ["Chọn file"])[0]
            chosen_path = pick_file_macos(prompt)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"path": chosen_path}).encode('utf-8'))

        elif parsed.path == "/api/progress":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with job_lock:
                self.wfile.write(json.dumps(job_state).encode('utf-8'))

        elif parsed.path == "/api/inspect-mkv":
            query = parse_qs(parsed.query)
            video_path = query.get("path", [""])[0]
            if not video_path or not os.path.isfile(video_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"valid": False, "message": "File không tồn tại"}).encode('utf-8'))
                return

            tracks = get_sub_tracks(video_path)
            eng_track = None
            for t in tracks:
                t_lang = t["lang"].lower()
                t_title = t["title"].lower()
                if t_lang in ["eng", "en"] or "english" in t_title:
                    eng_track = t
                    break

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "valid": True,
                "tracks_count": len(tracks),
                "has_english": eng_track is not None,
                "english_track": eng_track,
                "tracks": tracks
            }).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            config = json.loads(body)

            with job_lock:
                if job_state["status"] == "running":
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Tiến trình dịch đang chạy, vui lòng chờ hoàn tất!"}).encode('utf-8'))
                    return

            thread = threading.Thread(target=run_translation_job, args=(config,), daemon=True)
            thread.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started"}).encode('utf-8'))

        elif parsed.path == "/api/test-key":

            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            req_data = json.loads(body)
            api_type = req_data.get("api_type", "gemini")
            api_key = req_data.get("api_key", "").strip()
            endpoint = req_data.get("endpoint", "https://api.deepseek.com/chat/completions")

            if not api_key:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "message": "Chưa nhập API Key!"}).encode('utf-8'))
                return

            try:
                if api_type == "gemini":
                    test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
                    r = requests.post(test_url, json={"contents": [{"parts": [{"text": "Hello"}]}]}, timeout=15)
                    if r.status_code == 200:
                        msg = "Kết nối Google Gemini thành công! Key hoàn toàn hợp lệ."
                        ok = True
                    else:
                        msg = f"Lỗi Gemini ({r.status_code}): {r.text[:120]}"
                        ok = False
                else:
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    payload = {"model": req_data.get("model", "deepseek-chat"), "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}
                    r = requests.post(endpoint, json=payload, headers=headers, timeout=15)
                    if r.status_code == 200:
                        msg = "Kết nối DeepSeek / OpenAI thành công! Key hợp lệ."
                        ok = True
                    else:
                        msg = f"Lỗi ({r.status_code}): {r.text[:120]}"
                        ok = False
            except Exception as ex:
                ok = False
                msg = f"Lỗi kết nối: {str(ex)[:100]}"

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": ok, "message": msg}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()



def start_server():
    server = HTTPServer(('127.0.0.1', PORT), SubtitleServerHandler)
    url = f"http://127.0.0.1:{PORT}"
    print("=" * 65)
    print(f"🚀 TriSub AI Server đã khởi động thành công!")
    print(f"👉 Đang mở trình duyệt tại: {url}")
    print("   Nhấn Ctrl + C trong Terminal để dừng máy chủ bất cứ lúc nào.")
    print("=" * 65)
    
    # Mở trình duyệt sau 0.5s
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Đã tắt máy chủ.")
        server.server_close()


if __name__ == '__main__':
    start_server()
