import os
import sys
import time
import threading
import tempfile
import urllib.parse
import urllib.request
from queue import Queue
import importlib
import torch
import gradio as gr

try:
    gr.close_all()
except Exception:
    pass

import upscale
importlib.reload(upscale)
from upscale import upscale_video

for arg in sys.argv:
    if arg.startswith("--alias="):
        os.environ["TINYURL_ALIAS"] = arg.split("=", 1)[1]

device_obj, device_type, device_desc = upscale.get_best_device()
encoder_flags, encoder_desc = upscale.get_hevc_encoder_flags(device_type)

if device_type == 'mps':
    device_badge = f"🍎 Apple Silicon M3 Pro (MPS Metal Acceleration) | MÃ HÓA: {encoder_desc}"
elif device_type == 'cuda':
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_badge = f"🔥 {num_gpus}x NVIDIA CUDA (Multi-Processing) | MÃ HÓA: {encoder_desc}"
    else:
        device_badge = f"🚀 Single NVIDIA GPU (CUDA) | MÃ HÓA: {encoder_desc}"
else:
    device_badge = f"💻 CPU Software Mode | MÃ HÓA: {encoder_desc}"

MODEL_MAP = {
    "Real-CUGAN 2x Conservative (Mặc định cho SubsPlease Web-DL - Cực Nét Vector)": "cugan_conservative",
    "Real-CUGAN 2x No-Denoise (Giữ nguyên hạt - Tối ưu cho Blu-ray Remux)": "cugan_no_denoise",
    "Real-CUGAN 2x Denoise3x (Khử nhiễu nặng - Cho Anime cũ/nhiễu)": "cugan_denoise3x"
}

CUSTOM_CSS = """
.container {
    max-width: 1360px;
    margin: 0 auto;
    padding: 20px;
}
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    color: #ffffff;
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-box p {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
}
.badge {
    display: inline-block;
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid #0284c7;
    color: #38bdf8;
    font-family: monospace;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 9999px;
    margin-top: 14px;
}
.panel-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
}
"""

def create_tinyurl(target_url, custom_alias=None, api_token=None):
    custom_alias = custom_alias or os.environ.get("TINYURL_ALIAS")
    api_token = api_token or os.environ.get("TINYURL_API_TOKEN")

    if custom_alias:
        try:
            api_url = f"https://tinyurl.com/api-create.php?url={urllib.parse.quote(target_url)}&alias={urllib.parse.quote(custom_alias)}"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                res_url = resp.read().decode('utf-8').strip()
                if "tinyurl.com" in res_url:
                    return res_url
        except Exception:
            pass

    try:
        api_url = f"https://tinyurl.com/api-create.php?url={urllib.parse.quote(target_url)}"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode('utf-8').strip()
    except Exception:
        return None

def process_ui(magnet_or_path, video_file, model_choice, progress=gr.Progress(track_tqdm=True)):
    target_input = None
    if magnet_or_path and magnet_or_path.strip():
        target_input = magnet_or_path.strip()
    elif video_file is not None:
        target_input = video_file
    else:
        raise gr.Error("❌ Vui lòng dán Link Magnet (SubsPlease) HOẶC đường dẫn file HOẶC tải tệp video từ máy tính!")

    model_name = MODEL_MAP.get(model_choice, "cugan_conservative")
    progress_queue = Queue()

    def progress_cb(pct, desc=""):
        progress_queue.put((pct, desc))
        if pct is not None:
            progress(pct, desc=desc)

    yield None, gr.update(visible=False), f"⏳ Đang khởi tạo luồng giải mã Real-CUGAN Pro Native 2x (4K HEVC 10-bit)..."

    output_result = [None]
    error_result = [None]

    def worker():
        try:
            res = upscale.upscale_video(
                video_input=target_input,
                model_name=model_name,
                progress_callback=progress_cb
            )
            output_result[0] = res
        except Exception as e:
            error_result[0] = e
        finally:
            progress_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        try:
            item = progress_queue.get(timeout=0.2)
            if item is None:
                break
            pct, desc = item
            if desc:
                yield gr.update(), gr.update(), desc
        except Exception:
            if not thread.is_alive() and progress_queue.empty():
                break

    thread.join()

    if error_result[0]:
        raise gr.Error(f"❌ Lỗi xử lý: {str(error_result[0])}")

    output_path = output_result[0]
    yield output_path, gr.update(value=output_path, visible=True), f"✨ Nâng cấp thành công! Tập phim 4K Ultra-HD hoàn chỉnh (.mkv) sẵn sàng tải về."

with gr.Blocks(title="AI Video Upscaler 4K - Real-CUGAN Pro", theme=gr.themes.Default(), css=CUSTOM_CSS) as app:
    with gr.Column(elem_classes=["container"]):
        with gr.Group(elem_classes=["header-box"]):
            gr.Markdown(f"""
            # 🎬 AI Video Upscaler 4K - Real-CUGAN Pro
            Hệ thống chuyên dụng nâng cấp Anime 1080p lên **4K Ultra-HD (3840x2160 Native 2x)**. Khôi phục nét vẽ vector nguyên bản, mã hóa HEVC 10-bit chống banding và bảo tồn 100% Phụ đề mềm (.ass) & Âm thanh gốc.
            
            <div class="badge">THIẾT BỊ: {device_badge}</div>
            """)

        with gr.Accordion("📖 Hướng dẫn sử dụng nhanh (MacBook Pro Apple Silicon)", open=False):
            gr.Markdown("""
            ### 📖 Hướng Dẫn Sử Dụng
            1. **Nạp Tệp Anime**: Kéo thả tệp anime `.mkv` / `.mp4` vào ô tải lên HOẶC dán đường dẫn tệp trên máy (ví dụ: `/Users/xis108/Downloads/Mushoku_Tensei_14.mkv`).
            2. **Mô Hình AI**: Giữ nguyên mặc định `Real-CUGAN 2x Conservative` (Tối ưu tuyệt đối cho nguồn Web-DL Crunchyroll / SubsPlease).
            3. **Bắt Đầu**: Bấm **"🚀 Nâng Cấp Video 4K"**. Sau khi xử lý xong, tệp 4K Ultra-HD hoàn chỉnh sẽ nằm sẵn trong `~/Movies/Upscaled` và sẵn sàng xem ngay trên IINA!
            """)

        # 1. Ô NHẬP LINK MAGNET / ĐƯỜNG DẪN TẬP PHIM
        magnet_input = gr.Textbox(
            label="📁 Đường Dẫn File trên Mac / Kaggle HOẶC Link Magnet",
            placeholder="Ví dụ: /Users/xis108/Downloads/Mushoku_Tensei_S02E14.mkv hoặc magnet:?xt=urn:btih:...",
            lines=2
        )

        # 2. KHUNG VIDEO XEM TRƯỚC VÀ KẾT QUẢ
        with gr.Row(equal_height=True):
            file_input = gr.Video(
                label="📁 Hoặc Tải Tệp Video Từ Máy Tính (.mkv / .mp4)",
                sources=["upload"],
                scale=1
            )
            output_preview = gr.Video(
                label="✨ Video 4K Kết Quả (Real-CUGAN Pro 2x UHD)",
                interactive=False,
                scale=1
            )

        # 3. THANH TIẾN ĐỘ THỜI GIAN THỰC
        status_box = gr.Textbox(
            label="📊 Tiến Độ & Trạng Thái Thời Gian Thực (Live Progress)",
            value="Chờ dán link Magnet hoặc chọn tệp anime...",
            interactive=False
        )

        # 4. BẢNG CẤU HÌNH & NÚT BẮT ĐẦU / TẢI VỀ
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["panel-box"]):
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="Real-CUGAN 2x Conservative (Mặc định cho SubsPlease Web-DL - Cực Nét Vector)",
                        label="🤖 Mô Hình AI (Real-CUGAN Pro Native 2x)",
                        info="Native 2x phóng đại trực tiếp 1080p lên 4K thuần khiết, bảo toàn 100% màu sắc và nét vẽ gốc."
                    )
            with gr.Column(scale=5):
                submit_btn = gr.Button("🚀 Nâng Cấp Video 4K (Real-CUGAN Pro Native 2x)", variant="primary", size="lg")
                download_file = gr.File(
                    label="📥 Tải tệp 4K kết quả (.mkv đầy đủ Sub & Audio)",
                    visible=False
                )

        submit_btn.click(
            fn=process_ui,
            inputs=[magnet_input, file_input, model_dropdown],
            outputs=[output_preview, download_file, status_box]
        )

if __name__ == '__main__':
    share_mode = True if ("--share" in sys.argv or "--public" in sys.argv or os.environ.get("GRADIO_SHARE") == "True") else False
    allowed_dirs = ["/kaggle/working", "/tmp", tempfile.gettempdir(), os.getcwd()]
    
    app_obj, local_url, share_url = app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share_mode,
        allowed_paths=allowed_dirs,
        prevent_thread_lock=True
    )

    if share_url:
        print("\n" + "="*68, flush=True)
        print(f"🌐 GRADIO ORIGINAL URL: {share_url}", flush=True)
        tiny_url = create_tinyurl(share_url)
        if tiny_url:
            print(f"🔗 TINYURL SHORTLINK:   {tiny_url}", flush=True)
            print(f"💡 Dùng ngay link TinyURL trên để mở WebUI!", flush=True)
        print("="*68 + "\n", flush=True)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("🛑 Ứng dụng đã dừng.")
