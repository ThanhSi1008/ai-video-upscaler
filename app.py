import os
import sys
import time
import threading
import tempfile
import json
import urllib.parse
import urllib.request
from queue import Queue
import importlib
import torch
import gradio as gr

# Clear any active Gradio servers/event loops to prevent Python 3.12 asyncio conflicts
try:
    gr.close_all()
except Exception:
    pass

# Reload upscale module để luôn áp dụng mã nguồn mới nhất trong RAM
import upscale
importlib.reload(upscale)
from upscale import upscale_video, get_device_and_codec

# Xác định phần cứng hiện tại
device, default_codec = get_device_and_codec("auto")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

if device.type == "cuda":
    if num_gpus >= 2:
        device_name = f"🔥 Dual NVIDIA T4 GPUs (Multi-Processing ~16+ FPS)"
    else:
        device_name = f"🚀 Single NVIDIA GPU (CUDA)"
elif device.type == "mps":
    device_name = "🍏 Apple Silicon GPU (Metal/MPS)"
else:
    device_name = "💻 CPU (x86_64)"

MODEL_MAP = {
    "AnimeVideoV3 (Mô hình Siêu Tốc 16+ FPS - Khuyên Dùng)": "animevideov3",
    "Real-ESRGAN x4Plus Anime 6B (Mô hình Siêu Nét Master Class - Chi tiết cực cao)": "x4plus_anime"
}

CODEC_MAP = {
    "Tự động chọn phần cứng tốt nhất (Auto-detect)": "auto",
    "Nvidia GPU (h264_nvenc - Master Quality -qp 14)": "h264_nvenc",
    "Nvidia GPU (hevc_nvenc - Master Quality HEVC)": "hevc_nvenc",
    "Apple Silicon (hevc_videotoolbox)": "hevc_videotoolbox",
    "CPU (libx264 - H.264 chuẩn)": "libx264"
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
    try:
        custom_alias = custom_alias or os.environ.get("TINYURL_ALIAS")
        api_token = api_token or os.environ.get("TINYURL_API_TOKEN")

        if api_token and custom_alias:
            req_data = json.dumps({
                "url": target_url,
                "domain": "tinyurl.com",
                "alias": custom_alias
            }).encode('utf-8')
            req = urllib.request.Request(
                "https://api.tinyurl.com/create",
                data=req_data,
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode())
                return res.get("data", {}).get("tiny_url")
        else:
            api_url = f"https://tinyurl.com/api-create.php?url={urllib.parse.quote(target_url)}"
            if custom_alias:
                api_url += f"&alias={urllib.parse.quote(custom_alias)}"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode('utf-8').strip()
    except Exception as e:
        print(f"⚠️ TinyURL Notice: {e}")
        return None

def process_ui(video_file, model_choice, codec_choice, res_choice, detail_strength, color_boost, progress=gr.Progress(track_tqdm=True)):
    if video_file is None:
        raise gr.Error("❌ Vui lòng kéo thả hoặc chọn 1 tệp video MP4/MOV từ máy tính của bạn!")

    keep_highest = (res_choice == "Giữ tỷ lệ gốc tối đa (Keep Highest 4x)")
    encoder_codec = CODEC_MAP.get(codec_choice, "auto")
    model_name = MODEL_MAP.get(model_choice, "animevideov3")

    progress_queue = Queue()

    def progress_cb(pct, desc=""):
        progress_queue.put((pct, desc))
        if pct is not None:
            progress(pct, desc=desc)

    yield None, gr.update(visible=False), f"⏳ Đang khởi tạo luồng giải mã video AI 4K Master Class..."

    output_result = [None]
    error_result = [None]

    def worker():
        try:
            res = upscale.upscale_video(
                video_input=video_file,
                model_name=model_name,
                encoder_codec=encoder_codec,
                keep_highest=keep_highest,
                detail_strength=float(detail_strength),
                color_boost=color_boost,
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
    yield output_path, gr.update(value=output_path, visible=True), f"✨ Nâng cấp thành công! Tệp 4K kết quả Master Quality sẵn sàng tải về."

with gr.Blocks(title="AI Video Upscaler 4K - WebUI", theme=gr.themes.Default(), css=CUSTOM_CSS) as app:
    with gr.Column(elem_classes=["container"]):
        with gr.Group(elem_classes=["header-box"]):
            gr.Markdown(f"""
            # 🎬 AI Video Upscaler 4K - Ultra High Speed
            Nâng cấp và tăng tốc video lên độ phân giải **4K Ultra-HD (3840x2160)** bằng mô hình AI Real-ESRGAN chuyên dụng.
            
            <div class="badge">THIẾT BỊ: {device_name} | KHUYÊN DÙNG: {default_codec}</div>
            """)

        with gr.Accordion("📖 Hướng dẫn sử dụng & Thông số kỹ thuật", open=False):
            gr.Markdown("""
            ### 📖 Hướng Dẫn Sử Dụng
            1. **Tải Tệp Video Gốc**: Kéo thả hoặc chọn tệp video (`.mp4`, `.mov`, `.mkv`...) vào ô **"Video Gốc (Original Input)"** ở bên trái.
            2. **Cấu Hình Tối Ưu**: 
               - **Mô Hình AI**: Chọn `Real-ESRGAN x4Plus Anime 6B` nếu bạn muốn khôi phục chi tiết cực sâu cho từng nét vẽ nhân vật và hạt kỹ xảo.
               - **Cường Độ Chi Tiết**: Tùy chỉnh thanh trượt từ `0.0` đến `1.0` (Khuyên dùng `0.35` - `0.60`).
               - **Anime 4K HDR Color Boost**: Bật tăng cường độ rực rỡ và độ tương phản màu tương tự chuẩn 4K HDR.
            3. **Bắt Đầu Nâng Cấp**: Bấm nút **"🚀 Nâng Cấp Video 4K"** và theo dõi thanh tiến độ thời gian thực trực quan ngay bên dưới 2 khung video.
            4. **Xem Trước & Tải Về**: Video 4K sắc nét xuất hiện ở khung bên phải **"Video 4K Kết Quả"**. Bấm nút **"📥 Tải Tệp 4K Về Máy"** để hoàn tất.
            
            ---
            ### ⚡ Công Nghệ Tăng Cường Chi Tiết Đột Phá
            - **Mạng Neural RRDBNet 6B**: Kiến trúc Residual-in-Residual Dense Block giúp tái tạo nét vẽ Anime sắc sảo như bản vẽ Vector gốc.
            - **Bộ Lọc GPU Dynamic Contrast & Color Vibrance**: Tối ưu hóa màu sắc rực rỡ và độ tương phản sâu tự nhiên chuẩn 4K HDR.
            - **Multi-Processing Dual GPU Split**: Tự động phân chia và xử lý song song trên cả 2 Card NVIDIA T4 (Kaggle) giúp tốc độ lên tới **16+ FPS**.
            - **Lọc 5x5 Laplacian Pyramid GPU Filter**: Thuật toán phục hồi chi tiết kim tự tháp 5x5 trực tiếp trên PyTorch Tensor.
            - **NVENC Spatial & Temporal AQ (-qp 14 Master Quality)**: Phân bổ bitrate thông minh cho từng vùng chi tiết cao và chuyển động nhanh.
            """)

        # 1. 2 KHUNG VIDEO NẰM NGANG HÀNG NHAU (SIDE-BY-SIDE EQUAL HEIGHT & EQUAL WIDTH)
        with gr.Row(equal_height=True):
            file_input = gr.Video(
                label="📁 Video Gốc (Original Input - Drag & Drop)",
                sources=["upload"],
                scale=1
            )
            output_preview = gr.Video(
                label="✨ Video 4K Kết Quả (Upscaled 4K Video)",
                interactive=False,
                scale=1
            )

        # 2. THANH TIẾN ĐỘ THỜI GIAN THỰC ĐƯỢC CHUYỂN XUỐNG DƯỚI 2 KHUNG VIDEO
        status_box = gr.Textbox(
            label="📊 Tiến Độ & Trạng Thái Thời Gian Thực (Live Progress)",
            value="Chờ tải tệp video...",
            interactive=False
        )

        # 3. BẢNG CẤU HÌNH & NÚT BẮT ĐẦU / TẢI VỀ
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Group(elem_classes=["panel-box"]):
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="AnimeVideoV3 (Mô hình Siêu Tốc 16+ FPS - Khuyên Dùng)",
                        label="🤖 Mô Hình AI Nâng Cấp (AI Upscale Model)",
                        info="Real-ESRGAN x4Plus Anime 6B là mô hình sâu chuyên tái tạo chi tiết vi mô cực nét."
                    )
                    codec_dropdown = gr.Dropdown(
                        choices=list(CODEC_MAP.keys()),
                        value="Tự động chọn phần cứng tốt nhất (Auto-detect)",
                        label="🎬 Bộ Mã Hóa Phần Cứng (Video Encoder)",
                        info="Tự động chọn mã hóa phần cứng siêu tốc NVENC (Nvidia GPU)."
                    )
                    res_radio = gr.Radio(
                        choices=["Đưa về 4K Ultra-HD (3840x2160)", "Giữ tỷ lệ gốc tối đa (Keep Highest 4x)"],
                        value="Đưa về 4K Ultra-HD (3840x2160)",
                        label="📐 Tùy Chọn Độ Phân Giải Đầu Ra",
                        info="4K Ultra-HD: Chuẩn hóa 4K sắc nét bảo toàn tỷ lệ gốc."
                    )
                    detail_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.35,
                        step=0.05,
                        label="✨ Cường Độ Tăng Cường Chi Tiết Vi Mô (5x5 GPU Laplacian Filter)",
                        info="0.0: Mặc định gốc | 0.35: Sắc Nét Cao (Khuyên Dùng) | 0.60+: Siêu Sắc Nét Cực Hạn (Master Quality)"
                    )
                    vivid_checkbox = gr.Checkbox(
                        label="🎨 Tăng Cường Độ Tương Phản & Độ Rực Rỡ Màu Sắc (Anime 4K HDR Color Boost)",
                        value=True,
                        info="Giúp màu sắc phim đậm đà, đường nét viền đen sâu hơn và các hiệu ứng ánh sáng/kỹ xảo rực rỡ hơn."
                    )
            with gr.Column(scale=6):
                submit_btn = gr.Button("🚀 Nâng Cấp Video 4K (Start Upscaling)", variant="primary", size="lg")
                download_file = gr.File(
                    label="📥 Tải tệp 4K kết quả về máy",
                    visible=False
                )

        submit_btn.click(
            fn=process_ui,
            inputs=[file_input, model_dropdown, codec_dropdown, res_radio, detail_slider, vivid_checkbox],
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
            print(f"💡 Bạn có thể lưu hoặc dùng ngay link TinyURL trên để mở WebUI!", flush=True)
        print("="*68 + "\n", flush=True)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("🛑 Ứng dụng đã dừng.")
