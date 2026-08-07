import os
import sys
import time
import threading
import tempfile
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

CODEC_MAP = {
    "Tự động chọn phần cứng tốt nhất (Auto-detect)": "auto",
    "Nvidia GPU (h264_nvenc - Siêu tốc)": "h264_nvenc",
    "Nvidia GPU (hevc_nvenc - Chuẩn HEVC)": "hevc_nvenc",
    "Apple Silicon (hevc_videotoolbox)": "hevc_videotoolbox",
    "CPU (libx264 - H.264 chuẩn)": "libx264"
}

CUSTOM_CSS = """
.container {
    max-width: 1280px;
    margin: 0 auto;
    padding: 20px;
}
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    color: #ffffff;
}
.header-box h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 10px 0;
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
    margin-top: 16px;
}
.panel-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px;
    margin-top: 16px;
}
"""

def process_ui(video_file, codec_choice, res_choice, progress=gr.Progress(track_tqdm=True)):
    if video_file is None:
        raise gr.Error("❌ Vui lòng kéo thả hoặc chọn 1 tệp video MP4/MOV từ máy tính của bạn!")

    keep_highest = (res_choice == "Giữ tỷ lệ gốc tối đa (Keep Highest 4x)")
    encoder_codec = CODEC_MAP.get(codec_choice, "auto")

    progress_queue = Queue()

    def progress_cb(pct, desc=""):
        progress_queue.put((pct, desc))
        if pct is not None:
            progress(pct, desc=desc)

    yield None, gr.update(visible=False), "⏳ Đang khởi tạo luồng giải mã video AI 4K..."

    output_result = [None]
    error_result = [None]

    def worker():
        try:
            res = upscale.upscale_video(
                video_input=video_file,
                encoder_codec=encoder_codec,
                keep_highest=keep_highest,
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
    yield output_path, gr.update(value=output_path, visible=True), f"✨ Nâng cấp thành công! Tệp 4K kết quả sẵn sàng tải về."

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
            1. **Tải Tệp Video**: Kéo thả hoặc chọn tệp video (`.mp4`, `.mov`, `.mkv`...) ở cột bên trái. Bạn có thể xem trước video gốc ngay tại đây.
            2. **Cấu Hình**: Chọn bộ mã hóa phần cứng (NVENC GPU) và tùy chọn độ phân giải mong muốn.
            3. **Bắt Đầu Nâng Cấp**: Bấm nút **"🚀 Nâng Cấp Video 4K"** và theo dõi tiến độ thời gian thực ở cột bên phải.
            4. **Tải Về**: Trình phát 4K lớn ở cột bên phải sẽ hiển thị video nét căng. Bấm nút **"📥 Tải Tệp 4K Về Máy"** để hoàn tất.
            
            ---
            ### ⚡ Công Nghệ Nổi Bật
            - **Multi-Processing Dual GPU Split**: Tự động phân chia và xử lý song song trên cả 2 Card NVIDIA T4 (Kaggle) giúp tốc độ lên tới **16+ FPS** (rút ngắn thời gian xử lý từ vài phút xuống chỉ còn vài chục giây).
            - **Chuẩn Mã Hóa NVENC P4 HQ**: Giữ trọn vẹn chi tiết sắc nét cho cả cảnh cận cảnh nhân vật lẫn các cảnh hiệu ứng chiến đấu phức tạp.
            """)

        with gr.Row():
            # CỘT BÊN TRÁI: UPLOAD VIDEO GỐC (VỪA LÀ DROPZONE VỪA LÀ TRÌNH PHÁT VIDEO GỐC) & CẤU HÌNH
            with gr.Column(scale=5):
                file_input = gr.Video(
                    label="📁 Tệp Video Gốc (Kéo thả hoặc chọn tệp MP4, MOV, MKV...)",
                    sources=["upload"]
                )
                
                with gr.Group(elem_classes=["panel-box"]):
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

                submit_btn = gr.Button("🚀 Nâng Cấp Video 4K (Start Upscaling)", variant="primary", size="lg")

            # CỘT BÊN PHẢI: TIẾN ĐỘ & TRÌNH PHÁT KẾT QUẢ 4K TO TOÀN MÀN HÌNH + NÚT TẢI VỀ
            with gr.Column(scale=6):
                status_box = gr.Textbox(
                    label="📊 Tiến Độ & Trạng Thái Thời Gian Thực (Live Progress)",
                    value="Chờ tải tệp video...",
                    interactive=False
                )
                
                output_preview = gr.Video(
                    label="✨ Video 4K Kết Quả Sắc Nét (Upscaled 4K Video)",
                    interactive=False
                )
                
                download_file = gr.File(
                    label="📥 Tải tệp 4K kết quả về máy",
                    visible=False
                )

        submit_btn.click(
            fn=process_ui,
            inputs=[file_input, codec_dropdown, res_radio],
            outputs=[output_preview, download_file, status_box]
        )

if __name__ == '__main__':
    share_mode = True if ("--share" in sys.argv or "--public" in sys.argv) else False
    allowed_dirs = ["/kaggle/working", "/tmp", tempfile.gettempdir(), os.getcwd()]
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share_mode,
        allowed_paths=allowed_dirs
    )
