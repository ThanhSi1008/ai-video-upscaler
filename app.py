import os
import time
import threading
from queue import Queue
import importlib
import torch
import gradio as gr

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
    max-width: 1200px;
    margin: 0 auto;
    padding: 16px;
}
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    color: #ffffff;
}
.header-box h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    color: #38bdf8;
}
.header-box p {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
}
.badge {
    display: inline-block;
    background: #0284c7;
    color: #ffffff;
    font-family: monospace;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-top: 14px;
}
.panel-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
}
"""

def process_ui(video_file, codec_choice, res_choice, progress=gr.Progress(track_tqdm=True)):
    if video_file is None:
        raise gr.Error("❌ Vui lòng kéo hoặc chọn 1 tệp video MP4/MOV từ máy tính của bạn!")

    keep_highest = (res_choice == "Giữ tỷ lệ gốc tối đa (Keep Highest 4x)")
    encoder_codec = CODEC_MAP.get(codec_choice, "auto")

    progress_queue = Queue()

    def progress_cb(pct, desc=""):
        progress_queue.put((pct, desc))
        if pct is not None:
            progress(pct, desc=desc)

    yield None, None, gr.update(visible=False), "⏳ Đang khởi tạo luồng giải mã video AI 4K..."

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
                yield gr.update(), gr.update(), gr.update(), desc
        except Exception:
            if not thread.is_alive() and progress_queue.empty():
                break

    thread.join()

    if error_result[0]:
        raise gr.Error(f"❌ Lỗi xử lý: {str(error_result[0])}")

    output_path = output_result[0]
    yield video_file, output_path, gr.update(value=output_path, visible=True), f"✨ Nâng cấp thành công! Tệp 4K kết quả sẵn sàng tải về."

with gr.Blocks(title="AI Video Upscaler 4K - WebUI") as app:
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
            1. **Tải Tệp Video**: Kéo thả hoặc bấm chọn tệp video (`.mp4`, `.mov`, `.mkv`...) từ máy tính của bạn.
            2. **Cấu Hình**: Chọn bộ mã hóa phần cứng (NVENC GPU) và độ phân giải mong muốn.
            3. **Bắt Đầu Nâng Cấp**: Bấm nút **"🚀 Nâng Cấp Video 4K"** và theo dõi tiến độ thời gian thực.
            4. **Tải Về**: Xem trước kết quả sắc nét 4K và bấm nút **"📥 Tải Tệp 4K Về Máy"**.
            
            ---
            ### ⚡ Công Nghệ Nổi Bật
            - **Multi-Processing Dual GPU Split**: Tự động phân chia và xử lý song song trên cả 2 Card NVIDIA T4 (Kaggle) giúp tốc độ lên tới **16+ FPS** (rút ngắn thời gian xử lý 1 phút 35 giây xuống chỉ còn vài chục giây).
            - **Mã Hóa Tách Luồng (Decoupled Stream)**: Đảm bảo 100% video hoàn tất đủ thời lượng và đồng bộ âm thanh gốc cực kỳ sắc nét.
            """)

        with gr.Row():
            with gr.Column(scale=5):
                file_input = gr.Video(
                    label="📁 Tệp Video Nguồn (Kéo thả hoặc chọn tệp MP4, MOV, MKV...)",
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

            with gr.Column(scale=6):
                status_box = gr.Textbox(
                    label="📊 Tiến Độ & Trạng Thái Thời Gian Thực (Live Progress)",
                    value="Chờ tải tệp video...",
                    interactive=False
                )
                
                gr.Markdown("### 🎬 Trình Phát Xem Trước & Kết Quả")
                with gr.Row():
                    input_preview = gr.Video(label="Video Gốc (Original Input)", interactive=False)
                    output_preview = gr.Video(label="Video 4K Sắc Nét (Upscaled Output)", interactive=False)
                
                download_file = gr.File(label="📥 Tải tệp 4K kết quả về máy", visible=False)

        submit_btn.click(
            fn=process_ui,
            inputs=[file_input, codec_dropdown, res_radio],
            outputs=[input_preview, output_preview, download_file, status_box]
        )

if __name__ == '__main__':
    theme = gr.themes.Default()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=CUSTOM_CSS
    )
