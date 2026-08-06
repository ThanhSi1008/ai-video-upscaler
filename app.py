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
device_name = "Nvidia GPU (CUDA)" if device.type == "cuda" else ("Apple Silicon GPU (MPS)" if device.type == "mps" else "CPU")

CODEC_MAP = {
    "Tự động (Auto-detect)": "auto",
    "Nvidia GPU (hevc_nvenc)": "hevc_nvenc",
    "Nvidia GPU (h264_nvenc)": "h264_nvenc",
    "Apple Silicon (hevc_videotoolbox)": "hevc_videotoolbox",
    "CPU (libx264)": "libx264",
    "CPU (libx265)": "libx265"
}

CUSTOM_CSS = """
.container {
    max-width: 1100px;
    margin: 0 auto;
    padding: 10px;
}
.tech-header {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 12px;
    margin-bottom: 20px;
}
.tech-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #0f172a;
    margin: 0 0 4px 0;
}
.tech-header p {
    font-size: 0.9rem;
    color: #64748b;
    margin: 0;
}
.sys-info {
    font-family: monospace;
    font-size: 0.82rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 6px 12px;
    border-radius: 4px;
    color: #334155;
    margin-top: 10px;
    display: inline-block;
}
"""

def process_ui(video_file, youtube_url, codec_choice, res_choice, progress=gr.Progress(track_tqdm=True)):
    video_input = None
    if youtube_url and youtube_url.strip():
        video_input = youtube_url.strip()
    elif video_file is not None:
        video_input = video_file
    else:
        raise gr.Error("❌ Vui lòng tải lên tệp video hoặc dán đường dẫn YouTube!")

    keep_highest = (res_choice == "Giữ tỷ lệ gốc tối đa (Keep Highest)")
    encoder_codec = CODEC_MAP.get(codec_choice, "auto")

    progress_queue = Queue()

    def progress_cb(pct, desc=""):
        progress_queue.put((pct, desc))
        if pct is not None:
            progress(pct, desc=desc)

    yield None, None, gr.update(visible=False), "⏳ Đang kết nối luồng và tải video..."

    output_result = [None]
    error_result = [None]

    def worker():
        try:
            res = upscale.upscale_video(
                video_input=video_input,
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
    yield video_input, output_path, gr.update(value=output_path, visible=True), f"✨ Hoàn tất nâng cấp video 4K thành công!"

with gr.Blocks(title="Video Upscaler & Encoder") as app:
    with gr.Column(elem_classes=["container"]):
        with gr.Group(elem_classes=["tech-header"]):
            gr.Markdown(f"""
            # Video Upscaler & Encoder Tool
            Công cụ mã hóa và nâng cấp độ phân giải video sử dụng mô hình Real-ESRGAN / SRVGGNetCompact.
            
            <div class="sys-info">SYSTEM: Device={device_name} | Recommended_Codec={default_codec}</div>
            """)

        with gr.Accordion("📖 Hướng dẫn sử dụng & Giải thích chi tiết chức năng", open=False):
            gr.Markdown("""
            ### 📖 Hướng Dẫn Sử Dụng Chi Tiết

            #### 1. Nguồn Video Đầu Vào (Input Source) ❓
            - **📁 Tải tệp Video**: Sử dụng khi bạn muốn nâng cấp tệp video sẵn có trên máy tính (hỗ trợ đầy đủ các định dạng như MP4, MKV, MOV, AVI...).
            - **🔗 Link YouTube**: Nhập trực tiếp đường dẫn video từ YouTube (ví dụ: `https://www.youtube.com/watch?v=...`). Hệ thống sẽ tự động tải video gốc về xử lý.

            #### 2. Bộ Mã Hóa Video (Video Encoder) ❓
            - **Tự động (Auto-detect)** *(Khuyên dùng)*: Hệ thống tự phát hiện và chọn bộ mã hóa tối ưu nhất cho thiết bị của bạn.
            - **Nvidia GPU (`hevc_nvenc` / `h264_nvenc`)**: Mã hóa bằng nhân phần cứng chuyên dụng trên card đồ họa NVIDIA (dành cho Kaggle GPU / Linux Server).
            - **Apple Silicon (`hevc_videotoolbox`)**: Tăng tốc mã hóa phần cứng chuyên dụng cho các dòng máy Mac chip Apple Silicon (M1/M2/M3/M4).
            - **CPU (`libx264` / `libx265`)**: Mã hóa bằng vi xử lý CPU (dành cho máy không trang bị card đồ họa GPU).

            #### 3. Tùy Chọn Độ Phân Giải (Output Resolution) ❓
            - **Đưa về 4K Ultra-HD (3840x2160)** *(Khuyên dùng)*: Nâng cấp AI và đưa video về độ phân giải chuẩn 4K sắc nét mà vẫn bảo toàn đúng tỷ lệ khung hình gốc (không bị méo hay biến dạng hình ảnh).
            - **Giữ tỷ lệ gốc tối đa (Keep Highest)**: Tự động nâng cấp AI lên gấp 4 lần kích thước gốc mà không áp dụng giới hạn chuẩn 4K.

            #### 4. Khôi Phục Tiến Trình & Dọn Dẹp Tự Động ❓
            - **Tự động khôi phục (Auto Resume)**: Nếu quá trình xử lý bị tạm dừng hoặc ngắt kết nối giữa chừng, khi bạn bật lại ứng dụng sẽ tự động chạy tiếp từ phần trăm bị dở mà không tốn công chạy lại từ đầu.
            - **Tự động dọn dẹp (Auto Cleanup)**: Sau khi hoàn thành, ứng dụng tự động xóa toàn bộ file rác và chỉ giữ lại duy nhất 1 tệp video 4K sắc nét để bạn tải về.
            """)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.TabItem("📁 Tải tệp Video"):
                        file_input = gr.Video(
                            label="📁 Tệp Video Đầu Vào ❓ (Hỗ trợ tệp MP4, MKV, MOV...)",
                            sources=["upload"]
                        )
                    with gr.TabItem("🔗 Link YouTube"):
                        url_input = gr.Textbox(
                            label="🔗 URL YouTube ❓",
                            info="Dán link YouTube (ví dụ: https://www.youtube.com/watch?v=...). Hệ thống sẽ tự động tải video gốc về xử lý.",
                            placeholder="https://www.youtube.com/watch?v=...",
                            lines=1
                        )
                
                with gr.Group():
                    codec_dropdown = gr.Dropdown(
                        choices=list(CODEC_MAP.keys()),
                        value="Tự động (Auto-detect)",
                        label="🎬 Bộ Mã Hóa Video (Video Encoder) ❓",
                        info="Tự động chọn phần cứng tốt nhất: NVENC (Nvidia GPU), VideoToolbox (Apple Mac), hoặc libx264 (CPU)."
                    )
                    res_radio = gr.Radio(
                        choices=["Đưa về 4K Ultra-HD (3840x2160)", "Giữ tỷ lệ gốc tối đa (Keep Highest)"],
                        value="Đưa về 4K Ultra-HD (3840x2160)",
                        label="📐 Tùy Chọn Độ Phân Giải Đầu Ra ❓",
                        info="4K Ultra-HD: Nâng cấp và chuẩn hóa về 4K sắc nét | Keep Highest: Nhân 4x kích thước gốc."
                    )

                submit_btn = gr.Button("🚀 Bắt đầu xử lý (Start Processing)", variant="primary", size="lg")

            with gr.Column(scale=6):
                status_box = gr.Textbox(
                    label="📊 Tiến Độ & Trạng Thái Xử Lý (Live Progress Status)",
                    value="Chờ bắt đầu...",
                    interactive=False
                )
                
                gr.Markdown("#### Xem trước & Kết quả")
                with gr.Row():
                    input_preview = gr.Video(label="Original Video", interactive=False)
                    output_preview = gr.Video(label="Upscaled 4K Video", interactive=False)
                
                download_file = gr.File(label="📥 Tải tệp 4K kết quả", visible=False)

        submit_btn.click(
            fn=process_ui,
            inputs=[file_input, url_input, codec_dropdown, res_radio],
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
