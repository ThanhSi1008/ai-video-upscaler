import os
import torch
import gradio as gr
from upscale import upscale_video, get_device_and_codec

# Xác định phần cứng hiện tại
device, default_codec = get_device_and_codec("auto")
device_name = "Nvidia GPU (CUDA)" if device.type == "cuda" else ("Apple Silicon GPU (MPS)" if device.type == "mps" else "CPU")

CODEC_MAP = {
    "Tự động (Khuyên dùng)": "auto",
    "Nvidia GPU (hevc_nvenc)": "hevc_nvenc",
    "Nvidia GPU (h264_nvenc)": "h264_nvenc",
    "Apple Silicon (hevc_videotoolbox)": "hevc_videotoolbox",
    "CPU (libx264)": "libx264",
    "CPU (libx265)": "libx265"
}

CUSTOM_CSS = """
.container {
    max-width: 1200px;
    margin: 0 auto;
}
.header-box {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(168, 85, 247, 0.3);
}
.header-box h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.header-box p {
    font-size: 1.1rem;
    color: #cbd5e1;
}
.device-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.4);
    color: #4ade80;
    border-radius: 20px;
    font-weight: 600;
    margin-top: 0.5rem;
}
"""

def process_ui(video_file, youtube_url, codec_choice, res_choice, progress=gr.Progress()):
    video_input = None
    if youtube_url and youtube_url.strip():
        video_input = youtube_url.strip()
    elif video_file is not None:
        video_input = video_file
    else:
        raise gr.Error("❌ Vui lòng tải lên 1 tệp video hoặc dán đường dẫn YouTube!")

    keep_highest = (res_choice == "Giữ nguyên tỷ lệ tối đa (Keep Highest 4K+)")
    encoder_codec = CODEC_MAP.get(codec_choice, "auto")

    def progress_cb(pct, desc=""):
        if pct is not None:
            progress(pct, desc=desc)
        else:
            progress(0, desc=desc)

    try:
        output_path = upscale_video(
            video_input=video_input,
            encoder_codec=encoder_codec,
            keep_highest=keep_highest,
            progress_callback=progress_cb
        )
        return video_input, output_path, gr.update(value=output_path, visible=True)
    except Exception as e:
        raise gr.Error(f"❌ Lỗi xử lý: {str(e)}")

with gr.Blocks(title="AI Video Upscaler 4K") as app:
    with gr.Column(elem_classes=["container"]):
        with gr.Group(elem_classes=["header-box"]):
            gr.Markdown(f"""
            # ✨ AI Anime Video Super-Resolution & 4K Upscaler
            Nâng cấp chất lượng video Anime / Video sắc nét 4K bằng mạng nơ-ron nhân tạo **Real-ESRGAN (SRVGGNetCompact)**.
            
            <div class="device-badge">⚡ Phần cứng phát hiện: {device_name} (Codec đề xuất: {default_codec})</div>
            """)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.TabItem("📁 Tải lên Video"):
                        file_input = gr.Video(label="Tệp video đầu vào (MP4, MKV, MOV)", sources=["upload"])
                    with gr.TabItem("🔗 Link YouTube"):
                        url_input = gr.Textbox(
                            label="Đường dẫn Video YouTube",
                            placeholder="https://www.youtube.com/watch?v=...",
                            lines=1
                        )
                
                with gr.Group():
                    codec_dropdown = gr.Dropdown(
                        choices=list(CODEC_MAP.keys()),
                        value="Tự động (Khuyên dùng)",
                        label="🎬 Bộ Mã Hóa Video (Video Encoder)"
                    )
                    res_radio = gr.Radio(
                        choices=["Đưa về chuẩn 4K Ultra-HD (Khuyên dùng)", "Giữ nguyên tỷ lệ tối đa (Keep Highest 4K+)"],
                        value="Đưa về chuẩn 4K Ultra-HD (Khuyên dùng)",
                        label="📐 Tùy Chọn Độ Phân Giải Đầu Ra"
                    )

                submit_btn = gr.Button("🚀 Bắt đầu Nâng cấp Video (Start Upscale)", variant="primary", size="lg")

            with gr.Column(scale=6):
                gr.Markdown("### 📺 Kết Quả So Sánh (Video Preview)")
                with gr.Row():
                    input_preview = gr.Video(label="Original Video", interactive=False)
                    output_preview = gr.Video(label="Upscaled 4K Video", interactive=False)
                
                download_file = gr.File(label="📥 Tải xuống Video 4K hoàn chỉnh", visible=False)

        submit_btn.click(
            fn=process_ui,
            inputs=[file_input, url_input, codec_dropdown, res_radio],
            outputs=[input_preview, output_preview, download_file]
        )

if __name__ == '__main__':
    theme = gr.themes.Soft(primary_hue="purple", secondary_hue="indigo")
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=CUSTOM_CSS
    )
