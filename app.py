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

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(repo_root)

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
    "⚡ NGUỒN B: AnimeJaNai V3 Compact (Khuyên dùng WEB-DL Gốc: SubsPlease/Erai - Siêu tốc ~25–40 phút/tập)": "animejanai_v3_compact",
    "⚡ NGUỒN A: AnimeJaNai V3 Sharp (Khuyên dùng BDRip 10-bit: Hi10P/Main 10 - Nét đanh giữ grain, siêu tốc ~25–40 phút/tập)": "animejanai_v3_sharp",
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

def process_ui(drive_or_path, video_file, model_choice, progress=gr.Progress(track_tqdm=True)):
    target_input = None
    cleaned_input = drive_or_path.strip() if drive_or_path else ""

    # Chặn link Magnet/P2P để bảo vệ an toàn tài khoản Colab/Kaggle
    if cleaned_input.startswith("magnet:"):
        raise gr.Error("❌ Hệ thống không hỗ trợ Magnet/Torrent nhằm tuân thủ điều khoản chống P2P của Google Colab (tránh bị khoá tài khoản). Vui lòng lưu video vào Google Drive hoặc tải tệp trực tiếp!")

    # Nếu người dùng chỉ để nguyên tiền tố mặc định mà không điền tên file
    if cleaned_input in ["/content/drive/MyDrive/Resources", "/content/drive/MyDrive/Resources/"]:
        if video_file is None:
            raise gr.Error("❌ Bạn chưa điền tên file anime sau đường dẫn! Ví dụ: /content/drive/MyDrive/Resources/Mushoku_Tensei_S02E14.mkv")
        target_input = video_file
    elif cleaned_input:
        # Nếu người dùng chỉ gõ tên file mà quên tiền tố (ví dụ: Mushoku_Tensei_14.mkv)
        if not cleaned_input.startswith("/") and not cleaned_input.startswith("http"):
            target_input = f"/content/drive/MyDrive/Resources/{cleaned_input}"
        else:
            target_input = cleaned_input
    elif video_file is not None:
        target_input = video_file
    else:
        raise gr.Error("❌ Vui lòng điền tên file trong thư mục /content/drive/MyDrive/Resources/ HOẶC dán link Google Drive!")

    model_name = MODEL_MAP.get(model_choice, "animejanai_v3_compact")
    progress_queue = Queue()

    def progress_cb(pct, desc=""):
        progress_queue.put((pct, desc))
        if pct is not None:
            progress(pct, desc=desc)

    yield None, gr.update(visible=False), f"⏳ Đang khởi tạo luồng giải mã Native 2x (4K HEVC 10-bit & Checkpoints an toàn)..."

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

with gr.Blocks(title="AI Video Upscaler 4K - Anime Native 2x UHD", theme=gr.themes.Default(), css=CUSTOM_CSS) as app:
    with gr.Column(elem_classes=["container"]):
        with gr.Group(elem_classes=["header-box"]):
            gr.Markdown(f"""
            # 🎬 AI Video Upscaler 4K - Native 2x Ultra-HD
            Hệ thống chuyên dụng nâng cấp Anime 1080p lên **4K Ultra-HD (3840x2160 Native 2x)**. Khôi phục nét vẽ vector nguyên bản, mã hóa HEVC 10-bit chống banding và bảo tồn 100% Phụ đề mềm (.ass) & Âm thanh gốc.
            Tích hợp cơ chế **Checkpoint Tự Động** chống ngắt quãng session Google Colab Free.
            
            <div class="badge">THIẾT BỊ: {device_badge}</div>
            """)

        with gr.Accordion("📖 Hướng dẫn sử dụng nhanh (Google Colab & Mac)", open=False):
            gr.Markdown("""
            ### 📖 Hướng Dẫn Sử Dụng
            1. **Tập Phim Nguồn**: Điền thêm tên file vào sau đường dẫn `/content/drive/MyDrive/Resources/` (ví dụ: `/content/drive/MyDrive/Resources/Mushoku_Tensei_14.mkv`) HOẶC dán link chia sẻ Google Drive.
            2. **Mô Hình AI**:
               - **NGUỒN B: AnimeJaNai V3 Compact**: Khuyên dùng cho **WEB-DL Gốc** (SubsPlease, Erai-raws, Crunchyroll/Netflix). Tốc độ siêu tốc ~25–40 phút/tập, hoàn thành trong ngân sách 2 giờ.
               - **NGUỒN A: AnimeJaNai V3 Sharp**: Khuyên dùng cho **BDRip 10-bit (Hi10P / Main 10)**. Giữ nét đanh thép, tận dụng dải màu 10-bit đã deband sạch từ các nhóm encode uy tín (VCB-Studio, Beatrice-Raws...) để triệt tiêu hiện tượng banding.
            3. **Bắt Đầu**: Bấm **"🚀 Nâng Cấp Video 4K"**. Tập phim 4K Ultra-HD sẽ được mã hóa và xuất thẳng về thư mục `/content/drive/MyDrive/Upscaled`!
            """)

        # 1. Ô NHẬP LINK GOOGLE DRIVE / ĐƯỜNG DẪN TẬP PHIM
        drive_link_input = gr.Textbox(
            value="/content/drive/MyDrive/Resources/",
            label="☁️ Đường Dẫn File Trong Drive (/content/drive/MyDrive/Resources/...) HOẶC Link Google Drive",
            placeholder="Chỉ cần điền thêm tên file vào sau (ví dụ: Mushoku_Tensei_S02E14.mkv) HOẶC dán link chia sẻ Drive https://drive.google.com/...",
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
                label="✨ Video 4K Kết Quả (Native 2x UHD)",
                interactive=False,
                scale=1
            )

        # 3. THANH TIẾN ĐỘ THỜI GIAN THỰC
        status_box = gr.Textbox(
            label="📊 Tiến Độ & Trạng Thái Thời Gian Thực (Live Progress)",
            value="Chờ dán link Google Drive hoặc chọn tệp anime...",
            interactive=False
        )

        # 4. BẢNG CẤU HÌNH & NÚT BẮT ĐẦU / TẢI VỀ
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group(elem_classes=["panel-box"]):
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_MAP.keys()),
                        value="⚡ NGUỒN B: AnimeJaNai V3 Compact (Khuyên dùng WEB-DL Gốc: SubsPlease/Erai - Siêu tốc ~25–40 phút/tập)",
                        label="🤖 Mô Hình AI (Super-Resolution Native 2x UHD)",
                        info="Mô hình AI siêu phân giải chuyên dụng cho Anime, xử lý Native 4K UHD với tốc độ vượt trội và giữ nguyên 100% chi tiết gốc."
                    )
            with gr.Column(scale=5):
                submit_btn = gr.Button("🚀 Nâng Cấp Video 4K (Native 2x UHD)", variant="primary", size="lg")
                download_file = gr.File(
                    label="📥 Tải tệp 4K kết quả (.mkv đầy đủ Sub & Audio)",
                    visible=False
                )

        submit_btn.click(
            fn=process_ui,
            inputs=[drive_link_input, file_input, model_dropdown],
            outputs=[output_preview, download_file, status_box]
        )

if __name__ == '__main__':
    share_mode = True if ("--share" in sys.argv or "--public" in sys.argv or os.environ.get("GRADIO_SHARE") == "True") else False
    allowed_dirs = ["/kaggle/working", "/tmp", tempfile.gettempdir(), os.getcwd(), os.path.expanduser('~/Movies/Upscaled'), "/content"]

    # Trên Linux/Colab: tự động giải phóng port 7860 nếu có tiến trình zombie cũ chiếm giữ
    if sys.platform.startswith("linux"):
        try:
            import subprocess
            subprocess.run(["fuser", "-k", "7860/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        except Exception:
            pass

    server_port = 7860
    launch_res = None
    try:
        launch_res = app.queue().launch(
            server_name="0.0.0.0",
            server_port=server_port,
            share=share_mode,
            allowed_paths=allowed_dirs,
            prevent_thread_lock=True
        )
    except Exception as e_port:
        print(f"⚠️ Port {server_port} không khả dụng ({e_port}), chuyển sang tự động chọn port trống...")
        launch_res = app.queue().launch(
            server_name="0.0.0.0",
            share=share_mode,
            allowed_paths=allowed_dirs,
            prevent_thread_lock=True
        )

    share_url = None
    if isinstance(launch_res, tuple) and len(launch_res) == 3:
        share_url = launch_res[2]
    if not share_url:
        share_url = getattr(app, "share_url", None)

    if share_url:
        print("\n" + "="*68, flush=True)
        print(f"🌐 GRADIO PUBLIC URL:  {share_url}", flush=True)
        tiny_url = create_tinyurl(share_url)
        if tiny_url:
            print(f"🔗 TINYURL SHORTLINK:   {tiny_url}", flush=True)
            print(f"💡 Dùng ngay link TinyURL trên để mở WebUI!", flush=True)
        print("="*68 + "\n", flush=True)

    try:
        app.block()
    except KeyboardInterrupt:
        print("🛑 Ứng dụng đã dừng.")
