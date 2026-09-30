# 🔍 Tool Kiểm Tra Chất Lượng Video: 1080p Gốc hay Fake 720p (Descaling Analysis Tool)

Một công cụ chuyên dụng, **hoàn toàn độc lập** (không phụ thuộc và không can thiệp vào tool AI Upscale hiện tại), dùng để kiểm định chất lượng thực sự của video hoặc hình ảnh xem có phải là **Full HD 1080p thực sự (True Native)**, hay chỉ là **720p bị phóng to (Fake 1080p)**, hoặc bản vẽ Anime gốc của Studio ở **810p/878p**.

---

## 1. 📐 Nguyên Lý Toán Học (Descaling Inverse Error Analysis)

Khi một video $720\text{p}$ bị phóng to lên $1080\text{p}$ bằng các thuật toán nội suy (Bicubic / Bilinear / Lanczos):
- Video đó vốn **không có chi tiết tần số cao thực sự** của chuẩn $1080\text{p}$.
- Khi ta thu nhỏ ($downscale$) khung hình về đúng độ phân giải gốc $720\text{p}$ rồi phóng to ($upscale$) ngược lại bằng cùng thuật toán nội suy, sai số sai khác điểm ảnh (**MSE - Mean Squared Error**) giữa khung hình tái tạo và khung hình ban đầu sẽ **chạm đáy cục bộ (Local Minimum / Valley)** rơi sâu về gần $0$.
- Ngược lại, nếu là **$1080\text{p}$ thực sự**, việc hạ độ phân giải về bất kỳ mốc nào (kể cả $720\text{p}$, $810\text{p}$) cũng sẽ làm mất vĩnh viễn chi tiết tần số cao, khiến đồ thị sai số **tăng dốc đều (đơn điệu)** khi độ phân giải giảm dần.

---

## 2. ⚡ Cài Đặt Môi Trường

Tool sử dụng các thư viện xử lý ảnh tiêu chuẩn, nhẹ và chạy cực nhanh trên mọi máy tính (CPU / Apple Silicon / Windows / Linux):

```bash
pip3 install opencv-python numpy matplotlib
```
*(Đã được cấu hình và cài đặt sẵn sàng trên hệ thống của bạn).*

---

## 3. 🚀 Các Cách Sử Dụng Tool

### Cách 1: Quét trực tiếp tệp Video (Khuyên dùng - Tiện lợi nhất)
Bạn **không cần** phải gõ lệnh ffmpeg trích xuất ảnh thủ công. Tool sẽ tự động quét qua các mốc thời gian của video, đo độ sắc nét (toán tử Laplacian), bỏ qua cảnh đen/mờ/motion blur và chọn khung hình có chi tiết cao nhất:

```bash
python3 check_native.py my_video.mp4
```

### Cách 2: Kiểm tra video tại một cảnh cụ thể (Chỉ định thời gian)
Nếu bạn muốn soi đúng một cảnh tĩnh nhiều chi tiết ở phút thứ 4 giây 30:

```bash
python3 check_native.py "[SubsPlease] Mushoku Tensei S3 - 14 (1080p).mkv" --time 00:04:30
# hoặc truyền số giây:
python3 check_native.py my_video.mp4 -t 270
```

### Cách 3: Quét nhiều khung hình và lấy trung bình (Tăng độ tin cậy)
Tự động lấy top 3 khung hình sắc nét nhất video và tính trung bình đường cong sai số:

```bash
python3 check_native.py my_video.mp4 --multi-frame
```

### Cách 4: Kiểm tra tệp hình ảnh tĩnh (.png, .jpg)
Nếu bạn đã có sẵn ảnh mẫu:

```bash
python3 check_native.py sample.png
```

### Cách 5: Khởi chạy Giao Diện WebUI Độc Lập Trực Quan (Kéo - Thả)
Tool tích hợp sẵn Web Server nhẹ bằng Python Standard Library (không cần cài thêm Gradio), mở trực tiếp trên trình duyệt:

```bash
python3 check_native.py --web
```
👉 Truy cập ngay tại: **`http://localhost:7865`** để kéo thả video, chọn cảnh và xem đồ thị trực quan.

### Cách 6: Xuất kết quả dạng JSON (Dành cho Scripting / Tự động hóa)
```bash
python3 check_native.py my_video.mp4 --json
```

---

## 4. 📊 Đọc & Diễn Giải Kết Quả

Sau khi chạy xong, tool sẽ in báo cáo chi tiết ra Terminal và xuất biểu đồ `result_analysis.png`:

| Kết quả | Dấu hiệu trên đồ thị MSE | Ý nghĩa thực tế |
| :--- | :--- | :--- |
| **❌ FAKE 1080p** *(Kéo từ 720p)* | Đồ thị tụt sâu tạo thành **"thung lũng" (đáy võng)** rõ rệt ngay tại mốc **$720\text{p}$**. Biểu đồ $\Delta MSE$ bên dưới có cột màu đỏ $> 0$. | Video chất lượng thấp bị trang web phim / uploader kéo giãn lên 1080p để câu view. |
| **✅ 1080p GỐC THỰC SỰ** *(True Native)* | Đồ thị là đường dốc **giảm liên tục và chạm mức 0 duy nhất tại $1080\text{p}$**, không có bất kỳ thung lũng nào ở $720\text{p}$ hay $810\text{p}$. | Bản Master xịn, độ nét chuẩn Full HD bảo toàn đầy đủ hạt chi tiết và nét vẽ mịn. |
| **🎨 ANIME STUDIO MASTER** *(~810p hoặc ~878p)* | Đáy võng nằm tại **$810\text{p}$** hoặc **$878\text{p}$**. | Hoạt hình Anime vẽ tay và composite gốc tại độ phân giải của Studio (như KyoAni, CloverWorks ở ~810p; MAPPA, Wit Studio ở ~878p) rồi studio upscale lên 1080p TV/Blu-ray. Đây là bản gốc chính quy của nhà sản xuất. |

---

## 5. 🛠️ Danh Sách Tham Số Đầy Đủ

```
usage: check_native.py [-h] [-t TIME] [-c CROP] [-o OUTPUT] [-m] [-j] [--web] [--port PORT] [input]

options:
  input                Đường dẫn đến tệp video (.mp4, .mkv...) hoặc ảnh (.png, .jpg)
  -t, --time TIME      Mốc thời gian trích xuất từ video (ví dụ: 00:04:30 hoặc 270)
  -c, --crop CROP      Kích thước vùng cắt trung tâm (mặc định: 600px)
  -o, --output OUTPUT  Tên tệp đồ thị PNG xuất ra (mặc định: result_analysis.png)
  -m, --multi-frame    Quét trung bình 3 khung hình sắc nét nhất trong video
  -j, --json           Xuất kết quả dưới định dạng JSON
  --web                Khởi chạy giao diện WebUI trực quan độc lập
  --port PORT          Cổng mạng cho WebUI (mặc định: 7865)
```
