# 📊 Benchmarks & Evaluations

Thư mục chứa toàn bộ các kịch bản đo lường hiệu năng (Performance Benchmarks), kiểm thử độ ổn định đại từ (Pronoun Stability), và đánh giá chất lượng thị giác (Visual Fidelity / Blind AB) tối ưu hóa cho Windows 11 & NVIDIA RTX.

## 📁 Cấu trúc thư mục

### 1. `benchmarks/upscale/`
Các kịch bản đánh giá hệ thống AI Video Upscaling:
- `benchmark_models_comparison.py`: So sánh tốc độ và VRAM giữa các mô hình Super-Resolution trên CUDA.
- `profile_tail_latency.py`: Đo đạc độ trễ đuôi (P95, P99 tail latency) của từng frame trên GPU NVIDIA RTX.
- `audit_local_metrics.py` & `audit_small_block_dedup.py`: Đo tỷ lệ nén và bắt chuyển động của Deduplication.
- `chaos_test_resume.py`: Kiểm thử khả năng phục hồi khi xảy ra ngắt quãng tiến trình.
- `ab_test/`: Dữ liệu kiểm tra thị giác độc lập (Double-Blind A/B).

### 2. `benchmarks/subtrans/`
Các kịch bản đánh giá hệ thống dịch thuật phụ đề AI:
- `eval_golden_pronouns.py`: Đánh giá độ nhất quán xưng hô đại từ trên bộ Golden Set (Mushoku Tensei).
- `eval_golden_pronouns_multigenre.py`: Đánh giá xưng hô đa thể loại anime (Fantasy, Action, Drama, Romance).
- `validate_pronouns_regex.py`: Kiểm tra độ chính xác của các biểu thức chính quy nhận diện danh xưng.

### 3. `benchmarks/research/`
Kịch bản nghiên cứu và hiệu chuẩn thuật toán:
- `calibrate_b1_b2.py`: Hiệu chuẩn ngưỡng phát hiện chuyển động.
- `benchmark_dis_synthetic_b1_b2.py`: Kiểm thử dòng chảy quang học DIS trên dữ liệu tổng hợp.
- `measure_a1_a2_windows.py`: Đo đạc hiệu năng và độ chính xác FP16 trên Windows CUDA.
- `measure_a2_animejanai.py`: Đo đạc đối chiếu sai số độc lập giữa mô hình nội bộ và Spandrel CPU.
- `make_blind_ab.py`: Sinh bài kiểm tra mù A/B cho các mô hình SR.
