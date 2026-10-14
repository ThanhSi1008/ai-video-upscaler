# 📊 Benchmarks & Evaluations

Thư mục chứa toàn bộ các kịch bản đo lường hiệu năng (Performance Benchmarks), kiểm thử độ bền (Soak Tests), kiểm thử độ ổn định đại từ (Pronoun Stability), và đánh giá chất lượng thị giác (Visual Fidelity / Blind AB).

## 📁 Cấu trúc thư mục

### 1. `benchmarks/upscale/`
Các kịch bản đánh giá hệ thống AI Video Upscaling:
- `benchmark_3way_blind.py`: Đánh giá mù 3 mô hình (RealESR-AnimeVideoV3, AnimeJaNai, ArtCNN).
- `benchmark_models_comparison.py`: So sánh tốc độ và VRAM giữa các mô hình.
- `benchmark_mps_vs_ncnn.py`: So sánh hiệu năng giữa PyTorch MPS và ncnn Vulkan.
- `benchmark_backpressure.py`: Kiểm thử cơ chế điều tiết bộ nhớ đệm backpressure.
- `profile_tail_latency.py`: Đo đạc độ trễ đuôi (P95, P99 tail latency) của từng frame.
- `soak_end_to_end_mps.py` & `soak_test_animejanai_2000.py`: Kiểm thử độ bền 2.000–6.000 frames.
- `audit_dedup_full.py` & `audit_small_block_dedup.py`: Đo tỷ lệ nén và bắt chuyển động của Deduplication.
- `chaos_test_resume.py`: Kiểm thử khả năng phục hồi khi xảy ra ngắt quãng tiến trình.
- `ab_test/`: Dữ liệu kiểm tra thị giác độc lập (Double-Blind A/B).

### 2. `benchmarks/subtrans/`
Các kịch bản đánh giá hệ thống dịch thuật phụ đề AI:
- `eval_golden_pronouns.py`: Đánh giá độ nhất quán xưng hô đại từ trên bộ Golden Set (Mushoku Tensei).
- `eval_golden_pronouns_multigenre.py`: Đánh giá xưng hô đa thể loại anime (Fantasy, Action, Drama, Romance).
- `validate_pronouns_regex.py`: Kiểm tra độ chính xác của các biểu thức chính quy nhận diện danh xưng.
