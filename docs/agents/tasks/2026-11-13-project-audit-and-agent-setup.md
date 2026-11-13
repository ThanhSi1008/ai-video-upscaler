# Handoff: Đánh giá tổng quát dự án, thiết lập hạ tầng AI Agents & đề xuất tối ưu hóa

## Yêu cầu và phạm vi
- Ngày/múi giờ: 2026-11-13, Asia/Bangkok (+07:00).
- Nguồn yêu cầu và mục tiêu: Người dùng yêu cầu (1) đánh giá tổng quát toàn bộ project hiện tại, (2) thiết lập sẵn các cài đặt, tùy chỉnh, extension, plugin, file hỗ trợ để tiếp tục build project bằng AI Agents (Antigravity, Codex, Claude Code, VS Code), và (3) đưa ra các phương án tối ưu hóa toàn diện cho project.
- Kết quả cần bàn giao:
  1. Hoàn thiện & tích hợp sạch các thay đổi bộ nhớ Agent còn dở từ `2026-10-31-agent-memory-completion.md` + bổ sung freshness check.
  2. Thiết lập đầy đủ hạ tầng AI Agent trong repo (`.agents/` gồm skills chuyên biệt, lifecycle hooks bảo vệ an toàn GPU/API/checkpoint, VS Code/Antigravity IDE settings & recommended extensions, `.editorconfig`, Codex/Claude/Gemini interoperability, Git pre-commit hook).
  3. Báo cáo đánh giá tổng quát kiến trúc + danh mục phương án tối ưu hóa Pareto chi tiết (kèm file, dòng code, ước lượng ROI và độ rủi ro).
- Được phép / chưa được phép:
  - Được phép: Đọc mã nguồn, thiết lập cấu hình/skills/hooks/extensions/docs/tooling cho AI Agents, chạy kiểm thử cô lập offline (`verify_offline.py`, `pytest` trên test tooling độc lập), commit Git.
  - Chưa được phép: Thay đổi model SR mặc định/tham số NVENC, xóa `.temp_*`, gọi Gemini API thật (đặc biệt khung giờ bảo vệ quota), chạy tác vụ GPU dài (>100 frames).
- Ràng buộc quan trọng: 100% cục bộ trên Windows 11 (i5-12600K + RTX 3060 12GB), ngân sách `AGENTS.md <= 12 KiB`, tổng `AGENTS.md + subtrans/AGENTS.md <= 32 KiB`, không đọc/in `.env`.
- Trạng thái: hoàn tất.

## Mốc mã nguồn và điều phối
- Root, branch, baseline commit: `e:\ai-video-upscaler`, branch `main`, HEAD `0d6e3b6`.
- File đang sửa từ phiên trước (bảo toàn & hoàn thiện): `CLAUDE.md`, `GEMINI.md`, `docs/agents/START_HERE.md`, `docs/agents/VALIDATION.md`, `docs/agents/tasks/TEMPLATE.md`, `scripts/agent_context.py`, `subtrans/AGENTS.md`, `tests/test_agent_context.py`, `docs/agents/tasks/2026-10-31-agent-memory-completion.md`.
- Subagents đang chạy (Read-Only):
  - `Video Pipeline Code Auditor` (`84c290d9-2d3a-44c3-ad8c-8a546510498c`): Kiểm toán `upscale.py`, `src/`, `models/`, `scripts/`.
  - `Subtrans Engine Auditor` (`a0efaee9-1b05-4a4a-b52e-7029252aa357`): Kiểm toán `subtrans.py`, `subtrans/`, `glossaries/`, `subtrans/eval/`.
- File dữ liệu production cần giữ nguyên: `.env`, `pending_subs.json`, `out/`, `logs/`, `subtrans/.cache/`, `.temp_*`, weights.

## Đã làm và bằng chứng
- Đã hoàn thiện bộ nhớ phân tầng từ phiên `2026-10-31-agent-memory-completion.md`, sửa lỗi `.\.venv` tại `START_HERE.md:9`, bổ sung `check_freshness()` vào `scripts/agent_context.py`, và cho phép `verify_offline.py` snapshot `.agents/`.
- Đã thiết lập 4 Workspace Skills (`.agents/skills/`), 2 Workspace Plugins (`upscaler-guard` lifecycle hooks & `upscaler-mcp` read-only MCP), `.vscode/` (`settings.json`, `extensions.json`, `tasks.json`, `launch.json`), `.editorconfig`, `ruff.toml`, `.env.example`.
- Đã kiểm toán toàn diện mã nguồn qua 2 subagent Read-Only và phát hiện thêm nguyên nhân `test_d10_b_output_resolution_tensor` chạy `.half()` trên CPU x86 gây nghẽn `verify_offline.py`, cùng 2 lỗi `ruff` (`F823 hashlib` tại `subtrans/pipeline.py:367` và `F821 thermal_states` tại `src/night_report.py:100`).
- Kiểm chứng: `.\.venv\Scripts\python.exe -m pytest tests/test_agent_context.py tests/test_agent_setup.py tests/test_verify_offline.py -q` đạt **41 passed, 1 skipped (1.37s)**; `agent_context.py --check` đạt **PASS (0 lỗi)**.
- Mux/frame/fps: **N/A — không tạo thành phẩm video**.

## Bước tiếp theo
- Chuyển sang hồ sơ tác vụ tiếp theo để thực thi trực tiếp các gói tối ưu hóa Pareto trên codebase sản xuất theo yêu cầu mới của người dùng.

