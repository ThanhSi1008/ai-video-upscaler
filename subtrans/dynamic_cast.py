# -*- coding: utf-8 -*-
"""
subtrans/dynamic_cast.py
Kiến trúc Quản lý Quan hệ Nhân vật Động Trễ (Hysteresis Dynamic Cast Manager):
- Phân tầng quan hệ: Quan hệ gốc (Baseline) vs. Thay đổi nhất thời (Situational Override)
  vs. Bước ngoặt cốt truyện vĩnh viễn (Permanent Arc Shift).
- Ngăn chặn bẫy "Naive Auto-Update Trap": Khi một nhân vật nổi cáu hoặc nói đùa nhất thời
  trong 1 cảnh (SITUATIONAL_SHIFT: 'temee/kisama' -> mày/tao), hệ thống KHÔNG ghi đè bảng
  xưng hô gốc cho toàn bộ phần còn lại của tập phim.
- Tự động bổ sung nhân vật mới xuất hiện (NEW_CHARACTER: confidence >= 0.75).
- Khóa chặt bước ngoặt cốt truyện (PERMANENT_ARC_SHIFT: confidence >= 0.90) vào nhánh arc_overrides.
"""

from typing import Dict, Any, List, Optional, Literal, Tuple
from pydantic import BaseModel, Field


class CastRelationDelta(BaseModel):
    scope: Literal["NEW_CHARACTER", "SITUATIONAL_SHIFT", "PERMANENT_ARC_SHIFT"]
    speaker: str
    listener: str
    pronoun_1p: str = Field(description="Đại từ ngôi 1 tiếng Việt (VD: anh, em, ta, tớ, tôi)")
    pronoun_2p: str = Field(description="Đại từ ngôi 2 tiếng Việt (VD: em, anh, ngươi, cậu, bạn)")
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class HysteresisCastManager:
    """
    Trình quản lý Cast Sheet phân tầng có trễ (Hysteresis):
    - relations: Bộ xưng hô baseline
    - arc_overrides: Bước ngoặt cốt truyện đã xác nhận
    - situational_history: Nhật ký ngữ cảnh theo cảnh (chỉ dùng tạm thời)
    """
    def __init__(self, initial_cast_data: Optional[Dict[str, Any]] = None):
        data = initial_cast_data or {}
        self.relations: Dict[str, Dict[str, Any]] = dict(data.get("relations", {}))
        self.arc_overrides: Dict[str, Dict[str, Any]] = dict(data.get("arc_overrides", {}))
        self.situational_log: List[Dict[str, Any]] = []

    def apply_deltas(self, deltas: List[CastRelationDelta]) -> Dict[str, int]:
        """
        Cập nhật Cast Sheet có chọn lọc dựa trên cấp độ scope và ngưỡng tin cậy.
        Returns:
            Thống kê số lượng cập nhật theo từng cấp độ.
        """
        stats = {"new_characters": 0, "situational_shifts": 0, "permanent_shifts": 0}

        for delta in deltas:
            pair_key = f"{delta.speaker.strip()} -> {delta.listener.strip()}"
            if delta.scope == "NEW_CHARACTER" and delta.confidence >= 0.75:
                self.relations[pair_key] = {
                    "1p": delta.pronoun_1p.strip(),
                    "2p": delta.pronoun_2p.strip(),
                    "reason": delta.reason,
                    "confidence": delta.confidence
                }
                stats["new_characters"] += 1

            elif delta.scope == "PERMANENT_ARC_SHIFT" and delta.confidence >= 0.90:
                self.arc_overrides[pair_key] = {
                    "1p": delta.pronoun_1p.strip(),
                    "2p": delta.pronoun_2p.strip(),
                    "reason": delta.reason,
                    "confidence": delta.confidence
                }
                stats["permanent_shifts"] += 1

            elif delta.scope == "SITUATIONAL_SHIFT":
                # Chỉ lưu vào nhật ký cảnh, TUYỆT ĐỐI không sửa relations toàn cục!
                self.situational_log.append({
                    "pair": pair_key,
                    "1p": delta.pronoun_1p,
                    "2p": delta.pronoun_2p,
                    "reason": delta.reason
                })
                stats["situational_shifts"] += 1

        return stats

    def resolve_pronouns(self, speaker: str, listener: str) -> Optional[Tuple[str, str]]:
        """
        Tra cứu đại từ xưng hô theo thứ tự ưu tiên:
        1. Arc Overrides (Bước ngoặt đã xác nhận)
        2. Relations Baseline (Quy tắc gốc)
        """
        pair_key = f"{speaker.strip()} -> {listener.strip()}"
        if pair_key in self.arc_overrides:
            rec = self.arc_overrides[pair_key]
            return rec.get("1p", ""), rec.get("2p", "")
        if pair_key in self.relations:
            rec = self.relations[pair_key]
            return rec.get("1p", ""), rec.get("2p", "")
        return None

    def to_prompt_context(self) -> str:
        """
        Sinh chuỗi ngữ cảnh quan hệ nhân vật để đưa vào prompt Gemini API.
        Ưu tiên arc_overrides lên đầu, sau đó là relations baseline.
        """
        lines = []
        if self.arc_overrides:
            lines.append("XƯNG HÔ BƯỚC NGOẶT (ARC OVERRIDES - BẮT BUỘC TUÂN THỦ):")
            for pair, rel in self.arc_overrides.items():
                lines.append(f"  • {pair}: {rel.get('1p', '')}/{rel.get('2p', '')} (Lý do: {rel.get('reason', '')})")
        if self.relations:
            lines.append("XƯNG HÔ CHUẨN CƠ BẢN (BASELINE RELATIONS):")
            for pair, rel in self.relations.items():
                if pair not in self.arc_overrides:
                    lines.append(f"  • {pair}: {rel.get('1p', '')}/{rel.get('2p', '')}")
        return "\n".join(lines)

    @classmethod
    def from_glossary(cls, glossary: Any) -> "HysteresisCastManager":
        """Khởi tạo Cast Manager từ đối tượng Glossary nếu có."""
        relations = {}
        if glossary and hasattr(glossary, "characters"):
            for char in glossary.characters:
                c_name = getattr(char, "name_vi", None) or getattr(char, "name_ja", "")
                for st in getattr(char, "speaks_to", []):
                    pair = f"{c_name} -> {st.listener}"
                    relations[pair] = {
                        "1p": st.self,
                        "2p": st.other,
                        "reason": getattr(st, "note", "") or "Glossary baseline",
                        "confidence": 1.0
                    }
        return cls({"relations": relations})

    def export_state(self) -> Dict[str, Any]:
        """Xuất toàn bộ trạng thái quan hệ nhân vật hiện hành."""
        return {
            "relations": self.relations,
            "arc_overrides": self.arc_overrides,
            "situational_shifts_count": len(self.situational_log)
        }

