from __future__ import annotations

import pytest

from viemmo.data.pii import find_pii, record_pii
from viemmo.data.schema import SCHEMA_VERSION, SOURCE, SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Liên hệ nguoi.dung@mien.vn để xác nhận.", "email"),
        ("Số liên hệ là +84 912 345 678.", "phone"),
        ("Mã căn cước là 079203001234.", "citizen_id"),
        ("Thẻ thử bị lộ: 4111 1111 1111 1111.", "payment_card"),
        ("api_key=abcdefgh12345678", "credential"),
        ("Nơi nhận: số 12 đường Hoa Mai.", "address"),
    ],
)
def test_each_pii_detector(text: str, kind: str) -> None:
    assert kind in {finding.kind for finding in find_pii(text)}


def test_false_positives_placeholders_and_resolved_record() -> None:
    assert find_pii("Có 12 bản ghi và 34 cột dữ liệu.") == []
    assert find_pii("Dùng [EMAIL], <phone> và test@example.com trong tài liệu.") == []

    record = {
        "schema_version": SCHEMA_VERSION,
        "id": "pilot-privacy-001-01",
        "category": "privacy_security",
        "group_id": "pilot-privacy-001",
        "source": SOURCE.copy(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Gửi tới [EMAIL]."},
            {"role": "assistant", "content": "Đã dùng dữ liệu giữ chỗ, không dùng địa chỉ thật."},
        ],
    }
    assert record_pii(record) == []
    record["messages"][1]["content"] = "Gửi tới user@private.vn."
    assert record_pii(record)[0]["kind"] == "email"
    record["messages"][1]["content"] = "Gửi tới [EMAIL]."
    assert record_pii(record) == []
