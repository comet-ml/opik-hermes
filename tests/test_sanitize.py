"""Payload sanitization helpers: truncation, base64 redaction, read_file, json."""
from __future__ import annotations


def test_truncate_long_string(plugin):
    out = plugin._truncate_text("x" * 100, 10)
    assert isinstance(out, str)
    assert out.startswith("x" * 10)
    assert "truncated" in out


def test_truncate_short_string_unchanged(plugin):
    assert plugin._truncate_text("hello", 100) == "hello"


def test_base64_data_uri_is_redacted_not_truncated(plugin):
    uri = "data:image/png;base64," + "A" * 5000
    assert plugin._is_base64_data_uri(uri)
    out = plugin._truncate_text(uri, 12000)  # under max — still redacted
    assert isinstance(out, dict)
    assert out["type"] == "data_uri"
    assert out["media_type"] == "image/png"
    assert out["omitted"] is True
    assert out["length"] == len(uri)


def test_safe_value_redacts_data_uri_nested(plugin):
    out = plugin._safe_value({"image": "data:image/jpeg;base64," + "B" * 3000})
    assert out["image"]["type"] == "data_uri"


def test_safe_value_caps_depth(plugin):
    deep = {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}}
    out = plugin._safe_value(deep)
    # At depth > 4 the value collapses to a sentinel rather than recursing forever.
    assert "<max-depth>" in repr(out)


def test_safe_value_caps_list_and_dict_width(plugin):
    assert len(plugin._safe_value(list(range(200)))) == 50
    assert len(plugin._safe_value({str(i): i for i in range(200)})) == 50


def test_safe_value_bytes(plugin):
    out = plugin._safe_value(b"abc")
    assert out == {"type": "bytes", "len": 3}


def test_maybe_parse_json_string_object(plugin):
    out = plugin._maybe_parse_json_string('{"k": 1}')
    assert out == {"k": 1}


def test_maybe_parse_json_string_with_trailing_hint(plugin):
    out = plugin._maybe_parse_json_string('{"k": 1} [Hint: do x]')
    assert out["k"] == 1
    assert out["_hint"].startswith("[Hint:")


def test_maybe_parse_non_json_passthrough(plugin):
    assert plugin._maybe_parse_json_string("just text") == "just text"


def test_read_file_payload_detected_and_normalized(plugin):
    payload = {
        "content": "1|first line\n2|second line",
        "total_lines": 2,
        "file_size": 42,
        "is_binary": False,
        "is_image": False,
    }
    assert plugin._looks_like_read_file_payload(payload)
    norm = plugin._normalize_read_file_payload(payload, args={"path": "/x", "offset": 0})
    assert norm["path"] == "/x"
    assert norm["returned_lines"]["count"] == 2
    assert norm["total_lines"] == 2


def test_read_file_base64_content_omitted(plugin):
    payload = {
        "content": "1|x",
        "total_lines": 1,
        "file_size": 9,
        "is_binary": True,
        "is_image": True,
        "base64_content": "Z" * 4000,
    }
    norm = plugin._normalize_read_file_payload(payload)
    assert norm["base64_content"]["omitted"] is True
    assert norm["base64_content"]["length"] == 4000
