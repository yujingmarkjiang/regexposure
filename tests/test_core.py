"""Offline unit tests: no network, no API key needed.  Run: pytest -q"""
import numpy as np

from regexposure import extract, llm, prompts
from regexposure.textutil import html_to_lines, token_chunks


def test_html_to_lines_keeps_inline_spans_together():
    html = "<html><body><div><span>ITEM</span> <span>1A.</span> <b>RISK FACTORS</b></div><p>Text&nbsp;here.</p></body></html>"
    assert html_to_lines(html) == ["ITEM 1A. RISK FACTORS", "Text here."]


def test_token_chunks_cover_text():
    text = " ".join(f"w{i}" for i in range(5000))
    parts = token_chunks(text, 1000, 0, "cl100k_base")
    assert len(parts) > 1 and "".join(parts) == text


def test_score_schema_and_instructions():
    labels, desc = [f"L{i}" for i in range(20)], [f"D{i}" for i in range(20)]
    s = prompts.score_schema(20)["properties"]["scores"]
    assert s["required"] == [f"d{i:02d}" for i in range(1, 21)]
    ins = prompts.score_instructions(labels, desc)
    for phrase in ["0       no connection", "81-100  central to the document", "the score\nis 0",
                   "not how\noften a word appears", "issuing agency's name alone is not evidence"]:
        assert phrase in ins


def test_exposure_product():
    firm, rule = np.array([90, 10, 0]), np.array([90, 90, 90])
    assert list(firm * rule / 100) == [81, 9, 0]


def test_output_text_parsing():
    body = {"status": "completed", "output": [{"type": "reasoning"},
            {"type": "message", "content": [{"type": "output_text", "text": '{"a": 1}'}]}]}
    assert llm.output_text(body) == '{"a": 1}'


def test_extract_parse_rejects_bad_range():
    ok = {"item_1": {"status": "present", "start_line": 5, "end_line": 9},
          "item_1a": {"status": "absent", "start_line": -1, "end_line": -1}, "boilerplate": []}
    assert extract.parse(ok) == ok
    bad = {**ok, "item_1": {"status": "present", "start_line": 9, "end_line": 5}}
    try:
        extract.parse(bad)
        raise AssertionError("should fail")
    except ValueError:
        pass
