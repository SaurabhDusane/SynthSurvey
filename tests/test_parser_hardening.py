"""Tests for parser hardening: host allowlist (SSRF) and fallback flagging."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from parsers.google_form_parser import GoogleFormParser


class TestHostAllowlist:
    def test_docs_google_allowed(self):
        p = GoogleFormParser("https://docs.google.com/forms/d/e/abc/viewform")
        assert p.url.endswith("/viewform")

    def test_forms_gle_shortlink_left_intact(self):
        p = GoogleFormParser("https://forms.gle/AbCdEf")
        assert p.url == "https://forms.gle/AbCdEf"

    @pytest.mark.parametrize("bad_url", [
        "http://169.254.169.254/latest/meta-data/",
        "https://evil.example.com/forms/d/e/x/viewform",
        "file:///etc/passwd",
        "http://localhost:8080/admin",
    ])
    def test_non_google_hosts_rejected(self, bad_url):
        with pytest.raises(ValueError):
            GoogleFormParser(bad_url)


class TestFallbackFlag:
    def test_default_is_structured(self):
        p = GoogleFormParser("https://docs.google.com/forms/d/e/x/viewform")
        assert p.parse_method == "structured"

    def test_html_fallback_flagged(self):
        p = GoogleFormParser("https://docs.google.com/forms/d/e/x/viewform")
        # No FB_PUBLIC_LOAD_DATA_ present -> forces the HTML DOM fallback.
        p.raw_html = (
            '<html><head><meta property="og:title" content="My Form"></head>'
            '<body><input name="entry.123" type="text"></body></html>'
        )
        schema = p.parse()
        assert p.parse_method == "html_fallback"
        assert schema.parse_method == "html_fallback"
