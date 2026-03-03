"""Tests for firecrawl_utils module."""

import re

from ai_cli_toolbox.firecrawl_utils import (
    _escape_yaml_double_quoted,
    _format_markdown_output,
)

_FRONTMATTER_FIELD = re.compile(r'^(\w+):\s*"(.*)"$')


def _parse_frontmatter(output: str) -> tuple[dict[str, str], str]:
    parts = output.split("---\n", maxsplit=2)
    meta: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        m = _FRONTMATTER_FIELD.match(line)
        if m:
            meta[m.group(1)] = m.group(2)
    body = parts[2].lstrip("\n")
    return meta, body


class TestFormatMarkdownOutput:
    def test_none_title_uses_fallback(self):
        # When
        result = _format_markdown_output(content="text", title=None, url="https://example.com")  # type: ignore[arg-type]  # testing None guard

        # Then
        meta, body = _parse_frontmatter(result)
        assert meta["title"] == ""  # noqa: PLC1901  # verifying exact empty-string fallback, not truthiness
        assert meta["url"] == "https://example.com"
        assert body == "text"

    def test_none_url_uses_fallback(self):
        # When
        result = _format_markdown_output(content="text", title="Title", url=None)  # type: ignore[arg-type]  # testing None guard

        # Then
        meta, body = _parse_frontmatter(result)
        assert meta["url"] == ""  # noqa: PLC1901  # verifying exact empty-string fallback, not truthiness
        assert meta["title"] == "Title"
        assert body == "text"

    def test_valid_inputs_returns_frontmatter(self):
        # When
        result = _format_markdown_output(content="hello", title="My Page", url="https://example.com")

        # Then
        meta, body = _parse_frontmatter(result)
        assert meta["title"] == "My Page"
        assert meta["url"] == "https://example.com"
        assert "scraped_at" in meta
        assert body == "hello"


class TestEscapeYamlDoubleQuoted:
    def test_empty_string(self):
        # When
        result = _escape_yaml_double_quoted("")

        # Then
        assert result == ""  # noqa: PLC1901  # verifying exact return value, not truthiness

    def test_string_with_double_quotes(self):
        # When
        result = _escape_yaml_double_quoted('He said "hello"')

        # Then
        assert result == 'He said \\"hello\\"'

    def test_string_with_backslashes(self):
        # When
        result = _escape_yaml_double_quoted("path\\to\\file")

        # Then
        assert result == "path\\\\to\\\\file"

    def test_string_with_newlines(self):
        # When
        result = _escape_yaml_double_quoted("line one\nline two")

        # Then
        assert result == "line one line two"
