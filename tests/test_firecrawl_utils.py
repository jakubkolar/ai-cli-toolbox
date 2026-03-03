"""Tests for firecrawl_utils module."""

from ai_cli_toolbox.firecrawl_utils import (
    _escape_yaml_double_quoted,
    _format_markdown_output,
)


class TestFormatMarkdownOutput:
    def test_none_title_uses_fallback(self):
        # When
        result = _format_markdown_output(content="text", title=None, url="https://example.com")  # type: ignore[arg-type]  # testing None guard

        # Then
        assert 'title: ""' in result
        assert 'url: "https://example.com"' in result

    def test_none_url_uses_fallback(self):
        # When
        result = _format_markdown_output(content="text", title="Title", url=None)  # type: ignore[arg-type]  # testing None guard

        # Then
        assert 'url: ""' in result

    def test_valid_inputs_returns_frontmatter(self):
        # When
        result = _format_markdown_output(content="hello", title="My Page", url="https://example.com")

        # Then
        assert result.startswith("---\n")
        assert 'title: "My Page"' in result
        assert 'url: "https://example.com"' in result
        assert result.endswith("hello")


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
