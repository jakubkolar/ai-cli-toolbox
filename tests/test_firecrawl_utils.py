"""Tests for firecrawl_utils module."""

from ai_cli_toolbox.firecrawl_utils import (
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
