"""Tests for yt_transcript module."""

from ai_cli_toolbox.yt_transcript import (
    MetadataResult,
    _extract_video_id,
    _format_output,
    _slugify_title,
)


class TestExtractVideoId:
    def test_youtube_watch_url(self):
        # Given
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        # When
        video_id = _extract_video_id(url)

        # Then
        assert video_id == "dQw4w9WgXcQ"

    def test_youtube_watch_url_without_www(self):
        # Given
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"

        # When
        video_id = _extract_video_id(url)

        # Then
        assert video_id == "dQw4w9WgXcQ"

    def test_youtu_be_short_url(self):
        # Given
        url = "https://youtu.be/dQw4w9WgXcQ"

        # When
        video_id = _extract_video_id(url)

        # Then
        assert video_id == "dQw4w9WgXcQ"

    def test_youtube_embed_url(self):
        # Given
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"

        # When
        video_id = _extract_video_id(url)

        # Then
        assert video_id == "dQw4w9WgXcQ"

    def test_youtube_watch_url_with_extra_params(self):
        # Given
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120"

        # When
        video_id = _extract_video_id(url)

        # Then
        assert video_id == "dQw4w9WgXcQ"

    def test_invalid_url_returns_none(self):
        # Given
        url = "https://example.com/video"

        # When
        video_id = _extract_video_id(url)

        # Then
        assert video_id is None

    def test_youtube_url_without_video_param_returns_none(self):
        # Given
        url = "https://www.youtube.com/watch"

        # When
        video_id = _extract_video_id(url)

        # Then
        assert video_id is None


class TestSlugifyTitle:
    def test_simple_title(self):
        # Given
        title = "Never Gonna Give You Up"
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        # When
        filename = _slugify_title(title, url)

        # Then
        assert filename.startswith("Never_Gonna_Give_You_Up_")
        assert filename.endswith(".md")
        assert len(filename) == len("Never_Gonna_Give_You_Up_") + 6 + 3  # + hash + .md

    def test_title_with_special_characters(self):
        # Given
        title = "Python 3.13: What's New & Cool!"
        url = "https://www.youtube.com/watch?v=abc123"

        # When
        filename = _slugify_title(title, url)

        # Then
        assert filename.startswith("Python_3_13_What_s_New_Cool_")
        assert filename.endswith(".md")

    def test_long_title_is_truncated(self):
        # Given
        title = "A" * 200
        url = "https://www.youtube.com/watch?v=xyz789"

        # When
        filename = _slugify_title(title, url, max_length=50)

        # Then
        assert len(filename) <= 50
        assert filename.endswith(".md")
        assert "_" in filename  # Contains hash separator

    def test_empty_title_uses_fallback(self):
        # Given
        title = "---"
        url = "https://www.youtube.com/watch?v=test123"

        # When
        filename = _slugify_title(title, url)

        # Then
        assert filename.startswith("video_")
        assert filename.endswith(".md")


class TestFormatOutput:
    def test_format_with_transcript(self):
        # Given
        metadata = MetadataResult(
            title="Test Video",
            channel="Test Channel",
            description="Test description",
            upload_date="2026-01-15",
            url="https://www.youtube.com/watch?v=test123",
        )
        transcript = "Hello, this is the transcript."

        # When
        output = _format_output(metadata, transcript)

        # Then
        assert 'title: "Test Video"' in output
        assert 'url: "https://www.youtube.com/watch?v=test123"' in output
        assert 'channel: "Test Channel"' in output
        assert 'upload_date: "2026-01-15"' in output
        assert "retrieved_at:" in output
        assert "Hello, this is the transcript." in output
        assert output.startswith("---\n")

    def test_format_without_transcript_shows_warning(self):
        # Given
        metadata = MetadataResult(
            title="Test Video",
            channel="Test Channel",
            description="Test description",
            upload_date="2026-01-15",
            url="https://www.youtube.com/watch?v=test123",
        )
        transcript = None

        # When
        output = _format_output(metadata, transcript)

        # Then
        assert "> **Warning**: Transcript unavailable for this video" in output

    def test_format_escapes_quotes_in_title(self):
        # Given
        metadata = MetadataResult(
            title='He said "Hello"',
            channel="Test Channel",
            description="Test description",
            upload_date="2026-01-15",
            url="https://www.youtube.com/watch?v=test123",
        )
        transcript = "Content"

        # When
        output = _format_output(metadata, transcript)

        # Then
        assert 'title: "He said \\"Hello\\""' in output

    def test_format_multiline_description(self):
        # Given
        metadata = MetadataResult(
            title="Test Video",
            channel="Test Channel",
            description="Line 1\nLine 2\nLine 3",
            upload_date="2026-01-15",
            url="https://www.youtube.com/watch?v=test123",
        )
        transcript = "Content"

        # When
        output = _format_output(metadata, transcript)

        # Then
        assert "description: |\n" in output
        assert "  Line 1\n" in output
        assert "  Line 2\n" in output
        assert "  Line 3\n" in output

    def test_format_escapes_quotes_in_channel(self):
        # Given
        metadata = MetadataResult(
            title="Test Video",
            channel='Channel with "Quotes"',
            description="Test description",
            upload_date="2026-01-15",
            url="https://www.youtube.com/watch?v=test123",
        )
        transcript = "Content"

        # When
        output = _format_output(metadata, transcript)

        # Then
        assert 'channel: "Channel with \\"Quotes\\""' in output
