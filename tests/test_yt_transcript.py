"""Tests for yt_transcript module."""

from ai_cli_toolbox.yt_transcript import (
    MetadataResult,
    _extract_video_id,
    _format_duration,
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


class TestFormatDuration:
    def test_seconds_only(self):
        assert _format_duration(45) == "45 sec"

    def test_minutes_and_seconds(self):
        assert _format_duration(310) == "5 min 10 sec"

    def test_exactly_one_minute(self):
        assert _format_duration(60) == "1 min 0 sec"

    def test_hrs_minutes_seconds(self):
        assert _format_duration(3665) == "1 hr 1 min 5 sec"

    def test_multiple_hrs(self):
        assert _format_duration(7384) == "2 hr 3 min 4 sec"


class TestSlugifyTitle:
    def test_simple_title(self):
        # When
        filename = _slugify_title(
            "Never Gonna Give You Up",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )

        # Then
        assert filename == "Never_Gonna_Give_You_Up_75170f.md"

    def test_title_with_special_characters(self):
        # When
        filename = _slugify_title(
            "Python 3.13: What's New & Cool!",
            "https://www.youtube.com/watch?v=abc123",
        )

        # Then
        assert filename == "Python_3_13_What_s_New_Cool_4da372.md"

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
        # When
        filename = _slugify_title(
            "---",
            "https://www.youtube.com/watch?v=test123",
        )

        # Then
        assert filename == "video_750e1b.md"


class TestFormatOutput:
    def test_format_with_transcript(self):
        # Given
        metadata = MetadataResult(
            title="Test Video",
            channel="Test Channel",
            description="Test description",
            upload_date="2026-01-15",
            url="https://www.youtube.com/watch?v=test123",
            duration=None,
            view_count=None,
            like_count=None,
            comment_count=None,
            channel_follower_count=None,
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
            duration=None,
            view_count=None,
            like_count=None,
            comment_count=None,
            channel_follower_count=None,
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
            duration=None,
            view_count=None,
            like_count=None,
            comment_count=None,
            channel_follower_count=None,
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
            duration=None,
            view_count=None,
            like_count=None,
            comment_count=None,
            channel_follower_count=None,
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
            duration=None,
            view_count=None,
            like_count=None,
            comment_count=None,
            channel_follower_count=None,
        )
        transcript = "Content"

        # When
        output = _format_output(metadata, transcript)

        # Then
        assert 'channel: "Channel with \\"Quotes\\""' in output

    def test_format_includes_optional_numeric_fields(self):
        # Given
        metadata = MetadataResult(
            title="Test Video",
            channel="Test Channel",
            description="Test description",
            upload_date="2026-01-15",
            url="https://www.youtube.com/watch?v=test123",
            duration=3600,
            view_count=1000000,
            like_count=50000,
            comment_count=2000,
            channel_follower_count=100000,
        )
        transcript = "Content"

        # When
        output = _format_output(metadata, transcript)

        # Then
        assert 'duration: "1 hr 0 min 0 sec"' in output
        assert "view_count: 1000000" in output
        assert "like_count: 50000" in output
        assert "comment_count: 2000" in output
        assert "channel_follower_count: 100000" in output

    def test_format_omits_none_numeric_fields(self):
        # Given
        metadata = MetadataResult(
            title="Test Video",
            channel="Test Channel",
            description="Test description",
            upload_date="2026-01-15",
            url="https://www.youtube.com/watch?v=test123",
            duration=310,
            view_count=None,
            like_count=50000,
            comment_count=None,
            channel_follower_count=None,
        )
        transcript = "Content"

        # When
        output = _format_output(metadata, transcript)

        # Then
        assert 'duration: "5 min 10 sec"' in output
        assert "like_count: 50000" in output
        assert "view_count:" not in output
        assert "comment_count:" not in output
        assert "channel_follower_count:" not in output
