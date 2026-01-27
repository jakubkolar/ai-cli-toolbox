"""YouTube transcript downloader CLI tool.

Downloads YouTube video transcripts as markdown files with YAML frontmatter.
Uses yt-dlp for metadata extraction and youtube-transcript-api for transcript fetching.
"""

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


@dataclass(frozen=True, slots=True)
class MetadataResult:
    """Video metadata extracted from YouTube."""

    title: str
    description: str
    upload_date: str
    url: str


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Result of processing a single video."""

    success: bool
    path: Path | None
    error: str | None
    skipped: bool = False


def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats.

    Supports:
    - youtube.com/watch?v=VIDEO_ID
    - youtu.be/VIDEO_ID
    - youtube.com/embed/VIDEO_ID

    :param url: YouTube video URL.
    :return: 11-character video ID, or None if URL format not recognized.
    """
    parsed = urlparse(url)

    # youtube.com/watch?v=VIDEO_ID
    if parsed.netloc in {"www.youtube.com", "youtube.com"}:
        if parsed.path == "/watch":
            query_params = parse_qs(parsed.query)
            video_ids = query_params.get("v")
            if video_ids:
                return video_ids[0]
        # youtube.com/embed/VIDEO_ID
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/")[1].split("/")[0]

    # youtu.be/VIDEO_ID
    if parsed.netloc == "youtu.be":
        return parsed.path.lstrip("/").split("/")[0]

    return None


def _slugify_title(title: str, url: str, max_length: int = 100) -> str:
    """Convert video title to safe filename with hash suffix.

    :param title: Video title.
    :param url: Video URL for hash uniqueness.
    :param max_length: Maximum filename length (default 100).
    :return: Safe filename with .md extension.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")

    if not slug:
        slug = "video"

    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]  # noqa: S324
    max_slug_length = max_length - len(url_hash) - 4  # 4 = underscore + .md

    if len(slug) > max_slug_length:
        slug = slug[:max_slug_length]

    return f"{slug}_{url_hash}.md"


def _fetch_metadata(url: str) -> MetadataResult | None:
    """Fetch video metadata using yt-dlp.

    :param url: YouTube video URL.
    :return: MetadataResult with title, description, upload_date, url. None on error.
    """
    ydl_opts = {"quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return None
            info_dict = ydl.sanitize_info(info)
            if info_dict is None:
                return None

            title = info_dict.get("title", "Untitled")
            description = info_dict.get("description", "")
            upload_date_raw = info_dict.get("upload_date", "")

            # Convert YYYYMMDD to YYYY-MM-DD
            upload_date = ""
            if upload_date_raw and len(upload_date_raw) == 8:
                upload_date = f"{upload_date_raw[:4]}-{upload_date_raw[4:6]}-{upload_date_raw[6:8]}"

            return MetadataResult(
                title=title,
                description=description,
                upload_date=upload_date,
                url=url,
            )
    except yt_dlp.utils.DownloadError as e:
        sys.stderr.write(f"Error fetching metadata: {e}\n")
        return None


def _fetch_transcript(video_id: str) -> str | None:
    """Fetch transcript using youtube-transcript-api.

    Prefers manually created transcripts over auto-generated (iteration order).

    :param video_id: YouTube video ID.
    :return: Plain text transcript, or None if unavailable.
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        # Get first available transcript (manually created come before generated)
        transcript = next(iter(transcript_list), None)
        if transcript is None:
            return None

        fetched = transcript.fetch()
        formatter = TextFormatter()
        return formatter.format_transcript(fetched)

    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Error fetching transcript: {e}\n")
        return None


def _format_output(metadata: MetadataResult, transcript: str | None) -> str:
    """Format output as markdown with YAML frontmatter.

    :param metadata: Video metadata.
    :param transcript: Transcript text, or None if unavailable.
    :return: Formatted markdown string.
    """
    retrieved_at = datetime.now(UTC).isoformat()

    # Escape quotes in title for YAML
    title_escaped = metadata.title.replace('"', '\\"')

    # Format description with proper YAML multi-line syntax
    description_lines = metadata.description.split("\n")
    if len(description_lines) > 1 or "\n" in metadata.description:
        description_yaml = "|\n" + "\n".join(f"  {line}" for line in description_lines)
    else:
        description_yaml = f'"{metadata.description}"'

    frontmatter = f'''---
title: "{title_escaped}"
url: "{metadata.url}"
description: {description_yaml}
upload_date: "{metadata.upload_date}"
retrieved_at: "{retrieved_at}"
---

'''

    body = "> **Warning**: Transcript unavailable for this video\n" if transcript is None else transcript

    return frontmatter + body


def _process_video(url: str, output_dir: Path, *, force: bool) -> ProcessResult:
    """Process a single YouTube video.

    :param url: YouTube video URL.
    :param output_dir: Directory to save output file.
    :param force: If True, overwrite existing files.
    :return: ProcessResult with success status, path, and error message.
    """
    # Extract video ID
    video_id = _extract_video_id(url)
    if video_id is None:
        return ProcessResult(
            success=False,
            path=None,
            error=f"Invalid YouTube URL: {url}",
        )

    # Fetch metadata first (needed for filename)
    sys.stderr.write(f"Processing: {url}\n")
    metadata = _fetch_metadata(url)
    if metadata is None:
        return ProcessResult(
            success=False,
            path=None,
            error="Failed to fetch video metadata",
        )

    sys.stderr.write(f"  Title: {metadata.title}\n")

    # Generate filename and check existence
    filename = _slugify_title(metadata.title, url)
    output_path = output_dir / filename

    if output_path.exists() and not force:
        sys.stderr.write(f"  Skipping (file exists): {output_path}\n")
        return ProcessResult(
            success=True,
            path=output_path,
            error=None,
            skipped=True,
        )

    # Fetch transcript
    transcript = _fetch_transcript(video_id)
    transcript_failed = transcript is None

    # Format and write output
    output = _format_output(metadata, transcript)
    output_path.write_text(output)

    if transcript_failed:
        sys.stderr.write("  Warning: Transcript unavailable, created partial file\n")
        sys.stderr.write(f"  Saved to: {output_path}\n")
        return ProcessResult(
            success=False,
            path=output_path,
            error="Transcript unavailable",
        )

    sys.stderr.write(f"  Saved to: {output_path}\n")
    return ProcessResult(
        success=True,
        path=output_path,
        error=None,
    )


def main() -> None:
    """Entry point for yt-transcript command."""
    parser = argparse.ArgumentParser(
        prog="yt-transcript",
        description="Download YouTube video transcripts as markdown files",
    )
    parser.add_argument(
        "urls",
        nargs="+",
        help="YouTube video URLs to process",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path(),
        help="Output directory for transcript files (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all URLs
    successes = 0
    failures = 0
    results: list[ProcessResult] = []

    for url in args.urls:
        result = _process_video(url, output_dir, force=args.force)
        results.append(result)
        if result.success:
            successes += 1
        else:
            failures += 1

    # Print summary if any failures
    if failures > 0:
        sys.stderr.write(f"\nSummary: {successes} succeeded, {failures} failed\n")
        for result in results:
            if not result.success and result.error:
                sys.stderr.write(f"  - {result.error}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
