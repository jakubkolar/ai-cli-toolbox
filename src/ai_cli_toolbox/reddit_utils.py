"""Reddit CLI utilities for fetching posts and comments via the .json endpoint.

Three CLI commands for Reddit content retrieval:
- reddit-scrape: Fetch single Reddit post/thread to XML
- reddit-batch-scrape: Fetch multiple Reddit posts to XML files
- reddit-feed: List posts from a subreddit feed
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree.ElementTree import (  # noqa: S405  # we generate XML, not parse untrusted input
    Element,
    ElementTree,
    SubElement,
    indent,
)

import requests

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REDDIT_DOMAINS = {"reddit.com", "www.reddit.com", "old.reddit.com"}
DEFAULT_MAX_DEPTH = 5

RATE_LIMIT_EPILOG = """
NOTE: Reddit rate limits unauthenticated requests to ~10 per minute.
If you receive a 429 error, wait 1-2 minutes before retrying.
This is a temporary limit, not a permanent failure.
"""


# =============================================================================
# Domain Models
# =============================================================================


@dataclass(frozen=True, slots=True)
class Post:
    """Parsed Reddit post."""

    title: str
    author: str
    score: int
    upvote_ratio: float
    created_at: str
    num_comments: int
    archived: bool
    locked: bool
    selftext: str
    subreddit: str
    url: str


@dataclass(frozen=True, slots=True)
class Comment:
    """Parsed Reddit comment with nested replies."""

    author: str
    score: int
    upvote_ratio: float
    created_at: str
    depth: int
    is_submitter: bool
    distinguished: str | None
    edited: bool
    body: str
    replies: tuple["Comment", ...]
    """Immutable sequence for frozen dataclass."""


@dataclass(frozen=True, slots=True)
class FeedPost:
    """Post summary from subreddit feed (no comments)."""

    title: str
    url: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_at: str
    selftext: str


# =============================================================================
# Exceptions
# =============================================================================


class RedditError(Exception):
    """Base exception for Reddit-related errors."""


# =============================================================================
# URL Handling
# =============================================================================


def _validate_reddit_url(url: str) -> str:
    """Validate URL is a Reddit domain and return normalized URL.

    :param url: URL to validate.
    :return: Normalized URL (trailing slash stripped).
    :raises RedditError: If URL is not a valid Reddit domain.
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if domain not in REDDIT_DOMAINS:
        msg = f"Invalid Reddit URL. Must be reddit.com or old.reddit.com, got: {domain}"
        raise RedditError(msg)

    return url.rstrip("/")


def _make_json_url(url: str) -> str:
    """Append .json suffix to Reddit URL.

    :param url: Normalized Reddit URL (no trailing slash).
    :return: URL with .json suffix.
    """
    return f"{url}.json"


# =============================================================================
# HTTP
# =============================================================================


def _fetch_json(url: str) -> dict[str, Any] | list[Any]:
    """Fetch JSON from Reddit with proper User-Agent header.

    :param url: URL to fetch (should already have .json suffix).
    :return: Parsed JSON response.
    :raises RedditError: On HTTP errors with appropriate messages.
    """
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        msg = f"Network error: {e}"
        raise RedditError(msg) from e

    if response.status_code == 403:
        msg = "Access denied. This should not happen with proper User-Agent."
        raise RedditError(msg)

    if response.status_code == 404:
        msg = "Post not found or may have been deleted"
        raise RedditError(msg)

    if response.status_code == 429:
        sys.stderr.write(
            "Reddit API is rate limited. Wait 1-2 minutes before retrying. This is not a permanent failure.\n"
        )
        msg = "Rate limited (HTTP 429)"
        raise RedditError(msg)

    if not response.ok:
        msg = f"HTTP error {response.status_code}"
        raise RedditError(msg)

    try:
        return response.json()
    except json.JSONDecodeError as e:
        msg = "Failed to parse Reddit response"
        raise RedditError(msg) from e


# =============================================================================
# Utility Functions
# =============================================================================


def _parse_timestamp(unix_ts: float) -> str:
    """Convert Unix timestamp to ISO 8601 UTC string.

    :param unix_ts: Unix timestamp (seconds since epoch).
    :return: ISO 8601 formatted string with Z suffix.
    """
    dt = datetime.fromtimestamp(unix_ts, UTC)
    return dt.isoformat().replace("+00:00", "Z")


def _slugify_url(url: str, max_length: int = 100) -> str:
    """Convert Reddit URL to safe filename with hash suffix.

    Format: {subreddit}_{post_id}_{hash}.xml

    :param url: Reddit URL.
    :param max_length: Maximum filename length (default 100).
    :return: Safe filename with .xml extension.
    """
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]

    subreddit = ""
    post_id = ""

    if len(path_parts) >= 2 and path_parts[0] == "r":
        subreddit = path_parts[1]

    if len(path_parts) >= 4 and path_parts[2] == "comments":
        post_id = path_parts[3]

    if subreddit and post_id:
        base = f"{subreddit}_{post_id}"
    else:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", parsed.path)
        slug = re.sub(r"_+", "_", slug).strip("_")
        base = slug or "reddit"

    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]  # noqa: S324  # not cryptographic, used for filename uniqueness
    max_base_length = max_length - len(url_hash) - 5  # 5 = underscore + .xml

    if len(base) > max_base_length:
        base = base[:max_base_length]

    return f"{base}_{url_hash}.xml"


def _truncate_text(text: str, length: int) -> str:
    """Truncate text with ellipsis for preview.

    :param text: Text to truncate.
    :param length: Maximum length (including ellipsis).
    :return: Truncated text with ... if needed.
    """
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


# =============================================================================
# JSON Parsing
# =============================================================================


def _parse_post(data: dict[str, Any]) -> Post:
    """Extract post fields from Reddit JSON.

    :param data: Post data from Reddit JSON (the "data" field of a t3 thing).
    :return: Post dataclass with essential fields.
    """
    return Post(
        title=data.get("title", ""),
        author=data.get("author", "[deleted]"),
        score=data.get("score", 0),
        upvote_ratio=data.get("upvote_ratio", 0.0),
        created_at=_parse_timestamp(data.get("created_utc", 0)),
        num_comments=data.get("num_comments", 0),
        archived=data.get("archived", False),
        locked=data.get("locked", False),
        selftext=data.get("selftext", ""),
        subreddit=data.get("subreddit", ""),
        url=f"https://www.reddit.com{data.get('permalink', '')}",
    )


def _parse_comment(data: dict[str, Any], max_depth: int, current_depth: int = 0) -> Comment | None:
    """Extract comment fields from Reddit JSON, recursively parsing replies.

    :param data: Comment data from Reddit JSON (the "data" field of a t1 thing).
    :param max_depth: Maximum depth to include.
    :param current_depth: Current recursion depth.
    :return: Comment dataclass, or None if filtered out.
    """
    body = data.get("body", "")

    # Filter deleted/removed comments
    if body in {"[deleted]", "[removed]"}:
        return None

    depth = data.get("depth", current_depth)

    # Skip comments deeper than max_depth
    if depth > max_depth:
        return None

    edited_raw = data.get("edited", False)
    edited = edited_raw is not False

    # Parse nested replies
    replies_data = data.get("replies", "")
    replies: tuple[Comment, ...] = ()
    if replies_data and isinstance(replies_data, dict):
        replies = _parse_comment_tree(replies_data, max_depth, depth + 1)

    return Comment(
        author=data.get("author", "[deleted]"),
        score=data.get("score", 0),
        upvote_ratio=data.get("upvote_ratio", 1.0),
        created_at=_parse_timestamp(data.get("created_utc", 0)),
        depth=depth,
        is_submitter=data.get("is_submitter", False),
        distinguished=data.get("distinguished"),
        edited=edited,
        body=body,
        replies=replies,
    )


def _parse_comment_tree(listing: dict[str, Any], max_depth: int, current_depth: int = 0) -> tuple[Comment, ...]:
    """Walk a Listing and parse all comments.

    :param listing: Listing object from Reddit JSON.
    :param max_depth: Maximum comment depth to include.
    :param current_depth: Current depth in the tree.
    :return: Tuple of parsed Comment dataclasses.
    """
    comments: list[Comment] = []

    if listing.get("kind") != "Listing":
        return ()

    children = listing.get("data", {}).get("children", [])

    for child in children:
        kind = child.get("kind")

        if kind == "t1":
            comment = _parse_comment(child.get("data", {}), max_depth, current_depth)
            if comment is not None:
                comments.append(comment)

        # Skip "more" markers for now - expansion would require additional requests

    return tuple(comments)


# =============================================================================
# XML Generation
# =============================================================================


def _build_xml_tree(
    post: Post,
    comments: tuple[Comment, ...],
    url: str,
    retrieved_at: str,
) -> Element:
    """Construct XML ElementTree from parsed post and comments.

    :param post: Parsed Post dataclass.
    :param comments: Tuple of parsed Comment dataclasses.
    :param url: Original Reddit URL.
    :param retrieved_at: ISO timestamp of retrieval.
    :return: Root Element of the XML tree.
    """
    root = Element("reddit-thread")
    root.set("url", url)
    root.set("subreddit", post.subreddit)
    root.set("retrieved_at", retrieved_at)

    # Post element
    post_elem = SubElement(root, "post")
    post_elem.set("title", post.title)
    post_elem.set("author", post.author)
    post_elem.set("score", str(post.score))
    post_elem.set("upvote_ratio", str(post.upvote_ratio))
    post_elem.set("created_at", post.created_at)
    post_elem.set("num_comments", str(post.num_comments))
    post_elem.set("archived", str(post.archived).lower())
    post_elem.set("locked", str(post.locked).lower())

    selftext_elem = SubElement(post_elem, "selftext")
    selftext_elem.text = post.selftext

    # Comments element
    comments_elem = SubElement(root, "comments")
    _add_comments_to_xml(comments_elem, comments)

    return root


def _add_comments_to_xml(parent: Element, comments: tuple[Comment, ...]) -> None:
    """Recursively add comments to XML element.

    :param parent: Parent XML element to add comments to.
    :param comments: Tuple of Comment dataclasses.
    """
    for comment in comments:
        comment_elem = SubElement(parent, "comment")
        comment_elem.set("author", comment.author)
        comment_elem.set("score", str(comment.score))
        comment_elem.set("upvote_ratio", str(comment.upvote_ratio))
        comment_elem.set("created_at", comment.created_at)
        comment_elem.set("depth", str(comment.depth))
        comment_elem.set("is_submitter", str(comment.is_submitter).lower())
        comment_elem.set("distinguished", comment.distinguished or "")
        comment_elem.set("edited", str(comment.edited).lower())

        body_elem = SubElement(comment_elem, "body")
        body_elem.text = comment.body

        if comment.replies:
            replies_elem = SubElement(comment_elem, "replies")
            _add_comments_to_xml(replies_elem, comment.replies)


def _check_output_exists(path: Path, *, force: bool) -> bool:
    """Check if output file exists and should be skipped.

    :param path: Path to check.
    :param force: If True, never skip (return False).
    :return: True if file exists and should be skipped, False otherwise.
    """
    if path.exists() and not force:
        sys.stderr.write(f"File exists, skipping (use --force to overwrite): {path}\n")
        return True
    return False


# =============================================================================
# Entry Point: reddit-scrape
# =============================================================================


def _scrape_thread(url: str, max_depth: int) -> tuple[Post, tuple[Comment, ...]]:
    """Fetch and parse a Reddit thread.

    :param url: Reddit post URL.
    :param max_depth: Maximum comment depth.
    :return: Tuple of (Post, comments tuple).
    :raises RedditError: On fetch or parse errors.
    """
    normalized_url = _validate_reddit_url(url)
    json_url = _make_json_url(normalized_url)

    data = _fetch_json(json_url)

    # Post thread returns array of two Listings
    if not isinstance(data, list) or len(data) < 2:
        msg = "Unexpected response format: expected array with post and comments"
        raise RedditError(msg)

    # Index 0: Post listing
    post_listing = data[0]
    post_children = post_listing.get("data", {}).get("children", [])
    if not post_children:
        msg = "No post found in response"
        raise RedditError(msg)

    post_thing = post_children[0]
    if post_thing.get("kind") != "t3":
        msg = f"Expected post (t3), got {post_thing.get('kind')}"
        raise RedditError(msg)

    post = _parse_post(post_thing.get("data", {}))

    # Index 1: Comments listing
    comments_listing = data[1]
    comments = _parse_comment_tree(comments_listing, max_depth)

    return post, comments


def main_scrape() -> None:
    """Entry point for reddit-scrape command.

    Fetches a single Reddit post/thread and outputs as XML.
    """
    parser = argparse.ArgumentParser(
        prog="reddit-scrape",
        description="Fetch a single Reddit post/thread and output as XML.",
        epilog=f"""
OUTPUT FORMAT:
  XML with metadata header and nested comment structure:
    <?xml version="1.0" encoding="UTF-8"?>
    <reddit-thread url="..." subreddit="..." retrieved_at="...">
      <post title="..." author="..." score="..." ...>
        <selftext>Post body</selftext>
      </post>
      <comments>
        <comment author="..." score="..." depth="0">
          <body>Comment text</body>
          <replies>
            <comment ...>...</comment>
          </replies>
        </comment>
      </comments>
    </reddit-thread>

FILE CONFLICT:
  If --output file exists: skip fetch, exit 0
  Use --force to re-fetch and overwrite
{RATE_LIMIT_EPILOG}
EXAMPLES:
  # Scrape to stdout
  reddit-scrape "https://www.reddit.com/r/python/comments/abc123/post_title/"

  # Scrape to file
  reddit-scrape -o thread.xml "https://www.reddit.com/r/python/comments/abc123/"

  # Limit comment depth
  reddit-scrape --max-depth 3 "https://www.reddit.com/r/python/comments/abc123/"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        metavar="URL",
        help="Reddit post URL",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="output file path (default: stdout)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        metavar="N",
        help=f"maximum comment nesting depth (default: {DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing output file (default: skip if exists)",
    )

    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output)
        if _check_output_exists(output_path, force=args.force):
            sys.exit(0)

    try:
        post, comments = _scrape_thread(args.url, args.max_depth)
    except RedditError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

    retrieved_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    root = _build_xml_tree(post, comments, args.url, retrieved_at)
    indent(root)

    tree = ElementTree(root)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            tree.write(f, encoding="unicode")
        sys.stderr.write(f"Fetched: {args.url} ({post.num_comments} comments)\n")
        sys.stderr.write(f"Saved to: {output_path}\n")
    else:
        print('<?xml version="1.0" encoding="UTF-8"?>')
        tree.write(sys.stdout, encoding="unicode")
        print()


# =============================================================================
# Entry Point: reddit-batch-scrape
# =============================================================================


def _deduplicate_urls(urls: list[str]) -> list[str]:
    """Remove duplicate URLs while preserving order.

    :param urls: List of URLs possibly containing duplicates.
    :return: List of unique URLs in original order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _collect_urls_from_args_or_stdin(args_urls: list[str]) -> list[str]:
    """Collect URLs from command args or stdin fallback.

    :param args_urls: URLs provided as positional arguments.
    :return: Deduplicated list of URLs.
    """
    urls = args_urls
    if not urls and not sys.stdin.isatty():
        stdin_content = sys.stdin.read()
        urls = stdin_content.split()
    return _deduplicate_urls(urls)


def _truncate_url(url: str, max_len: int = 60) -> str:
    """Truncate URL for progress display.

    :param url: URL to truncate.
    :param max_len: Maximum length (default 60).
    :return: Truncated URL with ellipsis if needed.
    """
    if len(url) <= max_len:
        return url
    return url[: max_len - 3] + "..."


def main_batch_scrape() -> None:
    """Entry point for reddit-batch-scrape command.

    Fetches multiple Reddit posts and saves each to individual XML files.
    """
    parser = argparse.ArgumentParser(
        prog="reddit-batch-scrape",
        description="Fetch multiple Reddit posts and save each to individual XML files.",
        epilog=f"""
INPUT:
  URLs can be provided via:
  1. Positional arguments: reddit-batch-scrape url1 url2 url3
  2. Stdin (when no args and stdin is piped): cat urls.txt | reddit-batch-scrape

  URLs split on any whitespace (spaces, tabs, newlines).
  Duplicate URLs are removed before processing.

OUTPUT:
  Files saved with URL-derived names: {{subreddit}}_{{post_id}}_{{hash}}.xml

FILE CONFLICT:
  Default: skip files that exist
  Use --force to re-fetch and overwrite

EXIT CODES:
  0  All URLs scraped successfully
  1  Complete failure (all URLs failed)
  2  Partial success (some URLs failed, some succeeded)
{RATE_LIMIT_EPILOG}
EXAMPLES:
  # Scrape multiple URLs
  reddit-batch-scrape url1 url2 url3 -o ./scraped/

  # Pipe from file
  cat reddit_urls.txt | reddit-batch-scrape -o ./scraped/

  # Force re-scrape
  reddit-batch-scrape --force -o ./scraped/ url1 url2
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "urls",
        nargs="*",
        metavar="URL",
        help="Reddit post URLs (or pipe via stdin)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        metavar="DIR",
        help="output directory for XML files (required)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        metavar="N",
        help=f"maximum comment nesting depth (default: {DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing files (default: skip if exists)",
    )

    args = parser.parse_args()
    urls = _collect_urls_from_args_or_stdin(args.urls)

    if not urls:
        sys.stderr.write("No URLs to process\n")
        sys.exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved, skipped, failed = 0, 0, []

    try:
        for i, url in enumerate(urls, 1):
            sys.stderr.write(f"Scraping {i}/{len(urls)}: {_truncate_url(url)}\n")
            file_path = output_dir / _slugify_url(url)

            if file_path.exists() and not args.force:
                sys.stderr.write(f"  Skipped (exists): {file_path.name}\n")
                skipped += 1
                continue

            try:
                post, comments = _scrape_thread(url, args.max_depth)
                retrieved_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
                root = _build_xml_tree(post, comments, url, retrieved_at)
                indent(root)

                tree = ElementTree(root)
                with file_path.open("w", encoding="utf-8") as f:
                    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    tree.write(f, encoding="unicode")

                saved += 1
                sys.stderr.write(f"  Saved: {file_path.name}\n")
            except RedditError as e:
                sys.stderr.write(f"  Error: {e}\n")
                failed.append(url)

    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")

    sys.stderr.write(f"\nSaved: {saved}, Skipped: {skipped}, Failed: {len(failed)}\n")
    if failed:
        sys.stderr.write("Failed URLs:\n")
        for url in failed:
            sys.stderr.write(f"  - {url}\n")

    # Exit code: 0 = success, 1 = total failure, 2 = partial
    if saved == 0 and failed:
        sys.exit(1)
    if failed:
        sys.exit(2)


# =============================================================================
# Entry Point: reddit-feed
# =============================================================================


def _fetch_feed(url: str, limit: int) -> list[FeedPost]:
    """Fetch subreddit feed posts.

    :param url: Subreddit URL.
    :param limit: Maximum number of posts to return.
    :return: List of FeedPost dataclasses.
    :raises RedditError: On fetch or parse errors.
    """
    normalized_url = _validate_reddit_url(url)
    json_url = _make_json_url(normalized_url)

    # Add limit parameter
    if "?" in json_url:
        json_url += f"&limit={limit}"
    else:
        json_url += f"?limit={limit}"

    data = _fetch_json(json_url)

    # Feed returns single Listing
    if not isinstance(data, dict) or data.get("kind") != "Listing":
        msg = "Unexpected response format: expected Listing"
        raise RedditError(msg)

    children = data.get("data", {}).get("children", [])
    posts: list[FeedPost] = []

    for child in children:
        if child.get("kind") == "t3":
            post_data = child.get("data", {})
            posts.append(
                FeedPost(
                    title=post_data.get("title", ""),
                    url=f"https://www.reddit.com{post_data.get('permalink', '')}",
                    author=post_data.get("author", "[deleted]"),
                    score=post_data.get("score", 0),
                    upvote_ratio=post_data.get("upvote_ratio", 0.0),
                    num_comments=post_data.get("num_comments", 0),
                    created_at=_parse_timestamp(post_data.get("created_utc", 0)),
                    selftext=post_data.get("selftext", ""),
                )
            )

    return posts


def main_feed() -> None:
    """Entry point for reddit-feed command.

    Lists posts from a subreddit feed.
    """
    parser = argparse.ArgumentParser(
        prog="reddit-feed",
        description="List posts from a subreddit feed with optional preview.",
        epilog=f"""
OUTPUT FORMAT:
  Plain text (default):
    ## Post Title Here
    URL: https://www.reddit.com/r/.../...
    Author: username | Score: 156 (0.94) | Comments: 42
    > Preview text if --preview specified...

    ---

  JSON (with --json):
    [{{"title": "...", "url": "...", "author": "...", ...}}]
{RATE_LIMIT_EPILOG}
EXAMPLES:
  # List top 25 posts
  reddit-feed "https://www.reddit.com/r/python/"

  # Limit to 10 posts with preview
  reddit-feed --limit 10 --preview 200 "https://www.reddit.com/r/python/"

  # JSON output
  reddit-feed --json "https://www.reddit.com/r/python/"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        metavar="URL",
        help="subreddit URL (e.g., https://www.reddit.com/r/python/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        metavar="N",
        help="maximum posts to return (default: 25)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="characters of selftext to include (default: 0 = no preview)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output as JSON array instead of plain text",
    )

    args = parser.parse_args()

    try:
        posts = _fetch_feed(args.url, args.limit)
    except RedditError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

    if args.json:
        output_data = []
        for post in posts:
            item: dict[str, str | int | float] = {
                "title": post.title,
                "url": post.url,
                "author": post.author,
                "score": post.score,
                "upvote_ratio": post.upvote_ratio,
                "num_comments": post.num_comments,
                "created_at": post.created_at,
            }
            if args.preview > 0:
                item["preview"] = _truncate_text(post.selftext, args.preview)
            output_data.append(item)
        print(json.dumps(output_data, indent=2))
    else:
        for post in posts:
            print(f"## {post.title}")
            print(f"URL: {post.url}")
            ratio_pct = int(post.upvote_ratio * 100)
            print(f"Author: {post.author} | Score: {post.score} ({ratio_pct}%) | Comments: {post.num_comments}")
            if args.preview > 0 and post.selftext:
                preview = _truncate_text(post.selftext, args.preview)
                print(f"> {preview}")
            print("\n---\n")


if __name__ == "__main__":
    main_scrape()
