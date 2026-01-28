"""Firecrawl CLI utilities for token-efficient web scraping workflows.

Four CLI commands wrapping the Firecrawl Python SDK:
- firecrawl-scrape: Scrape single URL to markdown
- firecrawl-search: Web search with metadata results
- firecrawl-map: Discover URLs on a website
- firecrawl-crawl: Crawl multiple pages to files
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from firecrawl import Firecrawl
from firecrawl.types import CrawlErrorsResponse, CrawlJob, Document, ScrapeOptions


def _get_client() -> Firecrawl:
    """Initialize Firecrawl client with API key from environment.

    Loads environment variables from .env files via python-dotenv,
    then reads FIRECRAWL_API_KEY.

    :return: Initialized Firecrawl client.
    :raises SystemExit: If FIRECRAWL_API_KEY is not set.
    """
    load_dotenv()
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if api_key is None:
        sys.stderr.write("Error: FIRECRAWL_API_KEY environment variable not set\n")
        sys.exit(1)
    return Firecrawl(api_key=api_key)  # ty: ignore[invalid-argument-type]


def _format_markdown_output(content: str, title: str, url: str) -> str:
    """Add YAML frontmatter to markdown content.

    :param content: The markdown content body.
    :param title: Page title for frontmatter.
    :param url: Source URL for frontmatter.
    :return: Markdown string with YAML frontmatter.
    """
    scraped_at = datetime.now(UTC).isoformat()
    frontmatter = f"""---
title: "{title}"
url: "{url}"
scraped_at: "{scraped_at}"
---

"""
    return frontmatter + content


def _slugify_url(url: str, max_length: int = 100) -> str:
    """Convert URL to safe filename with hash suffix.

    Extracts path from URL, replaces non-alphanumeric characters with underscores,
    truncates to max_length, and appends a short MD5 hash for uniqueness.

    :param url: The URL to convert.
    :param max_length: Maximum filename length (default 100).
    :return: Safe filename with .md extension.
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    if not path:
        path = parsed.netloc

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")

    if not slug:
        slug = "index"

    url_hash = hashlib.md5(url.encode()).hexdigest()[:6]  # noqa: S324
    max_slug_length = max_length - len(url_hash) - 4  # 4 = underscore + .md

    if len(slug) > max_slug_length:
        slug = slug[:max_slug_length]

    return f"{slug}_{url_hash}.md"


def _print_credits(credits_used: int) -> None:
    """Print credit usage to stderr.

    :param credits_used: Number of credits consumed.
    """
    sys.stderr.write(f"Credits used: {credits_used}\n")


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
# Entry Point: firecrawl-scrape
# =============================================================================


def main_scrape() -> None:
    """Entry point for firecrawl-scrape command.

    Scrapes a single URL and outputs clean markdown with YAML frontmatter.
    """
    parser = argparse.ArgumentParser(
        prog="firecrawl-scrape",
        description="Scrape a single URL and output clean markdown with YAML frontmatter.",
        epilog="""
OUTPUT FORMAT:
  Markdown with YAML frontmatter:
    ---
    title: "Page Title"
    url: "https://example.com/page"
    scraped_at: "2026-01-25T14:30:00Z"
    ---

    # Page content here...

  Output goes to stdout by default, or to file with --output.

FILE CONFLICT:
  If --output file exists: skip API call (save credits), exit 0
  Use --force to re-scrape and overwrite existing file

COST:
  1 credit per scrape

ENVIRONMENT:
  FIRECRAWL_API_KEY  Required. Load from .env file or environment.

EXAMPLES:
  # Scrape to stdout
  firecrawl-scrape https://example.com/docs/api

  # Scrape to file
  firecrawl-scrape https://example.com/docs/api -o api_docs.md

  # Include full page (navigation, footer)
  firecrawl-scrape --full-page https://example.com -o full.md

  # Force re-scrape existing file
  firecrawl-scrape --force -o existing.md https://example.com
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        metavar="URL",
        help="URL to scrape",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="output file path (default: stdout)",
    )
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="include navigation/footer (default: main content only)",
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

    client = _get_client()
    result = client.scrape(
        args.url,
        formats=["markdown"],
        only_main_content=not args.full_page,
    )

    title = result.metadata.title if result.metadata else "Untitled"
    url = result.metadata.source_url if result.metadata else args.url
    content = result.markdown or ""

    output = _format_markdown_output(content, title, url)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
        sys.stderr.write(f"Saved to: {output_path}\n")
    else:
        print(output)

    _print_credits(1)


# =============================================================================
# Entry Point: firecrawl-batch-scrape
# =============================================================================


def _truncate_url(url: str, max_len: int = 50) -> str:
    """Truncate URL for progress display.

    :param url: URL to truncate.
    :param max_len: Maximum length (default 50).
    :return: Truncated URL with ellipsis if needed.
    """
    if len(url) <= max_len:
        return url
    return url[: max_len - 3] + "..."


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


def _scrape_single_url(client: Firecrawl, url: str, file_path: Path, *, full_page: bool) -> tuple[bool, str | None]:
    """Scrape a single URL and save to file.

    :param client: Firecrawl client.
    :param url: URL to scrape.
    :param file_path: Path to save the scraped content.
    :param full_page: Whether to include full page content.
    :return: Tuple of (success, error_message).
    """
    try:
        result = client.scrape(url, formats=["markdown"], only_main_content=not full_page)
        title = result.metadata.title if result.metadata else "Untitled"
        source_url = result.metadata.source_url if result.metadata else url
        content = result.markdown or ""
        output = _format_markdown_output(content, title, source_url)
        file_path.write_text(output, encoding="utf-8")
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _print_batch_summary(saved: int, skipped: int, failed: list[str], credits_used: int) -> None:
    """Print batch scrape summary to stderr.

    :param saved: Number of successfully saved files.
    :param skipped: Number of skipped files.
    :param failed: List of failed URLs.
    :param credits_used: Total credits used.
    """
    sys.stderr.write(f"\nSaved: {saved}, Skipped: {skipped}, Failed: {len(failed)}\n")
    if failed:
        sys.stderr.write("Failed URLs:\n")
        for url in failed:
            sys.stderr.write(f"  - {url}\n")
    _print_credits(credits_used)


def main_batch_scrape() -> None:
    """Entry point for firecrawl-batch-scrape command.

    Scrapes multiple URLs and saves each to individual files.
    """
    parser = argparse.ArgumentParser(
        prog="firecrawl-batch-scrape",
        description="Scrape multiple URLs and save each to individual markdown files.",
        epilog="""
INPUT:
  URLs can be provided via:
  1. Positional arguments: firecrawl-batch-scrape url1 url2 url3
  2. Stdin (when no args and stdin is piped): cat urls.txt | firecrawl-batch-scrape

  URLs split on any whitespace (spaces, tabs, newlines).
  Duplicate URLs are removed before processing.

OUTPUT FORMAT:
  Each file contains markdown with YAML frontmatter:
    ---
    title: "Page Title"
    url: "https://example.com/page"
    scraped_at: "2026-01-25T14:30:00Z"
    ---

    # Page content...

FILENAME GENERATION:
  URL path slugified with hash suffix: path_segment_HASH.md
  Example: https://docs.example.com/api/auth -> api_auth_a1b2c3.md

FILE CONFLICT:
  Default: skip files that exist (save credits)
  Use --force to re-scrape and overwrite

EXIT CODES:
  0  All URLs scraped successfully (or no URLs to process)
  1  Complete failure (auth error, no URLs processed)
  2  Partial success (some URLs failed, some succeeded)

COST:
  1 credit per URL scraped (skipped files cost nothing)

ENVIRONMENT:
  FIRECRAWL_API_KEY  Required. Load from .env file or environment.

EXAMPLES:
  # Scrape multiple URLs to current directory
  firecrawl-batch-scrape https://example.com/page1 https://example.com/page2

  # Scrape to specific directory
  firecrawl-batch-scrape -o ./scraped/ url1 url2 url3

  # Pipe from firecrawl-map output
  firecrawl-map https://docs.example.com | firecrawl-batch-scrape -o ./docs/

  # Force re-scrape existing files
  firecrawl-batch-scrape --force -o ./docs/ url1 url2
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "urls",
        nargs="*",
        metavar="URL",
        help="URLs to scrape (or pipe via stdin)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        metavar="DIR",
        help="output directory for scraped files (default: current directory)",
    )
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="include navigation/footer (default: main content only)",
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
    client = _get_client()

    saved, skipped, failed, credits_used = 0, 0, [], 0

    try:
        for i, url in enumerate(urls, 1):
            sys.stderr.write(f"Scraping {i}/{len(urls)}: {_truncate_url(url)}\n")
            file_path = output_dir / _slugify_url(url)

            if file_path.exists() and not args.force:
                sys.stderr.write(f"  Skipped (exists): {file_path.name}\n")
                skipped += 1
                continue

            success, error = _scrape_single_url(client, url, file_path, full_page=args.full_page)
            if success:
                saved += 1
                credits_used += 1
            else:
                sys.stderr.write(f"  Error: {error}\n")
                failed.append(url)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")

    _print_batch_summary(saved, skipped, failed, credits_used)

    # Exit code: 0 = success, 1 = total failure, 2 = partial
    if saved == 0 and failed:
        sys.exit(1)
    if failed:
        sys.exit(2)


# =============================================================================
# Entry Point: firecrawl-search
# =============================================================================


def main_search() -> None:
    """Entry point for firecrawl-search command.

    Searches the web and returns result metadata.
    """
    parser = argparse.ArgumentParser(
        prog="firecrawl-search",
        description="Search the web and return result metadata (titles, URLs, descriptions).",
        epilog="""
OUTPUT FORMAT:
  Plain text (default):
    ## Result Title
    URL: https://example.com/page
    Description of the search result...

    ---

    ## Another Result
    URL: https://example.com/other
    Another description...

  JSON (with --json):
    [
      {
        "title": "Result Title",
        "url": "https://example.com/page",
        "description": "Description of the search result..."
      }
    ]

COST:
  2 credits per 10 results

ENVIRONMENT:
  FIRECRAWL_API_KEY  Required. Load from .env file or environment.

EXAMPLES:
  # Basic search
  firecrawl-search "python web scraping tutorial"

  # Limit results
  firecrawl-search --limit 5 "react hooks best practices"

  # JSON output for programmatic use
  firecrawl-search --json "API documentation examples"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        metavar="QUERY",
        help="search query string",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="number of results to return (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output as JSON array instead of plain text",
    )

    args = parser.parse_args()

    client = _get_client()
    result = client.search(args.query, limit=args.limit)

    results_data = result.web or []

    if args.json:
        json_output = [
            {
                "title": item.title or "",
                "url": item.url or "",
                "description": item.description or "",
            }
            for item in results_data
        ]
        print(json.dumps(json_output, indent=2))
    else:
        for item in results_data:
            print(f"## {item.title or 'Untitled'}")
            print(f"URL: {item.url}")
            print(item.description or "No description")
            print("\n---\n")

    used = (len(results_data) + 9) // 10 * 2
    _print_credits(used)


# =============================================================================
# Entry Point: firecrawl-map
# =============================================================================


def main_map() -> None:
    """Entry point for firecrawl-map command.

    Discovers all URLs on a website.
    """
    parser = argparse.ArgumentParser(
        prog="firecrawl-map",
        description="Discover all URLs on a website. Useful for planning which pages to scrape.",
        epilog="""
OUTPUT FORMAT:
  Plain text (default): one URL per line
    https://example.com/
    https://example.com/docs/
    https://example.com/docs/getting-started/
    https://example.com/api/

  JSON (with --json):
    [
      {"url": "https://example.com/", "title": "Home", "description": "..."},
      {"url": "https://example.com/docs/", "title": "Documentation", "description": "..."}
    ]

WORKFLOW:
  Map first, then selectively scrape:
    firecrawl-map https://docs.example.com > urls.txt
    # Review urls.txt, keep only needed URLs
    cat urls.txt | firecrawl-batch-scrape -o ./docs/

COST:
  1 credit per map operation

ENVIRONMENT:
  FIRECRAWL_API_KEY  Required. Load from .env file or environment.

EXAMPLES:
  # Discover all URLs on a site
  firecrawl-map https://docs.example.com

  # Limit number of URLs
  firecrawl-map --limit 100 https://docs.example.com

  # Filter URLs containing "api"
  firecrawl-map --search api https://docs.example.com

  # JSON output for programmatic use
  firecrawl-map --json https://docs.example.com

  # Pipe directly to batch-scrape
  firecrawl-map https://docs.example.com | firecrawl-batch-scrape -o ./docs/
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        metavar="URL",
        help="starting URL to map from",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="maximum URLs to return (default: API default)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output as JSON array instead of plain text",
    )
    parser.add_argument(
        "--search",
        metavar="PATTERN",
        help="filter URLs containing this string",
    )

    args = parser.parse_args()

    client = _get_client()

    map_params: dict[str, str | int] = {}
    if args.limit:
        map_params["limit"] = args.limit
    if args.search:
        map_params["search"] = args.search

    result = client.map(args.url, **map_params)

    links = result.links or []

    if args.json:
        json_output = [
            {"url": link.url, "title": link.title or "", "description": link.description or ""} for link in links
        ]
        print(json.dumps(json_output, indent=2))
    else:
        for link in links:
            title = f" - {link.title}" if link.title else ""
            print(f"{link.url}{title}")

    _print_credits(1)


# =============================================================================
# Entry Point: firecrawl-crawl
# =============================================================================

CRAWL_POLL_INTERVAL = 2


def _save_crawl_page(page: Document, output_dir: Path, *, skip_existing: bool) -> str:
    """Save a single crawled page to file.

    :return: "saved", "skipped", or "no_url"
    """
    page_url = page.metadata.source_url if page.metadata else ""
    if not page_url:
        return "no_url"

    file_path = output_dir / _slugify_url(page_url)

    if skip_existing and file_path.exists():
        return "skipped"

    title = page.metadata.title if page.metadata and page.metadata.title else "Untitled"
    content = page.markdown or ""
    file_path.write_text(_format_markdown_output(content, title, page_url))
    return "saved"


def _save_all_pages(pages: list[Document], output_dir: Path, *, skip_existing: bool) -> tuple[int, int]:
    """Save all crawled pages to files.

    :return: Tuple of (saved_count, skipped_count)
    """
    saved_count = 0
    skipped_count = 0
    for page in pages:
        status = _save_crawl_page(page, output_dir, skip_existing=skip_existing)
        if status == "saved":
            saved_count += 1
        elif status == "skipped":
            skipped_count += 1
    return saved_count, skipped_count


def _poll_crawl_status(client: Firecrawl, job_id: str, limit: int) -> CrawlJob:
    """Poll crawl status until complete/failed/cancelled, printing progress to stderr."""
    while True:
        status = client.get_crawl_status(job_id)

        # Track progress via scraped page count
        scraped = status.completed
        total = status.total or limit

        # Debug: show actual status value
        sys.stderr.write(f"[status={status.status}] Crawling: {scraped}/{total} pages\n")

        if status.status in {"completed", "failed", "cancelled"}:
            return status

        time.sleep(CRAWL_POLL_INTERVAL)


def _print_crawl_errors(client: Firecrawl, job_id: str) -> None:
    """Fetch and print crawl errors to stderr."""
    try:
        errors_response: CrawlErrorsResponse = client.get_crawl_errors(job_id)
        if errors_response.errors:
            sys.stderr.write("Errors:\n")
            for err in errors_response.errors:
                sys.stderr.write(f"  - {err.url}: {err.error}\n")
        if errors_response.robots_blocked:
            sys.stderr.write(f"Blocked by robots.txt: {len(errors_response.robots_blocked)} URLs\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Could not fetch error details: {e}\n")


def main_crawl() -> None:
    """Entry point for firecrawl-crawl command."""
    parser = argparse.ArgumentParser(
        prog="firecrawl-crawl",
        description="Crawl multiple pages from a website and save each to a markdown file.",
        epilog="""
OUTPUT FORMAT:
  Each page saved as markdown with YAML frontmatter:
    ---
    title: "Page Title"
    url: "https://example.com/page"
    scraped_at: "2026-01-25T14:30:00Z"
    ---

    # Page content...

FILENAME GENERATION:
  URL path slugified with hash suffix: path_segment_HASH.md
  Example: https://docs.example.com/api/auth -> api_auth_a1b2c3.md

FILE CONFLICT:
  Default: overwrite existing files
  Use --skip-existing to skip (save credits on re-runs)

PATH FILTERS:
  --include-path and --exclude-path accept glob patterns:
    --include-path "/docs/*"     Only crawl /docs/ pages
    --include-path "/api/*"      Can repeat for multiple patterns
    --exclude-path "/blog/*"     Skip blog pages

SITEMAP OPTIONS:
  include  Use sitemap + discovered links (default)
  skip     Ignore sitemap, discover links only
  only     Use sitemap only, no link discovery

EXIT CODES:
  0  Crawl completed successfully
  1  Crawl failed or was cancelled

COST:
  1 credit per page crawled

ENVIRONMENT:
  FIRECRAWL_API_KEY  Required. Load from .env file or environment.

EXAMPLES:
  # Crawl documentation site (max 50 pages)
  firecrawl-crawl https://docs.example.com -o ./docs/

  # Crawl with limit
  firecrawl-crawl --limit 100 https://docs.example.com -o ./docs/

  # Crawl only API docs
  firecrawl-crawl --include-path "/api/*" https://docs.example.com -o ./api/

  # Exclude blog, include everything else
  firecrawl-crawl --exclude-path "/blog/*" https://example.com -o ./site/

  # Include subdomains
  firecrawl-crawl --allow-subdomains https://example.com -o ./all/

  # Skip already-downloaded pages on re-run
  firecrawl-crawl --skip-existing https://docs.example.com -o ./docs/
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        metavar="URL",
        help="starting URL to crawl from",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        metavar="DIR",
        help="output directory for scraped files (required)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="maximum pages to crawl (default: 50)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        metavar="N",
        help="maximum link discovery depth (default: unlimited)",
    )
    parser.add_argument(
        "--include-path",
        action="append",
        metavar="PATTERN",
        help="only crawl paths matching pattern (can repeat)",
    )
    parser.add_argument(
        "--exclude-path",
        action="append",
        metavar="PATTERN",
        help="skip paths matching pattern (can repeat)",
    )
    parser.add_argument(
        "--allow-subdomains",
        action="store_true",
        help="include pages from subdomains",
    )
    parser.add_argument(
        "--sitemap",
        choices=["include", "skip", "only"],
        default="include",
        metavar="MODE",
        help="sitemap usage: include, skip, only (default: include)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip pages whose output file already exists (save credits)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    client = _get_client()

    # Start crawl job
    job = client.start_crawl(
        args.url,
        limit=args.limit,
        max_discovery_depth=args.max_depth,
        include_paths=args.include_path,
        exclude_paths=args.exclude_path,
        allow_subdomains=args.allow_subdomains,
        sitemap=args.sitemap,
        scrape_options=ScrapeOptions(formats=["markdown"], only_main_content=True),
    )
    job_id = job.id

    # Poll until complete, always save whatever we got
    try:
        result = _poll_crawl_status(client, job_id, args.limit)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted. Cancelling crawl job...\n")
        try:
            client.cancel_crawl(job_id)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"Failed to cancel crawl: {e}\n")
        result = client.get_crawl_status(job_id)

    pages = result.data or []
    saved_count, skipped_count = _save_all_pages(pages, output_dir, skip_existing=args.skip_existing)

    sys.stderr.write(f"Saved {saved_count} pages to {output_dir}/\n")
    if skipped_count > 0:
        sys.stderr.write(f"Skipped {skipped_count} existing files\n")

    _print_credits(result.credits_used)

    # Exit with error if crawl failed/cancelled
    if result.status == "failed":
        sys.stderr.write("Crawl failed.\n")
        _print_crawl_errors(client, job_id)
        sys.exit(1)
    if result.status == "cancelled":
        sys.stderr.write("Crawl was cancelled\n")
        sys.exit(1)


if __name__ == "__main__":
    main_crawl()
