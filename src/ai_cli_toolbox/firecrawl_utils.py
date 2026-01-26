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
        description="Scrape a URL and output clean markdown",
    )
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="Include navigation/footer (default: main content only)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")

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
# Entry Point: firecrawl-search
# =============================================================================


def main_search() -> None:
    """Entry point for firecrawl-search command.

    Searches the web and returns result metadata.
    """
    parser = argparse.ArgumentParser(
        prog="firecrawl-search",
        description="Search the web and return result metadata",
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--json", action="store_true", help="Output as JSON array")

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
        description="Discover URLs on a website",
    )
    parser.add_argument("url", help="Starting URL")
    parser.add_argument("--limit", type=int, help="Maximum URLs to return")
    parser.add_argument("--json", action="store_true", help="Output as JSON array")
    parser.add_argument("--search", help="Filter URLs containing this string")

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
        description="Crawl multiple pages and save to files",
    )
    parser.add_argument("url", help="Starting URL")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=50, help="Maximum pages to crawl (default: 50)")
    parser.add_argument("--max-depth", type=int, help="Maximum discovery depth")
    parser.add_argument("--include-path", action="append", help="Only crawl paths matching pattern (can repeat)")
    parser.add_argument("--exclude-path", action="append", help="Skip paths matching pattern (can repeat)")
    parser.add_argument("--allow-subdomains", action="store_true", help="Include subdomain pages")
    parser.add_argument(
        "--sitemap",
        choices=["include", "skip", "only"],
        default="include",
        help="How to use sitemap (default: include)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip pages whose output file already exists")

    args = parser.parse_args()
    output_dir = Path(args.output)

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
