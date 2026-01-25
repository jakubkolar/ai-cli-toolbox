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
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from firecrawl import Firecrawl
from firecrawl.types import ScrapeOptions
from tqdm import tqdm


def _get_client() -> Firecrawl:
    """Initialize Firecrawl client with API key from environment.

    Loads environment variables from .env files via python-dotenv,
    then reads FIRECRAWL_API_KEY.

    :return: Initialized Firecrawl client.
    :raises SystemExit: If FIRECRAWL_API_KEY is not set.
    """
    load_dotenv()
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        sys.stderr.write("Error: FIRECRAWL_API_KEY environment variable not set\n")
        sys.exit(1)
    return Firecrawl(api_key=api_key)


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
    parser.add_argument("--dry-run", action="store_true", help="Show estimated credits without scraping")

    args = parser.parse_args()

    if args.dry_run:
        sys.stderr.write("Estimated credits: 1\n")
        sys.exit(0)

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
    parser.add_argument("--dry-run", action="store_true", help="Show estimated credits without searching")

    args = parser.parse_args()

    if args.dry_run:
        estimated = (args.limit + 9) // 10 * 2  # 2 credits per 10 results
        sys.stderr.write(f"Estimated credits: {estimated}\n")
        sys.exit(0)

    client = _get_client()
    result = client.search(args.query, limit=args.limit)

    results_data = result.data or []

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

    credits = (len(results_data) + 9) // 10 * 2
    _print_credits(credits)


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
    parser.add_argument("--dry-run", action="store_true", help="Show estimated credits without mapping")

    args = parser.parse_args()

    if args.dry_run:
        sys.stderr.write("Estimated credits: 1\n")
        sys.exit(0)

    client = _get_client()

    map_params: dict[str, str | int] = {}
    if args.limit:
        map_params["limit"] = args.limit
    if args.search:
        map_params["search"] = args.search

    result = client.map_url(args.url, **map_params)

    links = result.links or []

    if args.json:
        print(json.dumps(links, indent=2))
    else:
        for link in links:
            print(link)

    _print_credits(1)


# =============================================================================
# Entry Point: firecrawl-crawl
# =============================================================================


def main_crawl() -> None:
    """Entry point for firecrawl-crawl command.

    Crawls multiple pages from a website and saves to files.
    """
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
    parser.add_argument("--dry-run", action="store_true", help="Show estimated credits without crawling")

    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.dry_run:
        sys.stderr.write(f"Estimated credits: up to {args.limit}\n")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client()

    crawl_params: dict[str, object] = {
        "limit": args.limit,
        "scrape_options": ScrapeOptions(formats=["markdown"], only_main_content=True),
        "poll_interval": 5,
    }

    if args.max_depth is not None:
        crawl_params["max_depth"] = args.max_depth
    if args.include_path:
        crawl_params["include_paths"] = args.include_path
    if args.exclude_path:
        crawl_params["exclude_paths"] = args.exclude_path
    if args.allow_subdomains:
        crawl_params["allow_subdomains"] = True
    if args.sitemap != "include":
        crawl_params["ignore_sitemap"] = args.sitemap == "skip"

    try:
        result = client.crawl(args.url, **crawl_params)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Crawl failed: {e}\n")
        sys.exit(1)

    pages = result.data or []

    saved_count = 0
    skipped_count = 0

    with tqdm(total=len(pages), desc="Saving pages", unit="page") as pbar:
        for page in pages:
            page_url = page.metadata.source_url if page.metadata else ""
            if not page_url:
                pbar.update(1)
                continue

            filename = _slugify_url(page_url)
            file_path = output_dir / filename

            if args.skip_existing and file_path.exists():
                skipped_count += 1
                pbar.update(1)
                continue

            if file_path.exists():
                sys.stderr.write(f"Overwriting: {file_path}\n")

            title = page.metadata.title if page.metadata else "Untitled"
            content = page.markdown or ""
            output = _format_markdown_output(content, title, page_url)

            file_path.write_text(output)
            saved_count += 1
            pbar.update(1)

    sys.stderr.write(f"Saved {saved_count} pages to {output_dir}/\n")
    if skipped_count > 0:
        sys.stderr.write(f"Skipped {skipped_count} existing files\n")

    _print_credits(result.credits_used)


if __name__ == "__main__":
    main_scrape()
