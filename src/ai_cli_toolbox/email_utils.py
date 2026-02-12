"""Email CLI utilities for token-efficient IMAP email management.

Six CLI commands for AI-agent-friendly email operations:
- email-list: List/search emails in a folder
- email-read: Fetch full email content by UID
- email-move: Move emails to a folder
- email-flag: Mark emails as read/unread, starred/unstarred
- email-draft: Create a draft email (with reply support and attachments)
- email-folder: Manage folders (list, create, rename, delete, exists)
"""

import argparse
import datetime
import email.message
import email.utils
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import html2text
from dotenv import load_dotenv
from imap_tools import AND, MailBox, MailMessageFlags
from imap_tools.errors import UnexpectedCommandStatusError
from imap_tools.message import MailAttachment, MailMessage

if TYPE_CHECKING:
    from imap_tools.folder import FolderInfo


class EmailError(Exception):
    """Domain-specific exception for email CLI operations."""


def _get_mailbox() -> MailBox:
    """Load env vars and return a connected (not logged in) MailBox."""
    load_dotenv()
    host = os.environ.get("IMAP_HOST")
    user = os.environ.get("IMAP_USER")
    password = os.environ.get("IMAP_PASSWORD")
    if not host or not user or not password:
        missing = [v for v in ("IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD") if not os.environ.get(v)]
        sys.stderr.write(f"Error: Missing required environment variables: {', '.join(missing)}\n")
        sys.exit(1)
    port = int(os.environ.get("IMAP_PORT", "993"))
    return MailBox(host, port=port)


def _find_special_folder(mb: MailBox, attribute: str) -> str:
    r"""Discover folder by RFC 6154 attribute (e.g. ``\\Drafts``).

    Falls back to well-known names if attribute not found.

    :param mb: Connected and logged-in MailBox.
    :param attribute: RFC 6154 attribute to search for.
    :return: Folder name matching the attribute.
    """
    folders: list[FolderInfo] = mb.folder.list()
    for folder in folders:
        if attribute in folder.flags:
            return folder.name

    # Fallback to well-known names
    well_known: dict[str, list[str]] = {
        "\\Drafts": ["[Gmail]/Drafts", "[GoogleMail]/Drafts", "INBOX.Drafts", "Drafts"],
        "\\Trash": ["[Gmail]/Trash", "[GoogleMail]/Trash", "INBOX.Trash", "Trash"],
        "\\Sent": ["[Gmail]/Sent Mail", "[GoogleMail]/Sent Mail", "INBOX.Sent", "Sent"],
    }
    folder_names = {f.name for f in folders}
    for fallback_name in well_known.get(attribute, []):
        if fallback_name in folder_names:
            return fallback_name

    sys.stderr.write(f"Warning: Could not find folder with attribute {attribute}\n")
    return "INBOX"


def _is_gmail() -> bool:
    """Check if the IMAP host is a Gmail server."""
    host = os.environ.get("IMAP_HOST", "")
    return "gmail" in host.lower() or "googlemail" in host.lower()


def _criteria_has_non_ascii(criteria: str | AND) -> bool:
    """Check if search criteria contain non-ASCII characters."""
    return not str(criteria).isascii()


def _build_gmail_raw_query(args: argparse.Namespace) -> str:
    """Build a Gmail X-GM-RAW search query string from CLI filter args.

    Maps CLI flags to Gmail search operators.
    See https://support.google.com/mail/answer/7190

    :param args: Parsed argparse namespace with filter flags.
    :return: Gmail search query string.
    """
    parts: list[str] = []
    if args.from_filter:
        parts.append(f"from:{args.from_filter}")
    if args.subject:
        parts.append(f"subject:{args.subject}")
    if args.body:
        # Gmail uses bare text for body search
        parts.append(args.body)
    if args.since:
        parts.append(f"after:{args.since.isoformat()}")
    if args.before:
        parts.append(f"before:{args.before.isoformat()}")
    if args.unseen:
        parts.append("is:unread")
    return " ".join(parts)


def _gmail_raw_uids(mb: MailBox, query: str) -> list[str]:
    """Search Gmail using X-GM-RAW extension and return matching UIDs.

    :param mb: Connected and logged-in MailBox.
    :param query: Gmail search query string.
    :return: List of matching UIDs.
    """
    # Use IMAP literal to transmit non-ASCII query bytes
    query_bytes = query.encode("utf-8")
    mb.client.literal = query_bytes  # type: ignore[assignment]  # imaplib accepts bytes at runtime, stubs type it as str
    result = mb.client.uid("SEARCH", "CHARSET", "UTF-8", "X-GM-RAW")
    if result[0] != "OK":
        msg = f"Gmail search failed: {result[1]}"
        raise EmailError(msg)
    return result[1][0].decode().split() if result[1][0] else []


def _fetch_via_gmail_raw(mb: MailBox, args: argparse.Namespace) -> tuple[list[MailMessage], str]:
    """Fetch messages using Gmail X-GM-RAW search for non-ASCII criteria.

    :param mb: Connected and logged-in MailBox.
    :param args: Parsed argparse namespace with filter flags.
    :return: Tuple of (messages, total_str for display).
    """
    gmail_query = _build_gmail_raw_query(args)
    all_uids = _gmail_raw_uids(mb, gmail_query)
    total_str = f" of {len(all_uids)}" if args.count else ""
    # Apply limit (most recent = highest UIDs)
    limited_uids = list(reversed(all_uids))[: args.limit] if args.limit else list(reversed(all_uids))
    uid_criteria = ",".join(limited_uids) if limited_uids else None
    if uid_criteria:
        messages = list(mb.fetch(f"UID {uid_criteria}", mark_seen=False))
        messages.sort(key=lambda m: int(m.uid), reverse=True)
    else:
        messages = []
    return messages, total_str


def _parse_date(value: str) -> datetime.date:
    """Parse a date string (YYYY-MM-DD) to ``datetime.date``.

    :param value: Date string in YYYY-MM-DD format.
    :return: Parsed date.
    :raises argparse.ArgumentTypeError: If the date format is invalid.
    """
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as e:
        msg = f"Invalid date format '{value}': {e}. Use YYYY-MM-DD."
        raise argparse.ArgumentTypeError(msg) from e


def _build_criteria(args: argparse.Namespace) -> str | AND:
    """Map CLI filter flags to imap_tools search criteria.

    :param args: Parsed argparse namespace with filter flags.
    :return: IMAP search criteria string or AND query object.
    """
    if args.query:
        return args.query

    # Build kwargs with only non-None values to satisfy AND's type requirements
    kwargs: dict[str, str | bool | datetime.date] = {}
    if args.from_filter:
        kwargs["from_"] = str(args.from_filter)
    if args.subject:
        kwargs["subject"] = str(args.subject)
    if args.body:
        kwargs["body"] = str(args.body)
    if args.since:
        kwargs["date_gte"] = args.since
    if args.before:
        kwargs["date_lt"] = args.before
    if args.unseen:
        kwargs["seen"] = False

    if kwargs:
        return AND(**kwargs)  # type: ignore[arg-type]  # kwargs values are str|bool|date, AND accepts these at runtime
    return "ALL"


def _format_email_block(msg: MailMessage, preview: int) -> str:
    """Format a single email for ``email-list`` output.

    :param msg: Mail message to format.
    :param preview: Number of body characters to include (0 = none).
    :return: Formatted text block.
    """
    lines = [
        f"UID: {msg.uid}",
        f"Date: {msg.date.strftime('%Y-%m-%d %H:%M') if msg.date.year > 1900 else 'Unknown'}",
        f"From: {msg.from_}",
        f"Subject: {msg.subject}",
        f"Flags: {' '.join(msg.flags) if msg.flags else '(none)'}",
    ]
    if preview > 0:
        body_text = msg.text or ""
        if body_text:
            truncated = body_text[:preview].replace("\n", " ").strip()
            lines.append(f"\n> {truncated}{'...' if len(body_text) > preview else ''}")
    lines.append("\n---")
    return "\n".join(lines)


def _email_block_to_dict(msg: MailMessage, preview: int) -> dict[str, object]:
    """Convert a mail message to a dict for JSON output.

    :param msg: Mail message.
    :param preview: Number of body characters to include.
    :return: Dict representation.
    """
    result: dict[str, object] = {
        "uid": msg.uid,
        "date": msg.date.isoformat() if msg.date.year > 1900 else None,
        "from": msg.from_,
        "subject": msg.subject,
        "flags": list(msg.flags),
    }
    if preview > 0:
        body_text = msg.text or ""
        result["preview"] = body_text[:preview]
    return result


# =============================================================================
# Folder management helpers
# =============================================================================


def _get_delimiter(mb: MailBox) -> str:
    """Discover the server's hierarchy delimiter from existing folders.

    :param mb: Connected and logged-in MailBox.
    :return: Hierarchy delimiter character.
    """
    folders: list[FolderInfo] = mb.folder.list()
    if folders:
        return folders[0].delim
    return "/"


def _normalize_folder_path(name: str, delimiter: str) -> str:
    """Replace ``/`` in user input with the server's hierarchy delimiter.

    :param name: Folder path as typed by the user.
    :param delimiter: Server's hierarchy delimiter.
    :return: Normalized folder path.
    """
    if delimiter == "/":
        return name
    return name.replace("/", delimiter)


def _create_folder_parents(mb: MailBox, name: str, delimiter: str) -> None:
    """Create folder and all intermediate parents (like ``mkdir -p``).

    :param mb: Connected and logged-in MailBox.
    :param name: Fully normalized folder path.
    :param delimiter: Server's hierarchy delimiter.
    """
    parts = name.split(delimiter)
    for i in range(1, len(parts) + 1):
        partial = delimiter.join(parts[:i])
        if not mb.folder.exists(partial):
            mb.folder.create(partial)


# =============================================================================
# Entry Point: email-folder
# =============================================================================


def _folder_list(mb: MailBox, args: argparse.Namespace) -> None:
    """Handle ``email-folder list`` subcommand."""
    folders: list[FolderInfo] = mb.folder.list()
    folder_data: list[dict[str, object]] = []
    for f in folders:
        try:
            status = mb.folder.status(f.name, ("MESSAGES", "UNSEEN"))
        except UnexpectedCommandStatusError:
            continue
        folder_data.append(
            {
                "name": f.name,
                "messages": status.get("MESSAGES", 0),
                "unseen": status.get("UNSEEN", 0),
            }
        )

    if args.json:
        print(json.dumps(folder_data, indent=2))
    else:
        max_name_len = max((len(str(fd["name"])) for fd in folder_data), default=10)
        for fd in folder_data:
            name = str(fd["name"])
            msgs = fd["messages"]
            unseen = fd["unseen"]
            print(f"{name:<{max_name_len}}  {msgs} messages, {unseen} unseen")


def _folder_create(mb: MailBox, args: argparse.Namespace) -> None:
    """Handle ``email-folder create`` subcommand."""
    delimiter = _get_delimiter(mb)
    normalized = _normalize_folder_path(args.name, delimiter)
    if mb.folder.exists(normalized):
        sys.stderr.write(f'Error: Folder "{args.name}" already exists.\n')
        sys.exit(1)
    _create_folder_parents(mb, normalized, delimiter)
    sys.stderr.write(f'Created folder "{args.name}"\n')


def _folder_rename(mb: MailBox, args: argparse.Namespace) -> None:
    """Handle ``email-folder rename`` subcommand."""
    delimiter = _get_delimiter(mb)
    old_normalized = _normalize_folder_path(args.old_name, delimiter)
    new_normalized = _normalize_folder_path(args.new_name, delimiter)
    if not mb.folder.exists(old_normalized):
        sys.stderr.write(f'Error: Folder "{args.old_name}" does not exist.\n')
        sys.exit(1)
    if mb.folder.exists(new_normalized):
        sys.stderr.write(f'Error: Folder "{args.new_name}" already exists.\n')
        sys.exit(1)
    mb.folder.rename(old_normalized, new_normalized)
    sys.stderr.write(f'Renamed folder "{args.old_name}" to "{args.new_name}"\n')


def _folder_delete(mb: MailBox, args: argparse.Namespace) -> None:
    """Handle ``email-folder delete`` subcommand."""
    delimiter = _get_delimiter(mb)
    normalized = _normalize_folder_path(args.name, delimiter)
    if not mb.folder.exists(normalized):
        sys.stderr.write(f'Error: Folder "{args.name}" does not exist.\n')
        sys.exit(1)
    if not args.force:
        status = mb.folder.status(normalized, ("MESSAGES",))
        count = status.get("MESSAGES", 0)
        if count > 0:
            sys.stderr.write(
                f'Error: Folder "{args.name}" has {count} messages. Move or delete messages first, or use --force.\n'
            )
            sys.exit(1)
    mb.folder.delete(normalized)
    sys.stderr.write(f'Deleted folder "{args.name}"\n')


def _folder_exists(mb: MailBox, args: argparse.Namespace) -> None:
    """Handle ``email-folder exists`` subcommand."""
    delimiter = _get_delimiter(mb)
    normalized = _normalize_folder_path(args.name, delimiter)
    if mb.folder.exists(normalized):
        sys.stderr.write(f'Folder "{args.name}" exists\n')
    else:
        sys.stderr.write(f'Folder "{args.name}" not found\n')
        sys.exit(1)


def main_folder() -> None:
    """Entry point for email-folder command.

    Manage mailbox folders: list, create, rename, delete, check existence.
    """
    parser = argparse.ArgumentParser(
        prog="email-folder",
        description="Manage mailbox folders.",
        epilog="""\
SUBCOMMANDS:
  list    List folders with message counts
  create  Create a new folder (with intermediate parents)
  rename  Rename a folder
  delete  Delete a folder (guards against non-empty)
  exists  Check if a folder exists (exit code 0/1)

ENVIRONMENT:
  IMAP_HOST      Required. IMAP server hostname.
  IMAP_USER      Required. IMAP username / email address.
  IMAP_PASSWORD  Required. IMAP password or app password.
  IMAP_PORT      Optional. IMAP port (default: 993).

EXAMPLES:
  email-folder list
  email-folder list --json
  email-folder create "Work/Projects/2026"
  email-folder rename "Old Name" "New Name"
  email-folder delete "Empty Folder"
  email-folder delete "Non-Empty" --force
  email-folder exists "INBOX"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # list
    sp_list = subparsers.add_parser("list", help="list folders with message counts")
    sp_list.add_argument("--json", action="store_true", help="output as JSON array")

    # create
    sp_create = subparsers.add_parser("create", help="create a new folder")
    sp_create.add_argument("name", help="folder name/path to create")

    # rename
    sp_rename = subparsers.add_parser("rename", help="rename a folder")
    sp_rename.add_argument("old_name", help="current folder name")
    sp_rename.add_argument("new_name", help="new folder name")

    # delete
    sp_delete = subparsers.add_parser("delete", help="delete a folder")
    sp_delete.add_argument("name", help="folder name to delete")
    sp_delete.add_argument("--force", action="store_true", help="delete even if folder contains messages")

    # exists
    sp_exists = subparsers.add_parser("exists", help="check if a folder exists")
    sp_exists.add_argument("name", help="folder name to check")

    args = parser.parse_args()

    dispatch = {
        "list": _folder_list,
        "create": _folder_create,
        "rename": _folder_rename,
        "delete": _folder_delete,
        "exists": _folder_exists,
    }

    try:
        mb = _get_mailbox()
        with mb.login(os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"], initial_folder="INBOX"):
            dispatch[args.subcommand](mb, args)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        sys.exit(1)
    except EmailError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except UnexpectedCommandStatusError as e:
        sys.stderr.write(f"IMAP error: {e}\n")
        sys.exit(1)
    except (OSError, ConnectionRefusedError) as e:
        sys.stderr.write(f"Connection error: {e}\n")
        sys.exit(1)


# =============================================================================
# Entry Point: email-list
# =============================================================================


def main_list() -> None:
    """Entry point for email-list command.

    List and search emails in a folder.
    """
    parser = argparse.ArgumentParser(
        prog="email-list",
        description="List and search emails in an IMAP folder.",
        epilog="""\
OUTPUT FORMAT:
  Plain text (default): one block per email:
    UID: 12345
    Date: 2026-01-29 08:30
    From: john@example.com
    Subject: Invoice Q1
    Flags: \\Seen \\Flagged

    > First 200 characters of body preview if --preview 200...

    ---

  JSON (with --json): array of objects with same fields.

FILTER BEHAVIOR:
  All filter flags combine with AND.
  --query is mutually exclusive with all other filter flags.
  --has-attachment is a post-fetch client-side filter (slower).

COST:
  Default fetch does NOT mark emails as read (mark_seen=False).

ENVIRONMENT:
  IMAP_HOST      Required. IMAP server hostname.
  IMAP_USER      Required. IMAP username / email address.
  IMAP_PASSWORD  Required. IMAP password or app password.
  IMAP_PORT      Optional. IMAP port (default: 993).

EXAMPLES:
  # List 20 most recent emails
  email-list

  # Search by sender and date
  email-list --from john@example.com --since 2026-01-01

  # Unread emails with body preview
  email-list --unseen --preview 200

  # Raw IMAP search
  email-list --query "SUBJECT invoice SINCE 01-Jan-2026"

  # JSON output
  email-list --json --limit 10
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--folder", default="INBOX", help="folder to list (default: INBOX)")
    parser.add_argument("--limit", type=int, default=20, help="maximum emails to return (default: 20)")
    parser.add_argument("--preview", type=int, default=0, help="characters of body to include (default: 0)")
    parser.add_argument("--from", dest="from_filter", metavar="SENDER", help="filter by sender (substring)")
    parser.add_argument("--subject", help="filter by subject (substring)")
    parser.add_argument("--body", help="filter by body text (substring)")
    parser.add_argument("--since", type=_parse_date, metavar="DATE", help="emails on or after this date (YYYY-MM-DD)")
    parser.add_argument("--before", type=_parse_date, metavar="DATE", help="emails before this date (YYYY-MM-DD)")
    parser.add_argument("--unseen", action="store_true", help="only unread emails")
    parser.add_argument(
        "--has-attachment", action="store_true", help="only emails with attachments (client-side filter)"
    )
    parser.add_argument("--query", help="raw IMAP search criteria (mutually exclusive with filter flags)")
    parser.add_argument("--count", action="store_true", help="show total count (extra IMAP call)")
    parser.add_argument("--json", action="store_true", help="output as JSON array")

    args = parser.parse_args()

    # Validate mutual exclusivity
    filter_flags = [args.from_filter, args.subject, args.body, args.since, args.before, args.unseen]
    if args.query and any(filter_flags):
        parser.error(
            "--query is mutually exclusive with filter flags (--from, --subject, --body, --since, --before, --unseen)"
        )

    try:
        mb = _get_mailbox()
        with mb.login(os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"], initial_folder=args.folder):
            criteria = _build_criteria(args)
            use_gmail_raw = _criteria_has_non_ascii(criteria)

            if use_gmail_raw and not _is_gmail():
                sys.stderr.write(
                    "Error: Non-ASCII search terms require Gmail (X-GM-RAW). "
                    "Standard IMAP does not support UTF-8 search on most servers.\n"
                )
                sys.exit(1)

            if use_gmail_raw:
                messages, total_str = _fetch_via_gmail_raw(mb, args)
            else:
                total_str = ""
                if args.count:
                    all_uids = mb.uids(criteria)
                    total_str = f" of {len(all_uids)}"

                messages = list(mb.fetch(criteria, limit=args.limit, mark_seen=False, reverse=True))

            if args.has_attachment:
                messages = [m for m in messages if len(m.attachments) > 0]

            sys.stderr.write(f"Showing {len(messages)}{total_str} messages in {args.folder}\n")

            if args.json:
                json_output = [_email_block_to_dict(m, args.preview) for m in messages]
                print(json.dumps(json_output, indent=2, ensure_ascii=False))
            else:
                for m in messages:
                    print(_format_email_block(m, args.preview))
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        sys.exit(1)
    except EmailError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except UnexpectedCommandStatusError as e:
        sys.stderr.write(f"IMAP error: {e}\n")
        sys.exit(1)
    except (OSError, ConnectionRefusedError) as e:
        sys.stderr.write(f"Connection error: {e}\n")
        sys.exit(1)


# =============================================================================
# Email formatting helpers
# =============================================================================


def _format_body(msg: MailMessage) -> str:
    """Return plain text body, converting HTML if needed."""
    if msg.text:
        return msg.text

    if msg.html:
        converter = html2text.HTML2Text()
        converter.body_width = 0
        converter.ignore_images = True
        converter.ignore_links = False
        return converter.handle(msg.html)

    return ""


def _yaml_escape(value: str) -> str:
    """Escape a string for use in double-quoted YAML values."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


def _format_email_full(msg: MailMessage) -> str:
    """Format email as YAML frontmatter + body for ``email-read`` markdown output."""
    date_str = msg.date.isoformat() if msg.date.year > 1900 else "Unknown"
    from_val = _yaml_escape(msg.from_values.full if msg.from_values else msg.from_)
    to_val = _yaml_escape(", ".join(addr.full for addr in msg.to_values) if msg.to_values else "")
    cc_val = _yaml_escape(", ".join(addr.full for addr in msg.cc_values) if msg.cc_values else "")
    subject_val = _yaml_escape(msg.subject or "")
    flags_str = ", ".join(msg.flags) if msg.flags else ""

    frontmatter_lines = [
        "---",
        f"uid: {msg.uid}",
        f'from: "{from_val}"',
        f'to: "{to_val}"',
    ]
    if cc_val:
        frontmatter_lines.append(f'cc: "{cc_val}"')
    frontmatter_lines.extend(
        [
            f'date: "{date_str}"',
            f'subject: "{subject_val}"',
            f"flags: [{flags_str}]",
            "---",
        ]
    )

    body = _format_body(msg)
    parts = ["\n".join(frontmatter_lines), "", body]

    if msg.attachments:
        parts.extend(["", _format_attachment_list(msg.attachments)])

    return "\n".join(parts)


def _format_attachment_list(attachments: list[MailAttachment]) -> str:
    """Format attachment metadata as a markdown list."""
    lines = ["## Attachments", ""]
    for att in attachments:
        size_kb = att.size / 1024
        size_str = f"{size_kb:.0f} KB" if size_kb >= 1 else f"{att.size} B"
        lines.append(f"- {att.filename} ({size_str}, {att.content_type})")
    return "\n".join(lines)


def _save_attachments(attachments: list[MailAttachment], directory: str) -> int:
    """Save attachments to a directory, handling filename conflicts."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    saved = 0
    for att in attachments:
        # Sanitize to basename to prevent path traversal via malicious MIME filenames
        filename = Path(att.filename or f"attachment_{saved}").name
        target = dir_path / filename
        counter = 1
        while target.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            target = dir_path / f"{stem}_{counter}{suffix}"
            counter += 1
        target.write_bytes(att.payload)
        saved += 1
    return saved


# =============================================================================
# Entry Point: email-read
# =============================================================================


def main_read() -> None:
    """Entry point for email-read command.

    Fetch full email content by UID.
    """
    parser = argparse.ArgumentParser(
        prog="email-read",
        description="Fetch full email content by UID.",
        epilog="""\
OUTPUT FORMAT:
  Markdown with YAML frontmatter (default):
    ---
    uid: 12345
    from: "John Doe <john@example.com>"
    to: "user@gmail.com"
    date: "2026-01-29T08:30:00"
    subject: "Invoice Q1"
    flags: [\\Seen, \\Flagged]
    ---

    Email body text here...

    ## Attachments

    - invoice.pdf (245 KB, application/pdf)

  Raw (with --raw): RFC822 message bytes to stdout.

ENVIRONMENT:
  IMAP_HOST      Required. IMAP server hostname.
  IMAP_USER      Required. IMAP username / email address.
  IMAP_PASSWORD  Required. IMAP password or app password.
  IMAP_PORT      Optional. IMAP port (default: 993).

EXAMPLES:
  # Read email by UID
  email-read 12345

  # Mark as read and save attachments
  email-read 12345 --mark-seen --save-attachments ./downloads/

  # Save to file
  email-read 12345 --output email.md

  # Raw RFC822 output
  email-read 12345 --raw
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("uid", help="email UID to fetch")
    parser.add_argument("--folder", default="INBOX", help="folder containing the email (default: INBOX)")
    parser.add_argument("--mark-seen", action="store_true", help="mark email as read when fetching")
    parser.add_argument("--save-attachments", metavar="DIR", help="directory to save attachments to")
    parser.add_argument("--output", metavar="FILE", help="output file path (default: stdout)")
    parser.add_argument("--raw", action="store_true", help="output raw RFC822 message")

    args = parser.parse_args()

    try:
        mb = _get_mailbox()
        with mb.login(os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"], initial_folder=args.folder):
            messages = list(mb.fetch(AND(uid=args.uid), mark_seen=args.mark_seen))
            if not messages:
                sys.stderr.write(f"No message found with UID {args.uid} in {args.folder}\n")
                sys.exit(1)

            msg = messages[0]

            if args.raw:
                sys.stdout.buffer.write(msg.obj.as_bytes())
                return

            output = _format_email_full(msg)

            if args.save_attachments and msg.attachments:
                saved = _save_attachments(msg.attachments, args.save_attachments)
                sys.stderr.write(f"Saved {saved} attachment(s) to {args.save_attachments}/\n")

            if args.output:
                Path(args.output).write_text(output, encoding="utf-8")
                sys.stderr.write(f"Saved to: {args.output}\n")
            else:
                print(output)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        sys.exit(1)
    except EmailError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except UnexpectedCommandStatusError as e:
        sys.stderr.write(f"IMAP error: {e}\n")
        sys.exit(1)
    except (OSError, ConnectionRefusedError) as e:
        sys.stderr.write(f"Connection error: {e}\n")
        sys.exit(1)


# =============================================================================
# Entry Point: email-flag
# =============================================================================


def main_flag() -> None:
    """Entry point for email-flag command.

    Mark emails as read/unread, starred/unstarred.
    """
    parser = argparse.ArgumentParser(
        prog="email-flag",
        description="Mark emails as read/unread, starred/unstarred.",
        epilog="""\
BEHAVIOR:
  Flags are independent: --seen --star sets both.
  --seen and --unseen are mutually exclusive.
  --star and --unstar are mutually exclusive.

ENVIRONMENT:
  IMAP_HOST      Required. IMAP server hostname.
  IMAP_USER      Required. IMAP username / email address.
  IMAP_PASSWORD  Required. IMAP password or app password.
  IMAP_PORT      Optional. IMAP port (default: 993).

EXAMPLES:
  # Mark as read
  email-flag 12345 --seen

  # Star multiple emails
  email-flag 12345 12346 12347 --star

  # Mark as unread and remove star
  email-flag 12345 --unseen --unstar
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("uids", nargs="+", help="one or more UIDs to flag")
    parser.add_argument("--folder", default="INBOX", help="folder containing the emails (default: INBOX)")
    parser.add_argument("--seen", action="store_true", help="mark as read")
    parser.add_argument("--unseen", action="store_true", help="mark as unread")
    parser.add_argument("--star", action="store_true", help="star/flag the email")
    parser.add_argument("--unstar", action="store_true", help="remove star/flag")

    args = parser.parse_args()

    if args.seen and args.unseen:
        parser.error("--seen and --unseen are mutually exclusive")
    if args.star and args.unstar:
        parser.error("--star and --unstar are mutually exclusive")
    if not any((args.seen, args.unseen, args.star, args.unstar)):
        parser.error("at least one flag operation required (--seen/--unseen/--star/--unstar)")

    try:
        mb = _get_mailbox()
        with mb.login(os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"], initial_folder=args.folder):
            operations: list[str] = []

            if args.seen:
                mb.flag(args.uids, MailMessageFlags.SEEN, True)  # noqa: FBT003  # imap-tools API requires bool positional
                operations.append(f"+{MailMessageFlags.SEEN}")
            if args.unseen:
                mb.flag(args.uids, MailMessageFlags.SEEN, False)  # noqa: FBT003  # imap-tools API requires bool positional
                operations.append(f"-{MailMessageFlags.SEEN}")
            if args.star:
                mb.flag(args.uids, MailMessageFlags.FLAGGED, True)  # noqa: FBT003  # imap-tools API requires bool positional
                operations.append(f"+{MailMessageFlags.FLAGGED}")
            if args.unstar:
                mb.flag(args.uids, MailMessageFlags.FLAGGED, False)  # noqa: FBT003  # imap-tools API requires bool positional
                operations.append(f"-{MailMessageFlags.FLAGGED}")

            sys.stderr.write(f"Flagged {len(args.uids)} message(s): {' '.join(operations)}\n")
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        sys.exit(1)
    except EmailError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except UnexpectedCommandStatusError as e:
        sys.stderr.write(f"IMAP error: {e}\n")
        sys.exit(1)
    except (OSError, ConnectionRefusedError) as e:
        sys.stderr.write(f"Connection error: {e}\n")
        sys.exit(1)


# =============================================================================
# Entry Point: email-move
# =============================================================================


def main_move() -> None:
    """Entry point for email-move command.

    Move emails to a folder.
    """
    parser = argparse.ArgumentParser(
        prog="email-move",
        description="Move emails to a folder.",
        epilog="""\
BEHAVIOR:
  Batch operation: accepts multiple UIDs.
  Uses IMAP MOVE command (or COPY + DELETE fallback).
  NEVER permanently deletes. Moving to trash is the only "delete" operation.

ENVIRONMENT:
  IMAP_HOST      Required. IMAP server hostname.
  IMAP_USER      Required. IMAP username / email address.
  IMAP_PASSWORD  Required. IMAP password or app password.
  IMAP_PORT      Optional. IMAP port (default: 993).

EXAMPLES:
  # Move to a folder
  email-move 12345 "Work/Archive"

  # Move multiple emails to trash
  email-move 12345 12346 "[Gmail]/Trash"

  # Move from a specific source folder
  email-move 12345 "[Gmail]/Trash" --folder "Work"
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("uids", nargs="+", help="one or more UIDs to move, followed by target folder")
    parser.add_argument("--folder", default="INBOX", help="source folder (default: INBOX)")

    args = parser.parse_args()

    if len(args.uids) < 2:
        parser.error("requires at least one UID and a target folder")

    # Last positional arg is the target folder, rest are UIDs
    target = args.uids[-1]
    uid_list = args.uids[:-1]

    try:
        mb = _get_mailbox()
        with mb.login(os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"], initial_folder=args.folder):
            mb.move(uid_list, target)
            sys.stderr.write(f"Moved {len(uid_list)} message(s) to {target}\n")
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        sys.exit(1)
    except EmailError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except UnexpectedCommandStatusError as e:
        sys.stderr.write(f"IMAP error: {e}\n")
        sys.exit(1)
    except (OSError, ConnectionRefusedError) as e:
        sys.stderr.write(f"Connection error: {e}\n")
        sys.exit(1)


# =============================================================================
# Reply and draft composition helpers
# =============================================================================

_LOCALE_FORMATS: Final[dict[str, str]] = {
    "en": "{month} {day}, {year} at {hour}:{minute}",
    "cs": "{day}. {month_num}. {year} v {hour}:{minute}",
}

_EN_MONTHS: Final = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _format_date_locale(dt: datetime.datetime, locale: str) -> str:
    """Format datetime per locale for reply attribution line.

    :param dt: Datetime to format.
    :param locale: Locale code (en, cs, or fallback to ISO).
    :return: Formatted date string.
    """
    fmt = _LOCALE_FORMATS.get(locale)
    if fmt is None:
        return dt.strftime("%Y-%m-%d %H:%M")

    return fmt.format(
        day=dt.day,
        month=_EN_MONTHS[dt.month - 1],
        month_num=dt.month,
        year=dt.year,
        hour=dt.hour,
        minute=f"{dt.minute:02d}",
    )


def _build_reply_body(original: MailMessage, user_body: str, locale: str) -> str:
    """Construct reply body with attribution line and quoted original.

    :param original: Original message being replied to.
    :param user_body: User's reply text.
    :param locale: Locale for date formatting.
    :return: Full reply body text.
    """
    original_body = _format_body(original)
    quoted = "\n".join(f"> {line}" for line in original_body.splitlines())

    date_str = _format_date_locale(original.date, locale)
    sender = original.from_

    attribution = f"{date_str}, {sender}:"

    parts = [user_body, "", attribution, quoted]
    return "\n".join(parts)


def _read_body_input(args: argparse.Namespace) -> str:
    """Return body from args or stdin."""
    if args.body:
        return args.body
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _collect_reply_all_cc(original: MailMessage, user: str, explicit_cc: str | None) -> str | None:
    """Build CC header for reply-all, excluding self.

    :param original: Original message.
    :param user: Current user's email address.
    :param explicit_cc: Explicit CC override from args.
    :return: CC header value or None.
    """
    user_lower = user.lower()
    recipients = [addr.full for addr in original.to_values if addr.email.lower() != user_lower]
    recipients.extend(addr.full for addr in original.cc_values if addr.email.lower() != user_lower)
    if explicit_cc:
        recipients.extend(addr.strip() for addr in explicit_cc.split(","))
    return ", ".join(recipients) if recipients else None


def _build_reply_subject(original_subject: str, override: str | None) -> str:
    """Build reply subject line, avoiding ``Re: Re:`` duplication.

    :param original_subject: Original message subject.
    :param override: Explicit subject override from args.
    :return: Subject string for reply.
    """
    if override:
        return override
    if re.match(r"^re:\s*", original_subject, re.IGNORECASE):
        return original_subject
    return f"Re: {original_subject}"


def _set_threading_headers(msg: email.message.EmailMessage, original: MailMessage) -> None:
    """Set In-Reply-To and References headers for threading.

    :param msg: Draft message being composed.
    :param original: Original message being replied to.
    """
    raw_id = original.headers.get("message-id", [""])[0]
    original_message_id = " ".join(raw_id.split())
    if original_message_id:
        msg["In-Reply-To"] = original_message_id
        raw_refs = original.headers.get("references", [""])[0]
        original_refs = " ".join(raw_refs.split())
        msg["References"] = f"{original_refs} {original_message_id}".strip()


def _compose_reply(
    mb: MailBox,
    args: argparse.Namespace,
    user: str,
    user_body: str,
    *,
    is_reply_all: bool,
) -> email.message.EmailMessage:
    """Compose a reply draft message.

    :param mb: Connected and logged-in MailBox.
    :param args: Parsed CLI args.
    :param user: Current user's email.
    :param user_body: User's reply text.
    :param is_reply_all: Whether this is a reply-all.
    :return: Composed EmailMessage.
    """
    reply_uid = args.reply_all_to_uid or args.reply_to_uid
    originals = list(mb.fetch(AND(uid=reply_uid), mark_seen=False))
    if not originals:
        sys.stderr.write(f"No message found with UID {reply_uid} in {args.folder}\n")
        sys.exit(1)
    original = originals[0]

    msg = email.message.EmailMessage()
    msg["To"] = args.to_addr or original.from_

    if is_reply_all:
        cc_val = _collect_reply_all_cc(original, user, args.cc)
        if cc_val:
            msg["Cc"] = cc_val
    elif args.cc:
        msg["Cc"] = args.cc

    msg["Subject"] = _build_reply_subject(original.subject or "", args.subject)
    _set_threading_headers(msg, original)

    body_text = _build_reply_body(original, user_body, args.locale)
    msg["From"] = user
    msg["Date"] = email.utils.formatdate(localtime=True)
    msg.set_content(body_text)
    return msg


def _compose_new_draft(args: argparse.Namespace, user: str, user_body: str) -> email.message.EmailMessage:
    """Compose a new (non-reply) draft message.

    :param args: Parsed CLI args.
    :param user: Current user's email.
    :param user_body: Email body text.
    :return: Composed EmailMessage.
    """
    msg = email.message.EmailMessage()
    msg["To"] = args.to_addr
    if args.cc:
        msg["Cc"] = args.cc
    msg["Subject"] = args.subject
    msg["From"] = user
    msg["Date"] = email.utils.formatdate(localtime=True)
    msg.set_content(user_body)
    return msg


# =============================================================================
# Attachment helpers
# =============================================================================

_MAX_ATTACHMENT_SIZE: Final = 25 * 1024 * 1024  # 25 MB


def _validate_attachments(paths: list[str], *, force: bool) -> list[Path]:
    """Validate attachment file paths and cumulative size.

    :param paths: File path strings from CLI args.
    :param force: Override the 25 MB size limit.
    :return: List of validated Path objects.
    """
    validated: list[Path] = []
    total_size = 0
    for p in paths:
        file_path = Path(p)
        if not file_path.is_file():
            sys.stderr.write(f'Error: Attachment not found or not a file: "{p}"\n')
            sys.exit(1)
        total_size += file_path.stat().st_size
        validated.append(file_path)

    if not force and total_size > _MAX_ATTACHMENT_SIZE:
        size_mb = total_size / (1024 * 1024)
        sys.stderr.write(
            f"Error: Total attachment size ({size_mb:.1f} MB) exceeds 25 MB limit. Use --force to override.\n"
        )
        sys.exit(1)

    return validated


def _attach_files(msg: email.message.EmailMessage, paths: list[Path]) -> None:
    """Attach files to an EmailMessage.

    :param msg: Email message to attach files to.
    :param paths: Validated file paths.
    """
    for file_path in paths:
        data = file_path.read_bytes()
        mime_type, _encoding = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"
        maintype, subtype = mime_type.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=file_path.name)


# =============================================================================
# Entry Point: email-draft
# =============================================================================


def main_draft() -> None:
    """Entry point for email-draft command.

    Create a draft email (with reply support and attachments).
    """
    parser = argparse.ArgumentParser(
        prog="email-draft",
        description="Create a draft email (with reply support and attachments).",
        epilog="""\
REPLY BEHAVIOR:
  --reply-all-to-uid: Reply-all (To = sender, CC = original To/CC minus self)
  --reply-to-uid: Reply to sender only
  Both set Subject, In-Reply-To, References headers automatically.

  Attribution line format: "{date}, {sender}:" followed by quoted original.

ATTACHMENTS:
  --attach is repeatable: --attach file1.pdf --attach file2.png
  MIME type auto-detected. 25 MB cumulative limit (--force to override).

DRAFT CREATION:
  Plain text body. Attachments create multipart MIME message.
  Created via IMAP APPEND to Drafts folder.
  User reviews and sends manually from email client.

ENVIRONMENT:
  IMAP_HOST      Required. IMAP server hostname.
  IMAP_USER      Required. IMAP username / email address.
  IMAP_PASSWORD  Required. IMAP password or app password.
  IMAP_PORT      Optional. IMAP port (default: 993).
  IMAP_LOCALE    Optional. Locale for date formatting (default: en).

EXAMPLES:
  # Create a new draft
  email-draft --to user@example.com --subject "Hello" --body "Hi there"

  # Draft with attachments
  email-draft --to user@example.com --subject "Files" --body "See attached" \\
    --attach report.pdf --attach data.csv

  # Reply to an email
  email-draft --reply-to-uid 12345 --body "Thanks for the info"

  # Reply-all with body from stdin
  echo "Sounds good" | email-draft --reply-all-to-uid 12345

  # Reply with CC override
  email-draft --reply-all-to-uid 12345 --cc extra@example.com --body "Adding CC"

SHELL QUOTING:
  zsh escapes ! in double quotes (history expansion). Use $'...' quoting
  for --body to avoid this and to support newlines via \\n:

    email-draft --reply-to-uid 123 --body $'Hi!\\n\\nThanks.\\n\\nRegards,\\nClaude'

  Alternatively, pipe the body via stdin:

    echo 'Message body here!' | email-draft --reply-to-uid 123
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--to", dest="to_addr", help="recipient address(es), comma-separated")
    parser.add_argument("--cc", help="CC recipient(s), comma-separated")
    parser.add_argument("--subject", help="email subject")
    parser.add_argument("--body", help="email body (plain text; reads stdin if omitted)")
    parser.add_argument("--attach", action="append", default=[], metavar="FILE", help="file to attach (repeatable)")
    parser.add_argument("--reply-all-to-uid", metavar="UID", help="UID of email to reply-all to")
    parser.add_argument("--reply-to-uid", metavar="UID", help="UID of email to reply to sender only")
    parser.add_argument("--folder", default="INBOX", help="folder of original email for replies (default: INBOX)")
    parser.add_argument(
        "--locale",
        default=os.environ.get("IMAP_LOCALE", "en"),
        help="locale for date formatting in reply attribution (default: en)",
    )
    parser.add_argument("--force", action="store_true", help="override attachment size limit")

    args = parser.parse_args()

    if args.reply_all_to_uid and args.reply_to_uid:
        parser.error("--reply-all-to-uid and --reply-to-uid are mutually exclusive")

    reply_uid = args.reply_all_to_uid or args.reply_to_uid

    if not reply_uid:
        if not args.to_addr:
            parser.error("--to is required for new drafts (not replying)")
        if not args.subject:
            parser.error("--subject is required for new drafts (not replying)")

    # Validate attachments before connecting (fail fast)
    attachment_paths: list[Path] = []
    if args.attach:
        attachment_paths = _validate_attachments(args.attach, force=args.force)

    user_body = _read_body_input(args)

    try:
        mb = _get_mailbox()
        user = os.environ["IMAP_USER"]
        with mb.login(user, os.environ["IMAP_PASSWORD"], initial_folder=args.folder):
            if reply_uid:
                msg = _compose_reply(mb, args, user, user_body, is_reply_all=bool(args.reply_all_to_uid))
            else:
                msg = _compose_new_draft(args, user, user_body)

            if attachment_paths:
                _attach_files(msg, attachment_paths)

            drafts_folder = _find_special_folder(mb, "\\Drafts")
            mb.append(msg.as_bytes(), drafts_folder, dt=datetime.datetime.now(tz=datetime.UTC))

            att_note = f" ({len(attachment_paths)} attachment(s))" if attachment_paths else ""
            sys.stderr.write(f"Draft created in {drafts_folder}{att_note}\n")
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        sys.exit(1)
    except EmailError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
    except UnexpectedCommandStatusError as e:
        sys.stderr.write(f"IMAP error: {e}\n")
        sys.exit(1)
    except (OSError, ConnectionRefusedError) as e:
        sys.stderr.write(f"Connection error: {e}\n")
        sys.exit(1)
