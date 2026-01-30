"""Email CLI utilities for token-efficient IMAP email management.

Six CLI commands for AI-agent-friendly email operations:
- email-list: List/search emails in a folder
- email-read: Fetch full email content by UID
- email-move: Move emails to a folder
- email-flag: Mark emails as read/unread, starred/unstarred
- email-draft: Create a draft email (with reply support)
- email-folders: List available folders with counts
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import html2text
from dotenv import load_dotenv
from imap_tools import AND, MailBox, MailMessageFlags
from imap_tools.errors import UnexpectedCommandStatusError
from imap_tools.message import MailAttachment, MailMessage

if TYPE_CHECKING:
    from imap_tools.folder import FolderInfo


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
        return AND(**kwargs)  # type: ignore[arg-type]
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
# Entry Point: email-folders
# =============================================================================


def main_folders() -> None:
    """Entry point for email-folders command.

    List available IMAP folders with message counts.
    """
    parser = argparse.ArgumentParser(
        prog="email-folders",
        description="List available email folders with message counts.",
        epilog="""\
OUTPUT FORMAT:
  Plain text (default): one folder per line with message counts
    INBOX                          142 messages, 5 unseen
    [Gmail]/Sent Mail              891 messages, 0 unseen

  JSON (with --json): array of objects with name, messages, unseen fields.

ENVIRONMENT:
  IMAP_HOST      Required. IMAP server hostname.
  IMAP_USER      Required. IMAP username / email address.
  IMAP_PASSWORD  Required. IMAP password or app password.
  IMAP_PORT      Optional. IMAP port (default: 993).

EXAMPLES:
  # List all folders
  email-folders

  # JSON output for programmatic use
  email-folders --json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output as JSON array instead of plain text",
    )

    args = parser.parse_args()

    try:
        mb = _get_mailbox()
        with mb.login(os.environ["IMAP_USER"], os.environ["IMAP_PASSWORD"], initial_folder="INBOX"):
            folders: list[FolderInfo] = mb.folder.list()
            folder_data: list[dict[str, object]] = []
            for f in folders:
                status = mb.folder.status(f.name, ("MESSAGES", "UNSEEN"))
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
                # Calculate column width for alignment
                max_name_len = max((len(str(fd["name"])) for fd in folder_data), default=10)
                for fd in folder_data:
                    name = str(fd["name"])
                    msgs = fd["messages"]
                    unseen = fd["unseen"]
                    print(f"{name:<{max_name_len}}  {msgs} messages, {unseen} unseen")
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
    except UnexpectedCommandStatusError as e:
        sys.stderr.write(f"IMAP error: {e}\n")
        sys.exit(1)
    except (OSError, ConnectionRefusedError) as e:
        sys.stderr.write(f"Connection error: {e}\n")
        sys.exit(1)


# =============================================================================
# Changeset B helpers
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


def _format_email_full(msg: MailMessage) -> str:
    """Format email as YAML frontmatter + body for ``email-read`` markdown output."""
    date_str = msg.date.isoformat() if msg.date.year > 1900 else "Unknown"
    from_val = msg.from_values.full if msg.from_values else msg.from_
    to_val = ", ".join(addr.full for addr in msg.to_values) if msg.to_values else ""
    cc_val = ", ".join(addr.full for addr in msg.cc_values) if msg.cc_values else ""
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
            f'subject: "{msg.subject}"',
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
        filename = att.filename or f"attachment_{saved}"
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
                mb.flag(args.uids, MailMessageFlags.SEEN, True)  # noqa: FBT003
                operations.append(f"+{MailMessageFlags.SEEN}")
            if args.unseen:
                mb.flag(args.uids, MailMessageFlags.SEEN, False)  # noqa: FBT003
                operations.append(f"-{MailMessageFlags.SEEN}")
            if args.star:
                mb.flag(args.uids, MailMessageFlags.FLAGGED, True)  # noqa: FBT003
                operations.append(f"+{MailMessageFlags.FLAGGED}")
            if args.unstar:
                mb.flag(args.uids, MailMessageFlags.FLAGGED, False)  # noqa: FBT003
                operations.append(f"-{MailMessageFlags.FLAGGED}")

            sys.stderr.write(f"Flagged {len(args.uids)} message(s): {' '.join(operations)}\n")
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
    except UnexpectedCommandStatusError as e:
        sys.stderr.write(f"IMAP error: {e}\n")
        sys.exit(1)
    except (OSError, ConnectionRefusedError) as e:
        sys.stderr.write(f"Connection error: {e}\n")
        sys.exit(1)
