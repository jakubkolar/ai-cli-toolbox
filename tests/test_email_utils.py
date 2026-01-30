"""Tests for email_utils module."""

import argparse
import datetime
import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from imap_tools import AND
from imap_tools.folder import FolderInfo

from ai_cli_toolbox.email_utils import (
    _build_criteria,
    _email_block_to_dict,
    _find_special_folder,
    _format_attachment_list,
    _format_body,
    _format_email_block,
    _format_email_full,
    _parse_date,
    main_flag,
    main_folders,
    main_list,
    main_move,
    main_read,
)

# =============================================================================
# Fixtures
# =============================================================================

_DEFAULT_DATE = datetime.datetime(2026, 1, 29, 8, 30, tzinfo=datetime.UTC)


@dataclass
class _MessageParams:
    uid: str = "100"
    from_: str = "sender@example.com"
    subject: str = "Test Subject"
    text: str = "Hello world"
    flags: tuple[str, ...] = ("\\Seen",)
    date: datetime.datetime | None = None
    attachments: list[MagicMock] | None = None


def _make_message(**kwargs: object) -> MagicMock:
    params = _MessageParams(**kwargs)  # type: ignore[arg-type]
    msg = MagicMock()
    msg.uid = params.uid
    msg.from_ = params.from_
    msg.subject = params.subject
    msg.text = params.text
    msg.flags = params.flags
    msg.date = params.date or _DEFAULT_DATE
    msg.attachments = params.attachments or []
    return msg


def _make_folder_info(name: str, flags: tuple[str, ...] = ()) -> FolderInfo:
    return FolderInfo(name=name, delim="/", flags=flags)


def _setup_mock_mailbox() -> MagicMock:
    mock_mb = MagicMock()
    mock_mb.login.return_value = mock_mb
    mock_mb.__enter__ = MagicMock(return_value=mock_mb)
    mock_mb.__exit__ = MagicMock(return_value=False)
    return mock_mb


# =============================================================================
# _parse_date
# =============================================================================


class TestParseDate:
    def test_valid_iso_date(self):
        # When
        result = _parse_date("2026-01-29")

        # Then
        assert result == datetime.date(2026, 1, 29)

    def test_invalid_date_raises_argument_type_error(self):
        # When / Then
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid date format"):
            _parse_date("not-a-date")


# =============================================================================
# _build_criteria
# =============================================================================


def _criteria_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "query": None,
        "from_filter": None,
        "subject": None,
        "body": None,
        "since": None,
        "before": None,
        "unseen": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildCriteria:
    def test_raw_query_passthrough(self):
        # When
        result = _build_criteria(_criteria_args(query="SUBJECT invoice"))

        # Then
        assert result == "SUBJECT invoice"

    def test_no_filters_returns_all(self):
        # When
        result = _build_criteria(_criteria_args())

        # Then
        assert result == "ALL"

    def test_from_filter_builds_and_query(self):
        # When
        result = _build_criteria(_criteria_args(from_filter="john@example.com"))

        # Then
        assert isinstance(result, AND)
        assert "FROM" in str(result)

    def test_unseen_filter(self):
        # When
        result = _build_criteria(_criteria_args(unseen=True))

        # Then
        assert isinstance(result, AND)
        assert "UNSEEN" in str(result)

    def test_combined_filters(self):
        # When
        result = _build_criteria(
            _criteria_args(
                from_filter="john",
                subject="invoice",
                since=datetime.date(2026, 1, 1),
                unseen=True,
            )
        )

        # Then
        criteria_str = str(result)
        assert "FROM" in criteria_str
        assert "SUBJECT" in criteria_str
        assert "SINCE" in criteria_str
        assert "UNSEEN" in criteria_str


# =============================================================================
# _find_special_folder
# =============================================================================


class TestFindSpecialFolder:
    def test_finds_folder_by_rfc6154_attribute(self):
        # Given
        mb = MagicMock()
        mb.folder.list.return_value = [
            _make_folder_info("INBOX"),
            _make_folder_info("[Gmail]/Drafts", flags=("\\HasNoChildren", "\\Drafts")),
        ]

        # When
        result = _find_special_folder(mb, "\\Drafts")

        # Then
        assert result == "[Gmail]/Drafts"

    def test_falls_back_to_well_known_name(self):
        # Given
        mb = MagicMock()
        mb.folder.list.return_value = [
            _make_folder_info("INBOX"),
            _make_folder_info("[Gmail]/Drafts"),
        ]

        # When
        result = _find_special_folder(mb, "\\Drafts")

        # Then
        assert result == "[Gmail]/Drafts"

    def test_returns_inbox_when_not_found(self):
        # Given
        mb = MagicMock()
        mb.folder.list.return_value = [
            _make_folder_info("INBOX"),
        ]

        # When
        result = _find_special_folder(mb, "\\Drafts")

        # Then
        assert result == "INBOX"


# =============================================================================
# _format_email_block
# =============================================================================


class TestFormatEmailBlock:
    def test_basic_format_no_preview(self):
        # Given
        msg = _make_message(uid="12345", from_="john@example.com", subject="Invoice Q1")

        # When
        result = _format_email_block(msg, preview=0)

        # Then
        assert "UID: 12345" in result
        assert "From: john@example.com" in result
        assert "Subject: Invoice Q1" in result
        assert "---" in result
        assert ">" not in result

    def test_format_with_preview(self):
        # Given
        msg = _make_message(text="Hello world, this is a test email body.")

        # When
        result = _format_email_block(msg, preview=10)

        # Then
        assert "> Hello worl..." in result

    def test_format_with_flags(self):
        # Given
        msg = _make_message(flags=("\\Seen", "\\Flagged"))

        # When
        result = _format_email_block(msg, preview=0)

        # Then
        assert "Flags: \\Seen \\Flagged" in result


# =============================================================================
# _email_block_to_dict
# =============================================================================


class TestEmailBlockToDict:
    def test_basic_dict_no_preview(self):
        # Given
        msg = _make_message(uid="12345", from_="john@example.com", subject="Test")

        # When
        result = _email_block_to_dict(msg, preview=0)

        # Then
        assert result == {
            "uid": "12345",
            "date": "2026-01-29T08:30:00+00:00",
            "from": "john@example.com",
            "subject": "Test",
            "flags": ["\\Seen"],
        }

    def test_dict_with_preview(self):
        # Given
        msg = _make_message(text="Hello world")

        # When
        result = _email_block_to_dict(msg, preview=5)

        # Then
        assert result["preview"] == "Hello"


# =============================================================================
# main_folders (integration-level with mocks)
# =============================================================================

_ENV_CREDS = {"IMAP_USER": "user@test.com", "IMAP_PASSWORD": "pass"}


class TestMainFolders:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_folders_text_output(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.status.return_value = {"MESSAGES": 42, "UNSEEN": 5}

        # When
        with patch("sys.argv", ["email-folders"]):
            main_folders()

        # Then
        captured = capsys.readouterr()
        assert "INBOX" in captured.out
        assert "42 messages" in captured.out
        assert "5 unseen" in captured.out

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_folders_json_output(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.status.return_value = {"MESSAGES": 10, "UNSEEN": 2}

        # When
        with patch("sys.argv", ["email-folders", "--json"]):
            main_folders()

        # Then
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data == [{"name": "INBOX", "messages": 10, "unseen": 2}]


# =============================================================================
# main_list (integration-level with mocks)
# =============================================================================


class TestMainList:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_list_text_output(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        msg = _make_message(uid="999", from_="test@example.com", subject="Hello")
        mock_mb.fetch.return_value = [msg]

        # When
        with patch("sys.argv", ["email-list"]):
            main_list()

        # Then
        captured = capsys.readouterr()
        assert "UID: 999" in captured.out
        assert "From: test@example.com" in captured.out
        assert "Showing 1 messages in INBOX" in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_list_has_attachment_filter(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        msg_with_att = _make_message(uid="1", attachments=[MagicMock()])
        msg_without_att = _make_message(uid="2", attachments=[])
        mock_mb.fetch.return_value = [msg_with_att, msg_without_att]

        # When
        with patch("sys.argv", ["email-list", "--has-attachment"]):
            main_list()

        # Then
        captured = capsys.readouterr()
        assert "UID: 1" in captured.out
        assert "UID: 2" not in captured.out


# =============================================================================
# Changeset B: _format_body, _format_email_full, _format_attachment_list
# =============================================================================


def _make_full_message(**kwargs: object) -> MagicMock:
    msg = _make_message(**kwargs)
    msg.from_values = MagicMock()
    msg.from_values.full = "Sender <sender@example.com>"
    msg.to_values = [MagicMock(full="Recipient <recipient@example.com>")]
    msg.cc_values = []
    msg.html = ""
    return msg


class TestFormatBody:
    def test_returns_text_when_available(self):
        # Given
        msg = MagicMock()
        msg.text = "Plain text body"
        msg.html = "<p>HTML body</p>"

        # When
        result = _format_body(msg)

        # Then
        assert result == "Plain text body"

    def test_converts_html_when_no_text(self):
        # Given
        msg = MagicMock()
        msg.text = ""
        msg.html = "<p>HTML body</p>"

        # When
        result = _format_body(msg)

        # Then
        assert "HTML body" in result

    def test_returns_empty_when_no_content(self):
        # Given
        msg = MagicMock()
        msg.text = ""
        msg.html = ""

        # When
        result = _format_body(msg)

        # Then
        assert not result


class TestFormatEmailFull:
    def test_basic_markdown_output(self):
        # Given
        msg = _make_full_message(uid="12345", subject="Test Email", text="Hello world")

        # When
        result = _format_email_full(msg)

        # Then
        assert "uid: 12345" in result
        assert 'subject: "Test Email"' in result
        assert "Hello world" in result
        assert "---" in result

    def test_includes_cc_when_present(self):
        # Given
        msg = _make_full_message()
        msg.cc_values = [MagicMock(full="cc@example.com")]

        # When
        result = _format_email_full(msg)

        # Then
        assert 'cc: "cc@example.com"' in result

    def test_includes_attachment_list(self):
        # Given
        att = MagicMock()
        att.filename = "doc.pdf"
        att.size = 10240
        att.content_type = "application/pdf"
        msg = _make_full_message(attachments=[att])

        # When
        result = _format_email_full(msg)

        # Then
        assert "## Attachments" in result
        assert "doc.pdf" in result


class TestFormatAttachmentList:
    def test_formats_attachment_metadata(self):
        # Given
        att = MagicMock()
        att.filename = "report.pdf"
        att.size = 2048
        att.content_type = "application/pdf"

        # When
        result = _format_attachment_list([att])

        # Then
        assert "## Attachments" in result
        assert "- report.pdf (2 KB, application/pdf)" in result

    def test_small_attachment_shows_bytes(self):
        # Given
        att = MagicMock()
        att.filename = "tiny.txt"
        att.size = 500
        att.content_type = "text/plain"

        # When
        result = _format_attachment_list([att])

        # Then
        assert "- tiny.txt (500 B, text/plain)" in result


# =============================================================================
# main_read
# =============================================================================


class TestMainRead:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_read_text_output(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        msg = _make_full_message(uid="555", subject="Test Read")
        mock_mb.fetch.return_value = [msg]

        # When
        with patch("sys.argv", ["email-read", "555"]):
            main_read()

        # Then
        captured = capsys.readouterr()
        assert "uid: 555" in captured.out
        assert 'subject: "Test Read"' in captured.out

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_read_not_found_exits(self, mock_get_mb: MagicMock):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.fetch.return_value = []

        # When / Then
        with patch("sys.argv", ["email-read", "999"]), pytest.raises(SystemExit):
            main_read()


# =============================================================================
# main_flag
# =============================================================================


class TestMainFlag:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_flag_seen(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb

        # When
        with patch("sys.argv", ["email-flag", "123", "--seen"]):
            main_flag()

        # Then
        mock_mb.flag.assert_called_once()
        captured = capsys.readouterr()
        assert "Flagged 1 message(s)" in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_flag_seen_and_star(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb

        # When
        with patch("sys.argv", ["email-flag", "123", "--seen", "--star"]):
            main_flag()

        # Then
        assert mock_mb.flag.call_count == 2
        captured = capsys.readouterr()
        assert "+\\Seen" in captured.err
        assert "+\\Flagged" in captured.err


# =============================================================================
# main_move
# =============================================================================


class TestMainMove:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_move_single_uid(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb

        # When
        with patch("sys.argv", ["email-move", "123", "[Gmail]/Trash"]):
            main_move()

        # Then
        mock_mb.move.assert_called_once_with(["123"], "[Gmail]/Trash")
        captured = capsys.readouterr()
        assert "Moved 1 message(s) to [Gmail]/Trash" in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_move_multiple_uids(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb

        # When
        with patch("sys.argv", ["email-move", "1", "2", "3", "Archive"]):
            main_move()

        # Then
        mock_mb.move.assert_called_once_with(["1", "2", "3"], "Archive")
        captured = capsys.readouterr()
        assert "Moved 3 message(s) to Archive" in captured.err
