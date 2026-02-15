"""Tests for email_utils module."""

import argparse
import datetime
import email.message
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from imap_tools import AND
from imap_tools.folder import FolderInfo

from ai_cli_toolbox.email_utils import (
    _attach_files,
    _build_criteria,
    _build_reply_body,
    _build_reply_subject,
    _create_folder_parents,
    _email_block_to_dict,
    _find_special_folder,
    _format_attachment_list,
    _format_body,
    _format_date_locale,
    _format_email_block,
    _format_email_full,
    _get_delimiter,
    _normalize_folder_path,
    _parse_date,
    _validate_attachments,
    _validate_uid,
    _validate_uids,
    _yaml_escape,
    main_draft,
    main_flag,
    main_folder,
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


def _make_folder_info(name: str, flags: tuple[str, ...] = (), delim: str = "/") -> FolderInfo:
    return FolderInfo(name=name, delim=delim, flags=flags)


def _setup_mock_mailbox() -> MagicMock:
    mock_mb = MagicMock()
    mock_mb.login.return_value = mock_mb
    mock_mb.__enter__ = MagicMock(return_value=mock_mb)
    mock_mb.__exit__ = MagicMock(return_value=False)
    return mock_mb


# =============================================================================
# _validate_uid / _validate_uids
# =============================================================================


class TestValidateUid:
    def test_valid_uid_returns_value(self):
        # When
        result = _validate_uid("123")

        # Then
        assert result == "123"

    def test_non_numeric_uid_exits(self):
        # When / Then
        with pytest.raises(SystemExit):
            _validate_uid("abc")

    def test_zero_uid_exits(self):
        # When / Then
        with pytest.raises(SystemExit):
            _validate_uid("0")

    def test_negative_uid_exits(self):
        # When / Then
        with pytest.raises(SystemExit):
            _validate_uid("-1")

    def test_float_uid_exits(self):
        # When / Then
        with pytest.raises(SystemExit):
            _validate_uid("1.5")


class TestValidateUids:
    def test_valid_uids_returns_list(self):
        # When
        result = _validate_uids(["1", "2", "3"])

        # Then
        assert result == ["1", "2", "3"]

    def test_invalid_uid_in_list_exits(self):
        # When / Then
        with pytest.raises(SystemExit):
            _validate_uids(["1", "abc", "3"])


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
# Folder management helpers
# =============================================================================


class TestGetDelimiter:
    def test_returns_delimiter_from_first_folder(self):
        # Given
        mb = MagicMock()
        mb.folder.list.return_value = [_make_folder_info("INBOX", delim=".")]

        # When
        result = _get_delimiter(mb)

        # Then
        assert result == "."

    def test_falls_back_to_slash_when_no_folders(self):
        # Given
        mb = MagicMock()
        mb.folder.list.return_value = []

        # When
        result = _get_delimiter(mb)

        # Then
        assert result == "/"


class TestNormalizeFolderPath:
    def test_replaces_slash_with_server_delimiter(self):
        # When
        result = _normalize_folder_path("Work/Projects/2026", ".")

        # Then
        assert result == "Work.Projects.2026"

    def test_noop_when_delimiter_is_slash(self):
        # When
        result = _normalize_folder_path("Work/Projects/2026", "/")

        # Then
        assert result == "Work/Projects/2026"

    def test_single_name_unchanged(self):
        # When
        result = _normalize_folder_path("Archive", ".")

        # Then
        assert result == "Archive"


class TestCreateFolderParents:
    def test_creates_intermediate_folders(self):
        # Given
        mb = MagicMock()
        mb.folder.exists.side_effect = [False, False, False]

        # When
        _create_folder_parents(mb, "Work.Projects.2026", ".")

        # Then
        assert mb.folder.create.call_count == 3
        mb.folder.create.assert_any_call("Work")
        mb.folder.create.assert_any_call("Work.Projects")
        mb.folder.create.assert_any_call("Work.Projects.2026")

    def test_skips_existing_parents(self):
        # Given
        mb = MagicMock()
        mb.folder.exists.side_effect = [True, False, False]

        # When
        _create_folder_parents(mb, "Work.Projects.2026", ".")

        # Then
        assert mb.folder.create.call_count == 2
        mb.folder.create.assert_any_call("Work.Projects")
        mb.folder.create.assert_any_call("Work.Projects.2026")


# =============================================================================
# main_folder (integration-level with mocks)
# =============================================================================

_ENV_CREDS = {"IMAP_USER": "user@test.com", "IMAP_PASSWORD": "pass"}


class TestMainFolderList:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_list_text_output(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.status.return_value = {"MESSAGES": 42, "UNSEEN": 5}

        # When
        with patch("sys.argv", ["email-folder", "list"]):
            main_folder()

        # Then
        captured = capsys.readouterr()
        assert "INBOX" in captured.out
        assert "42 messages" in captured.out
        assert "5 unseen" in captured.out

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_list_json_output(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.status.return_value = {"MESSAGES": 10, "UNSEEN": 2}

        # When
        with patch("sys.argv", ["email-folder", "list", "--json"]):
            main_folder()

        # Then
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data == [{"name": "INBOX", "messages": 10, "unseen": 2}]


class TestMainFolderCreate:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_create_new_folder(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.side_effect = [False, False]  # exists check + parent create

        # When
        with patch("sys.argv", ["email-folder", "create", "Archive"]):
            main_folder()

        # Then
        captured = capsys.readouterr()
        assert 'Created folder "Archive"' in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_create_existing_folder_succeeds_idempotent(
        self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]
    ):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.return_value = True

        # When
        with patch("sys.argv", ["email-folder", "create", "INBOX"]):
            main_folder()

        # Then
        captured = capsys.readouterr()
        assert "already exists" in captured.err
        assert "Error:" not in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_create_empty_name_errors(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]

        # When / Then
        with patch("sys.argv", ["email-folder", "create", ""]), pytest.raises(SystemExit):
            main_folder()

        captured = capsys.readouterr()
        assert "empty" in captured.err.lower()

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_create_whitespace_name_errors(self, mock_get_mb: MagicMock):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]

        # When / Then
        with patch("sys.argv", ["email-folder", "create", "  "]), pytest.raises(SystemExit):
            main_folder()

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_create_nested_folder_with_parents(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX", delim=".")]
        # exists check for full path: False; parent creation exists checks: False, False, False
        mock_mb.folder.exists.side_effect = [False, False, False, False]

        # When
        with patch("sys.argv", ["email-folder", "create", "Work/Projects/2026"]):
            main_folder()

        # Then
        assert mock_mb.folder.create.call_count == 3
        captured = capsys.readouterr()
        assert 'Created folder "Work/Projects/2026"' in captured.err


class TestMainFolderRename:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_rename_folder(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.side_effect = [True, False]  # old exists, new doesn't

        # When
        with patch("sys.argv", ["email-folder", "rename", "Old", "New"]):
            main_folder()

        # Then
        mock_mb.folder.rename.assert_called_once_with("Old", "New")
        captured = capsys.readouterr()
        assert 'Renamed folder "Old" to "New"' in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_rename_missing_source_errors(self, mock_get_mb: MagicMock):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.return_value = False

        # When / Then
        with patch("sys.argv", ["email-folder", "rename", "Missing", "New"]), pytest.raises(SystemExit):
            main_folder()

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_rename_target_exists_errors(self, mock_get_mb: MagicMock):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.side_effect = [True, True]  # both exist

        # When / Then
        with patch("sys.argv", ["email-folder", "rename", "Old", "Existing"]), pytest.raises(SystemExit):
            main_folder()


class TestMainFolderDelete:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_delete_empty_folder(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.return_value = True
        mock_mb.folder.status.return_value = {"MESSAGES": 0}

        # When
        with patch("sys.argv", ["email-folder", "delete", "Empty"]):
            main_folder()

        # Then
        mock_mb.folder.delete.assert_called_once_with("Empty")
        captured = capsys.readouterr()
        assert 'Deleted folder "Empty"' in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_delete_non_empty_folder_refused(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.return_value = True
        mock_mb.folder.status.return_value = {"MESSAGES": 15}

        # When / Then
        with patch("sys.argv", ["email-folder", "delete", "HasMail"]), pytest.raises(SystemExit):
            main_folder()

        captured = capsys.readouterr()
        assert "15 messages" in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_delete_non_empty_folder_with_force(self, mock_get_mb: MagicMock):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.return_value = True

        # When
        with patch("sys.argv", ["email-folder", "delete", "HasMail", "--force"]):
            main_folder()

        # Then
        mock_mb.folder.delete.assert_called_once_with("HasMail")

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_delete_missing_folder_errors(self, mock_get_mb: MagicMock):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.return_value = False

        # When / Then
        with patch("sys.argv", ["email-folder", "delete", "Missing"]), pytest.raises(SystemExit):
            main_folder()


class TestMainFolderExists:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_folder_exists_exits_zero(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.return_value = True

        # When
        with patch("sys.argv", ["email-folder", "exists", "INBOX"]):
            main_folder()

        # Then
        captured = capsys.readouterr()
        assert 'Folder "INBOX" exists' in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_folder_not_found_exits_one(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [_make_folder_info("INBOX")]
        mock_mb.folder.exists.return_value = False

        # When / Then
        with patch("sys.argv", ["email-folder", "exists", "Missing"]), pytest.raises(SystemExit) as exc_info:
            main_folder()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'Folder "Missing" not found' in captured.err


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

    def test_negative_limit_errors(self):
        # When / Then
        with patch("sys.argv", ["email-list", "--limit", "-1"]), pytest.raises(SystemExit):
            main_list()

    def test_zero_limit_errors(self):
        # When / Then
        with patch("sys.argv", ["email-list", "--limit", "0"]), pytest.raises(SystemExit):
            main_list()


# =============================================================================
# Email reading helpers
# =============================================================================


def _make_full_message(**kwargs: object) -> MagicMock:
    msg = _make_message(**kwargs)
    msg.from_values = MagicMock()
    msg.from_values.full = "Sender <sender@example.com>"
    msg.to_values = [MagicMock(full="Recipient <recipient@example.com>")]
    msg.cc_values = []
    msg.html = ""
    return msg


class TestYamlEscape:
    def test_escapes_double_quotes(self):
        # When
        result = _yaml_escape('He said "hello"')

        # Then
        assert result == 'He said \\"hello\\"'

    def test_escapes_backslashes(self):
        # When
        result = _yaml_escape("path\\to\\file")

        # Then
        assert result == "path\\\\to\\\\file"

    def test_escapes_newlines(self):
        # When
        result = _yaml_escape("line1\nline2\rline3")

        # Then
        assert result == "line1\\nline2\\rline3"

    def test_plain_string_unchanged(self):
        # When
        result = _yaml_escape("plain text")

        # Then
        assert result == "plain text"


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

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_read_invalid_uid_exits(self, mock_get_mb: MagicMock):
        # When / Then
        with patch("sys.argv", ["email-read", "abc"]), pytest.raises(SystemExit):
            main_read()

        mock_get_mb.assert_not_called()

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", _ENV_CREDS)
    def test_read_raw_with_output_writes_file(
        self, mock_get_mb: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        msg = _make_full_message(uid="123")
        msg.obj = MagicMock()
        msg.obj.as_bytes.return_value = b"raw RFC822 bytes"
        mock_mb.fetch.return_value = [msg]
        output_file = tmp_path / "raw.eml"

        # When
        with patch("sys.argv", ["email-read", "123", "--raw", "--output", str(output_file)]):
            main_read()

        # Then
        assert output_file.read_bytes() == b"raw RFC822 bytes"
        captured = capsys.readouterr()
        assert "Saved to:" in captured.err


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


# =============================================================================
# Reply and draft composition helpers
# =============================================================================


class TestFormatDateLocale:
    def test_english_locale(self):
        # Given
        dt = datetime.datetime(2026, 1, 29, 8, 28, tzinfo=datetime.UTC)

        # When
        result = _format_date_locale(dt, "en")

        # Then
        assert result == "Jan 29, 2026 at 8:28"

    def test_czech_locale(self):
        # Given
        dt = datetime.datetime(2026, 1, 29, 8, 28, tzinfo=datetime.UTC)

        # When
        result = _format_date_locale(dt, "cs")

        # Then
        assert result == "29. 1. 2026 v 8:28"

    def test_unknown_locale_falls_back_to_iso(self):
        # Given
        dt = datetime.datetime(2026, 1, 29, 8, 28, tzinfo=datetime.UTC)

        # When
        result = _format_date_locale(dt, "fr")

        # Then
        assert result == "2026-01-29 08:28"


class TestBuildReplySubject:
    def test_adds_re_prefix(self):
        # When
        result = _build_reply_subject("Original Subject", None)

        # Then
        assert result == "Re: Original Subject"

    def test_avoids_double_re_prefix(self):
        # When
        result = _build_reply_subject("Re: Already replied", None)

        # Then
        assert result == "Re: Already replied"

    def test_case_insensitive_re_detection(self):
        # When
        result = _build_reply_subject("RE: Uppercase reply", None)

        # Then
        assert result == "RE: Uppercase reply"

    def test_override_replaces_subject(self):
        # When
        result = _build_reply_subject("Original", "Custom subject")

        # Then
        assert result == "Custom subject"


class TestBuildReplyBody:
    def test_builds_reply_with_attribution_and_quoted_body(self):
        # Given
        original = MagicMock()
        original.text = "Hello, how are you?"
        original.html = ""
        original.date = datetime.datetime(2026, 1, 29, 8, 28, tzinfo=datetime.UTC)
        original.from_ = "john@example.com"

        # When
        result = _build_reply_body(original, "I'm fine, thanks!", "en")

        # Then
        assert "I'm fine, thanks!" in result
        assert "Jan 29, 2026 at 8:28, john@example.com:" in result
        assert "> Hello, how are you?" in result

    def test_sentinel_date_shows_unknown(self):
        # Given
        original = MagicMock()
        original.text = "Old message"
        original.html = ""
        original.date = datetime.datetime(1900, 1, 1, 0, 0, tzinfo=datetime.UTC)
        original.from_ = "sender@example.com"

        # When
        result = _build_reply_body(original, "Reply", "en")

        # Then
        assert "Unknown, sender@example.com:" in result

    def test_re_prefix_dedup(self):
        # Given: subject already has Re: (tested in main_draft, but verifying concept)
        original = MagicMock()
        original.text = "Test"
        original.html = ""
        original.date = datetime.datetime(2026, 1, 29, 8, 0, tzinfo=datetime.UTC)
        original.from_ = "a@b.com"

        # When
        result = _build_reply_body(original, "Reply", "en")

        # Then
        assert "Reply" in result
        assert "> Test" in result


class TestMainDraft:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", {**_ENV_CREDS, "IMAP_USER": "me@test.com"})
    def test_new_draft(self, mock_get_mb: MagicMock, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [
            _make_folder_info("[Gmail]/Drafts", flags=("\\Drafts",)),
        ]

        # When
        with patch("sys.argv", ["email-draft", "--to", "user@example.com", "--subject", "Hi", "--body", "Hello"]):
            main_draft()

        # Then
        mock_mb.append.assert_called_once()
        captured = capsys.readouterr()
        assert "Draft created in [Gmail]/Drafts" in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", {**_ENV_CREDS, "IMAP_USER": "me@test.com"})
    def test_reply_to_uid(self, mock_get_mb: MagicMock):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        original = _make_full_message(uid="100", subject="Original Subject")
        original.headers = {"message-id": ["<abc@example.com>"], "references": [""]}
        mock_mb.fetch.return_value = [original]
        mock_mb.folder.list.return_value = [
            _make_folder_info("[Gmail]/Drafts", flags=("\\Drafts",)),
        ]

        # When
        with patch("sys.argv", ["email-draft", "--reply-to-uid", "100", "--body", "Thanks"]):
            main_draft()

        # Then
        mock_mb.append.assert_called_once()
        call_args = mock_mb.append.call_args
        draft_bytes = call_args[0][0]
        assert b"Re: Original Subject" in draft_bytes
        assert b"In-Reply-To: <abc@example.com>" in draft_bytes

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", {**_ENV_CREDS, "IMAP_USER": "me@test.com"})
    def test_reply_all_excludes_self(self, mock_get_mb: MagicMock):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        original = _make_full_message(uid="200", subject="Group thread")
        original.to_values = [
            MagicMock(email="me@test.com", full="Me <me@test.com>"),
            MagicMock(email="other@test.com", full="Other <other@test.com>"),
        ]
        original.cc_values = [MagicMock(email="cc@test.com", full="CC <cc@test.com>")]
        original.headers = {"message-id": ["<xyz@example.com>"], "references": [""]}
        mock_mb.fetch.return_value = [original]
        mock_mb.folder.list.return_value = [
            _make_folder_info("[Gmail]/Drafts", flags=("\\Drafts",)),
        ]

        # When
        with patch("sys.argv", ["email-draft", "--reply-all-to-uid", "200", "--body", "Agreed"]):
            main_draft()

        # Then
        mock_mb.append.assert_called_once()
        call_args = mock_mb.append.call_args
        draft_bytes = call_args[0][0]
        # Self (me@test.com) should not be in CC
        assert b"me@test.com" not in draft_bytes.split(b"Cc:")[1].split(b"\n")[0] if b"Cc:" in draft_bytes else True
        assert b"Other <other@test.com>" in draft_bytes
        assert b"CC <cc@test.com>" in draft_bytes

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", {**_ENV_CREDS, "IMAP_USER": "me@test.com"})
    def test_reply_avoids_double_re_prefix(self, mock_get_mb: MagicMock):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        original = _make_full_message(uid="300", subject="Re: Already replied")
        original.headers = {"message-id": ["<id@test.com>"], "references": [""]}
        mock_mb.fetch.return_value = [original]
        mock_mb.folder.list.return_value = [
            _make_folder_info("[Gmail]/Drafts", flags=("\\Drafts",)),
        ]

        # When
        with patch("sys.argv", ["email-draft", "--reply-to-uid", "300", "--body", "Ok"]):
            main_draft()

        # Then
        call_args = mock_mb.append.call_args
        draft_bytes = call_args[0][0]
        # Should keep "Re: Already replied", not "Re: Re: Already replied"
        assert b"Re: Already replied" in draft_bytes
        assert b"Re: Re:" not in draft_bytes


# =============================================================================
# Attachment helpers
# =============================================================================


class TestValidateAttachments:
    def test_valid_files(self, tmp_path: Path):
        # Given
        f1 = tmp_path / "doc.pdf"
        f1.write_bytes(b"PDF content")
        f2 = tmp_path / "image.png"
        f2.write_bytes(b"PNG content")

        # When
        result = _validate_attachments([str(f1), str(f2)], force=False)

        # Then
        assert result == [f1, f2]

    def test_missing_file_exits(self, tmp_path: Path):
        # When / Then
        with pytest.raises(SystemExit):
            _validate_attachments([str(tmp_path / "nonexistent.txt")], force=False)

    def test_size_limit_exceeded_exits(self, tmp_path: Path):
        # Given
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * (26 * 1024 * 1024))

        # When / Then
        with pytest.raises(SystemExit):
            _validate_attachments([str(big_file)], force=False)

    def test_size_limit_override_with_force(self, tmp_path: Path):
        # Given
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * (26 * 1024 * 1024))

        # When
        result = _validate_attachments([str(big_file)], force=True)

        # Then
        assert result == [big_file]


class TestAttachFiles:
    def test_attaches_pdf(self, tmp_path: Path):
        # Given
        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")
        msg = email.message.EmailMessage()
        msg.set_content("Body text")

        # When
        _attach_files(msg, [pdf_file])

        # Then
        attachments = list(msg.iter_attachments())
        assert len(attachments) == 1
        assert attachments[0].get_filename() == "report.pdf"
        assert attachments[0].get_content_type() == "application/pdf"

    def test_attaches_unknown_type_as_octet_stream(self, tmp_path: Path):
        # Given
        unknown_file = tmp_path / "data.qzx"
        unknown_file.write_bytes(b"binary data")
        msg = email.message.EmailMessage()
        msg.set_content("Body text")

        # When
        _attach_files(msg, [unknown_file])

        # Then
        attachments = list(msg.iter_attachments())
        assert len(attachments) == 1
        assert attachments[0].get_content_type() == "application/octet-stream"

    def test_attaches_multiple_files(self, tmp_path: Path):
        # Given
        f1 = tmp_path / "doc.pdf"
        f1.write_bytes(b"pdf")
        f2 = tmp_path / "image.png"
        f2.write_bytes(b"png")
        msg = email.message.EmailMessage()
        msg.set_content("Body")

        # When
        _attach_files(msg, [f1, f2])

        # Then
        attachments = list(msg.iter_attachments())
        assert len(attachments) == 2


class TestMainDraftWithAttachments:
    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", {**_ENV_CREDS, "IMAP_USER": "me@test.com"})
    def test_draft_with_attachment(self, mock_get_mb: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        mock_mb.folder.list.return_value = [
            _make_folder_info("[Gmail]/Drafts", flags=("\\Drafts",)),
        ]
        att_file = tmp_path / "doc.pdf"
        att_file.write_bytes(b"PDF content here")

        # When
        with patch(
            "sys.argv",
            [
                "email-draft",
                "--to",
                "user@example.com",
                "--subject",
                "With attachment",
                "--body",
                "See attached",
                "--attach",
                str(att_file),
            ],
        ):
            main_draft()

        # Then
        mock_mb.append.assert_called_once()
        draft_bytes = mock_mb.append.call_args[0][0]
        assert b"doc.pdf" in draft_bytes
        captured = capsys.readouterr()
        assert "1 attachment(s)" in captured.err

    @patch("ai_cli_toolbox.email_utils._get_mailbox")
    @patch.dict("os.environ", {**_ENV_CREDS, "IMAP_USER": "me@test.com"})
    def test_reply_with_attachment(self, mock_get_mb: MagicMock, tmp_path: Path):
        # Given
        mock_mb = _setup_mock_mailbox()
        mock_get_mb.return_value = mock_mb
        original = _make_full_message(uid="400", subject="Thread")
        original.headers = {"message-id": ["<thread@test.com>"], "references": [""]}
        mock_mb.fetch.return_value = [original]
        mock_mb.folder.list.return_value = [
            _make_folder_info("[Gmail]/Drafts", flags=("\\Drafts",)),
        ]
        att_file = tmp_path / "data.csv"
        att_file.write_bytes(b"a,b,c")

        # When
        with patch(
            "sys.argv",
            ["email-draft", "--reply-to-uid", "400", "--body", "Here's the data", "--attach", str(att_file)],
        ):
            main_draft()

        # Then
        mock_mb.append.assert_called_once()
        draft_bytes = mock_mb.append.call_args[0][0]
        assert b"Re: Thread" in draft_bytes
        assert b"data.csv" in draft_bytes
