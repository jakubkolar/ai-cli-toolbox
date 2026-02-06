"""Tests for reddit_utils module."""

import pytest

from ai_cli_toolbox.reddit_utils import (
    Post,
    RedditError,
    _make_json_url,
    _parse_comment,
    _parse_comment_tree,
    _parse_post,
    _parse_timestamp,
    _slugify_url,
    _truncate_text,
    _validate_reddit_url,
)


class TestValidateRedditUrl:
    def test_www_reddit_com_is_valid(self):
        # Given
        url = "https://www.reddit.com/r/python/comments/abc123/post_title/"

        # When
        result = _validate_reddit_url(url)

        # Then
        assert result == "https://www.reddit.com/r/python/comments/abc123/post_title"

    def test_reddit_com_without_www_is_valid(self):
        # Given
        url = "https://reddit.com/r/python/comments/abc123/"

        # When
        result = _validate_reddit_url(url)

        # Then
        assert result == "https://reddit.com/r/python/comments/abc123"

    def test_old_reddit_com_is_valid(self):
        # Given
        url = "https://old.reddit.com/r/python/comments/abc123/"

        # When
        result = _validate_reddit_url(url)

        # Then
        assert result == "https://old.reddit.com/r/python/comments/abc123"

    def test_invalid_domain_raises_error(self):
        # Given
        url = "https://example.com/r/python/"

        # When/Then
        with pytest.raises(RedditError, match="Invalid Reddit URL"):
            _validate_reddit_url(url)

    def test_strips_trailing_slash(self):
        # Given
        url = "https://www.reddit.com/r/python/"

        # When
        result = _validate_reddit_url(url)

        # Then
        assert not result.endswith("/")


class TestMakeJsonUrl:
    def test_appends_json_suffix(self):
        # Given
        url = "https://www.reddit.com/r/python/comments/abc123"

        # When
        result = _make_json_url(url)

        # Then
        assert result == "https://www.reddit.com/r/python/comments/abc123.json"


class TestParseTimestamp:
    def test_converts_unix_to_iso(self):
        # Given
        unix_ts = 1730103699.0  # Oct 28, 2024 08:21:39 UTC

        # When
        result = _parse_timestamp(unix_ts)

        # Then
        assert result == "2024-10-28T08:21:39Z"

    def test_zero_timestamp(self):
        # Given
        unix_ts = 0.0

        # When
        result = _parse_timestamp(unix_ts)

        # Then
        assert result == "1970-01-01T00:00:00Z"


class TestParsePost:
    def test_extracts_essential_fields(self):
        # Given
        data = {
            "title": "Test Post Title",
            "author": "test_user",
            "score": 42,
            "upvote_ratio": 0.95,
            "created_utc": 1730103699.0,
            "num_comments": 10,
            "archived": True,
            "locked": False,
            "selftext": "This is the post body",
            "subreddit": "python",
            "permalink": "/r/python/comments/abc123/test_post/",
        }

        # When
        result = _parse_post(data)

        # Then
        assert result == Post(
            title="Test Post Title",
            author="test_user",
            score=42,
            upvote_ratio=0.95,
            created_at="2024-10-28T08:21:39Z",
            num_comments=10,
            archived=True,
            locked=False,
            selftext="This is the post body",
            subreddit="python",
            url="https://www.reddit.com/r/python/comments/abc123/test_post/",
        )

    def test_handles_missing_fields_with_defaults(self):
        # Given
        data = {}

        # When
        result = _parse_post(data)

        # Then
        assert result.title == ""  # noqa: PLC1901
        assert result.author == "[deleted]"
        assert result.score == 0
        assert result.selftext == ""  # noqa: PLC1901


class TestParseComment:
    def test_extracts_essential_fields(self):
        # Given
        data = {
            "author": "commenter",
            "score": 5,
            "upvote_ratio": 0.8,
            "created_utc": 1730104544.0,
            "depth": 0,
            "is_submitter": False,
            "distinguished": None,
            "edited": False,
            "body": "This is a comment",
            "replies": "",
        }

        # When
        result = _parse_comment(data, max_depth=5)

        # Then
        assert result is not None
        assert result.author == "commenter"
        assert result.score == 5
        assert result.depth == 0
        assert result.body == "This is a comment"
        assert result.edited is False
        assert result.replies == ()

    def test_filters_deleted_comment(self):
        # Given
        data = {
            "author": "[deleted]",
            "body": "[deleted]",
            "depth": 0,
        }

        # When
        result = _parse_comment(data, max_depth=5)

        # Then
        assert result is None

    def test_filters_removed_comment(self):
        # Given
        data = {
            "author": "[deleted]",
            "body": "[removed]",
            "depth": 0,
        }

        # When
        result = _parse_comment(data, max_depth=5)

        # Then
        assert result is None

    def test_respects_max_depth(self):
        # Given
        data = {
            "author": "deep_commenter",
            "body": "Deep comment",
            "depth": 6,
        }

        # When
        result = _parse_comment(data, max_depth=5)

        # Then
        assert result is None

    def test_includes_comment_at_max_depth(self):
        # Given
        data = {
            "author": "commenter",
            "body": "At max depth",
            "depth": 5,
            "replies": "",
        }

        # When
        result = _parse_comment(data, max_depth=5)

        # Then
        assert result is not None
        assert result.depth == 5

    def test_edited_true_when_timestamp_present(self):
        # Given
        data = {
            "author": "commenter",
            "body": "Edited comment",
            "depth": 0,
            "edited": 1730105000.0,
            "replies": "",
        }

        # When
        result = _parse_comment(data, max_depth=5)

        # Then
        assert result is not None
        assert result.edited is True


class TestParseCommentTree:
    def test_parses_listing_with_comments(self):
        # Given
        listing = {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "author": "user1",
                            "body": "Comment 1",
                            "depth": 0,
                            "replies": "",
                        },
                    },
                    {
                        "kind": "t1",
                        "data": {
                            "author": "user2",
                            "body": "Comment 2",
                            "depth": 0,
                            "replies": "",
                        },
                    },
                ]
            },
        }

        # When
        result = _parse_comment_tree(listing, max_depth=5)

        # Then
        assert len(result) == 2
        assert isinstance(result, tuple)
        assert result[0].author == "user1"
        assert result[1].author == "user2"

    def test_skips_more_markers(self):
        # Given
        listing = {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "author": "user1",
                            "body": "Comment 1",
                            "depth": 0,
                            "replies": "",
                        },
                    },
                    {
                        "kind": "more",
                        "data": {
                            "count": 5,
                            "children": ["abc", "def"],
                        },
                    },
                ]
            },
        }

        # When
        result = _parse_comment_tree(listing, max_depth=5)

        # Then
        assert len(result) == 1
        assert result[0].author == "user1"

    def test_handles_nested_replies(self):
        # Given
        listing = {
            "kind": "Listing",
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "author": "parent",
                            "body": "Parent comment",
                            "depth": 0,
                            "replies": {
                                "kind": "Listing",
                                "data": {
                                    "children": [
                                        {
                                            "kind": "t1",
                                            "data": {
                                                "author": "child",
                                                "body": "Child comment",
                                                "depth": 1,
                                                "replies": "",
                                            },
                                        }
                                    ]
                                },
                            },
                        },
                    }
                ]
            },
        }

        # When
        result = _parse_comment_tree(listing, max_depth=5)

        # Then
        assert len(result) == 1
        assert result[0].author == "parent"
        assert len(result[0].replies) == 1
        assert result[0].replies[0].author == "child"

    def test_returns_empty_tuple_for_non_listing(self):
        # Given
        not_a_listing = {"kind": "something_else", "data": {}}

        # When
        result = _parse_comment_tree(not_a_listing, max_depth=5)

        # Then
        assert result == ()


class TestSlugifyUrl:
    def test_creates_filename_from_post_url(self):
        # Given
        url = "https://www.reddit.com/r/webscraping/comments/1gdx19g/best_methods/"

        # When
        result = _slugify_url(url)

        # Then
        assert result.startswith("webscraping_1gdx19g_")
        assert result.endswith(".xml")

    def test_adds_hash_for_uniqueness(self):
        # Given
        url1 = "https://www.reddit.com/r/python/comments/abc123/post1/"
        url2 = "https://www.reddit.com/r/python/comments/abc123/post2/"

        # When
        result1 = _slugify_url(url1)
        result2 = _slugify_url(url2)

        # Then
        assert result1 != result2

    def test_handles_non_standard_url(self):
        # Given
        url = "https://www.reddit.com/user/someuser/"

        # When
        result = _slugify_url(url)

        # Then
        assert result.endswith(".xml")
        assert "_" in result

    def test_respects_max_length(self):
        # Given
        url = "https://www.reddit.com/r/verylongsubredditname/comments/abc123/very_long_post_title_here/"

        # When
        result = _slugify_url(url, max_length=30)

        # Then
        assert len(result) <= 30
        assert result.endswith(".xml")


class TestTruncateText:
    def test_short_text_unchanged(self):
        # Given
        text = "Short text"

        # When
        result = _truncate_text(text, 50)

        # Then
        assert result == "Short text"

    def test_long_text_truncated_with_ellipsis(self):
        # Given
        text = "This is a much longer text that needs truncation"

        # When
        result = _truncate_text(text, 20)

        # Then
        assert len(result) == 20
        assert result.endswith("...")
        assert result == "This is a much lo..."

    def test_exact_length_unchanged(self):
        # Given
        text = "Exactly ten"

        # When
        result = _truncate_text(text, 11)

        # Then
        assert result == "Exactly ten"
