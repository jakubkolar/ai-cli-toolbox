"""Tests for reddit_utils module."""

from xml.etree.ElementTree import tostring  # noqa: S405  # output-only XML generation in tests

import pytest

from ai_cli_toolbox.reddit_utils import (
    Comment,
    Post,
    RedditError,
    _build_xml_tree,
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
        assert result == Post(
            title="",
            author="[deleted]",
            score=0,
            upvote_ratio=0.0,
            created_at="1970-01-01T00:00:00Z",
            num_comments=0,
            archived=False,
            locked=False,
            selftext="",
            subreddit="",
            url="https://www.reddit.com",
        )


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
        assert result == Comment(
            author="commenter",
            score=5,
            upvote_ratio=0.8,
            created_at="2024-10-28T08:35:44Z",
            depth=0,
            is_submitter=False,
            distinguished=None,
            edited=False,
            body="This is a comment",
            replies=(),
        )

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
            "score": 0,
            "upvote_ratio": 1.0,
            "created_utc": 0.0,
            "body": "At max depth",
            "depth": 5,
            "is_submitter": False,
            "distinguished": None,
            "edited": False,
            "replies": "",
        }

        # When
        result = _parse_comment(data, max_depth=5)

        # Then
        assert result == Comment(
            author="commenter",
            score=0,
            upvote_ratio=1.0,
            created_at="1970-01-01T00:00:00Z",
            depth=5,
            is_submitter=False,
            distinguished=None,
            edited=False,
            body="At max depth",
            replies=(),
        )

    def test_edited_true_when_timestamp_present(self):
        # Given
        data = {
            "author": "commenter",
            "score": 0,
            "upvote_ratio": 1.0,
            "created_utc": 0.0,
            "body": "Edited comment",
            "depth": 0,
            "is_submitter": False,
            "distinguished": None,
            "edited": 1730105000.0,
            "replies": "",
        }

        # When
        result = _parse_comment(data, max_depth=5)

        # Then
        assert result == Comment(
            author="commenter",
            score=0,
            upvote_ratio=1.0,
            created_at="1970-01-01T00:00:00Z",
            depth=0,
            is_submitter=False,
            distinguished=None,
            edited=True,
            body="Edited comment",
            replies=(),
        )


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
                            "score": 0,
                            "upvote_ratio": 1.0,
                            "created_utc": 0.0,
                            "body": "Comment 1",
                            "depth": 0,
                            "is_submitter": False,
                            "distinguished": None,
                            "edited": False,
                            "replies": "",
                        },
                    },
                    {
                        "kind": "t1",
                        "data": {
                            "author": "user2",
                            "score": 0,
                            "upvote_ratio": 1.0,
                            "created_utc": 0.0,
                            "body": "Comment 2",
                            "depth": 0,
                            "is_submitter": False,
                            "distinguished": None,
                            "edited": False,
                            "replies": "",
                        },
                    },
                ]
            },
        }

        # When
        result = _parse_comment_tree(listing, max_depth=5)

        # Then
        assert result == (
            Comment(
                author="user1",
                score=0,
                upvote_ratio=1.0,
                created_at="1970-01-01T00:00:00Z",
                depth=0,
                is_submitter=False,
                distinguished=None,
                edited=False,
                body="Comment 1",
                replies=(),
            ),
            Comment(
                author="user2",
                score=0,
                upvote_ratio=1.0,
                created_at="1970-01-01T00:00:00Z",
                depth=0,
                is_submitter=False,
                distinguished=None,
                edited=False,
                body="Comment 2",
                replies=(),
            ),
        )

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
                            "score": 0,
                            "upvote_ratio": 1.0,
                            "created_utc": 0.0,
                            "body": "Comment 1",
                            "depth": 0,
                            "is_submitter": False,
                            "distinguished": None,
                            "edited": False,
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
        assert result == (
            Comment(
                author="user1",
                score=0,
                upvote_ratio=1.0,
                created_at="1970-01-01T00:00:00Z",
                depth=0,
                is_submitter=False,
                distinguished=None,
                edited=False,
                body="Comment 1",
                replies=(),
            ),
        )

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
                            "score": 0,
                            "upvote_ratio": 1.0,
                            "created_utc": 0.0,
                            "body": "Parent comment",
                            "depth": 0,
                            "is_submitter": False,
                            "distinguished": None,
                            "edited": False,
                            "replies": {
                                "kind": "Listing",
                                "data": {
                                    "children": [
                                        {
                                            "kind": "t1",
                                            "data": {
                                                "author": "child",
                                                "score": 0,
                                                "upvote_ratio": 1.0,
                                                "created_utc": 0.0,
                                                "body": "Child comment",
                                                "depth": 1,
                                                "is_submitter": False,
                                                "distinguished": None,
                                                "edited": False,
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
        assert result == (
            Comment(
                author="parent",
                score=0,
                upvote_ratio=1.0,
                created_at="1970-01-01T00:00:00Z",
                depth=0,
                is_submitter=False,
                distinguished=None,
                edited=False,
                body="Parent comment",
                replies=(
                    Comment(
                        author="child",
                        score=0,
                        upvote_ratio=1.0,
                        created_at="1970-01-01T00:00:00Z",
                        depth=1,
                        is_submitter=False,
                        distinguished=None,
                        edited=False,
                        body="Child comment",
                        replies=(),
                    ),
                ),
            ),
        )

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
        assert result == "This is a much lo..."

    def test_exact_length_unchanged(self):
        # Given
        text = "Exactly ten"

        # When
        result = _truncate_text(text, 11)

        # Then
        assert result == "Exactly ten"

    def test_length_below_three_clamped(self):
        # When
        result = _truncate_text("hello", 2)

        # Then
        assert result == "..."

    def test_length_zero_clamped(self):
        # When
        result = _truncate_text("hello", 0)

        # Then
        assert result == "..."

    def test_length_three_with_long_text(self):
        # When
        result = _truncate_text("hello", 3)

        # Then
        assert result == "..."


class TestBuildXmlTree:
    def test_builds_basic_xml_structure(self):
        # Given
        post = Post(
            title="Test Title",
            author="test_user",
            score=42,
            upvote_ratio=0.95,
            created_at="2024-10-28T08:21:39Z",
            num_comments=1,
            archived=False,
            locked=False,
            selftext="Post body text",
            subreddit="python",
            url="https://www.reddit.com/r/python/comments/abc123/test/",
        )
        comments = (
            Comment(
                author="commenter",
                score=5,
                upvote_ratio=0.8,
                created_at="2024-10-28T08:35:44Z",
                depth=0,
                is_submitter=False,
                distinguished=None,
                edited=False,
                body="A comment",
                replies=(),
            ),
        )

        # When
        root = _build_xml_tree(
            post,
            comments,
            url="https://www.reddit.com/r/python/comments/abc123/test/",
            retrieved_at="2024-10-28T10:00:00Z",
        )

        # Then
        assert root.tag == "reddit-thread"
        assert root.get("url") == "https://www.reddit.com/r/python/comments/abc123/test/"
        assert root.get("subreddit") == "python"
        assert root.get("retrieved_at") == "2024-10-28T10:00:00Z"

        post_elem = root.find("post")
        assert post_elem is not None
        assert post_elem.get("title") == "Test Title"
        assert post_elem.get("author") == "test_user"
        assert post_elem.get("score") == "42"
        assert post_elem.get("upvote_ratio") == "0.95"
        assert post_elem.get("archived") == "false"
        assert post_elem.get("locked") == "false"

        selftext_elem = post_elem.find("selftext")
        assert selftext_elem is not None
        assert selftext_elem.text == "Post body text"

        comments_elem = root.find("comments")
        assert comments_elem is not None
        comment_elems = comments_elem.findall("comment")
        assert len(comment_elems) == 1
        assert comment_elems[0].get("author") == "commenter"
        assert comment_elems[0].get("score") == "5"
        assert comment_elems[0].get("depth") == "0"

        body_elem = comment_elems[0].find("body")
        assert body_elem is not None
        assert body_elem.text == "A comment"

    def test_builds_nested_comment_replies(self):
        # Given
        post = Post(
            title="Title",
            author="user",
            score=1,
            upvote_ratio=1.0,
            created_at="1970-01-01T00:00:00Z",
            num_comments=2,
            archived=False,
            locked=False,
            selftext="",
            subreddit="test",
            url="https://www.reddit.com/r/test/comments/123/post/",
        )
        comments = (
            Comment(
                author="parent",
                score=10,
                upvote_ratio=0.9,
                created_at="1970-01-01T00:00:00Z",
                depth=0,
                is_submitter=True,
                distinguished=None,
                edited=True,
                body="Parent text",
                replies=(
                    Comment(
                        author="child",
                        score=3,
                        upvote_ratio=1.0,
                        created_at="1970-01-01T00:00:00Z",
                        depth=1,
                        is_submitter=False,
                        distinguished="moderator",
                        edited=False,
                        body="Child text",
                        replies=(),
                    ),
                ),
            ),
        )

        # When
        root = _build_xml_tree(
            post,
            comments,
            url="https://www.reddit.com/r/test/comments/123/post/",
            retrieved_at="2024-01-01T00:00:00Z",
        )

        # Then
        comment_elem = root.find("comments/comment")
        assert comment_elem is not None
        assert comment_elem.get("is_submitter") == "true"
        assert comment_elem.get("edited") == "true"

        replies_elem = comment_elem.find("replies")
        assert replies_elem is not None
        child_elem = replies_elem.find("comment")
        assert child_elem is not None
        assert child_elem.get("author") == "child"
        assert child_elem.get("depth") == "1"
        assert child_elem.get("distinguished") == "moderator"

    def test_empty_comments_produces_empty_comments_element(self):
        # Given
        post = Post(
            title="Title",
            author="user",
            score=0,
            upvote_ratio=1.0,
            created_at="1970-01-01T00:00:00Z",
            num_comments=0,
            archived=False,
            locked=False,
            selftext="",
            subreddit="test",
            url="https://www.reddit.com/r/test/comments/123/post/",
        )

        # When
        root = _build_xml_tree(
            post,
            comments=(),
            url="https://www.reddit.com/r/test/comments/123/post/",
            retrieved_at="2024-01-01T00:00:00Z",
        )

        # Then
        comments_elem = root.find("comments")
        assert comments_elem is not None
        assert len(comments_elem.findall("comment")) == 0

    def test_xml_escapes_special_characters(self):
        # Given
        post = Post(
            title='Title with <angle> & "quotes"',
            author="user",
            score=0,
            upvote_ratio=1.0,
            created_at="1970-01-01T00:00:00Z",
            num_comments=0,
            archived=False,
            locked=False,
            selftext="Body with <html> & entities",
            subreddit="test",
            url="https://www.reddit.com/r/test/comments/123/post/",
        )

        # When
        root = _build_xml_tree(
            post,
            comments=(),
            url="https://www.reddit.com/r/test/comments/123/post/",
            retrieved_at="2024-01-01T00:00:00Z",
        )

        # Then
        xml_str = tostring(root, encoding="unicode")
        assert "&lt;angle&gt;" in xml_str
        assert "&amp;" in xml_str
        assert "&lt;html&gt;" in xml_str
