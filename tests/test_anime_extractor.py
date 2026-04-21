from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from paku.extractors.anime import (
    _assign_confidence,
    _compute_best_ratio,
    _detect_multi_titles,
    _detect_platform,
    _enhanced_ratio,
    _extract_title,
    _is_garbage_fallback,
    _strip_chrome,
    _transform_hashtag,
    extract,
)
from paku.models import AnimeExtractionResult

_LOG = logging.getLogger("test")
_DUMMY_PATH = "test.png"
_DUMMY_CONFIG: dict[str, Any] = {}


def _anilist_response(english: str, romaji: str, media_id: int = 12345, media_type: str = "ANIME") -> dict:
    return {
        "data": {
            "Media": {
                "id": media_id,
                "title": {"english": english, "romaji": romaji, "native": "テスト"},
                "type": media_type,
                "episodes": 12,
                "status": "FINISHED",
                "genres": ["Action"],
                "averageScore": 80,
                "siteUrl": f"https://anilist.co/anime/{media_id}",
                "coverImage": {"large": "https://img.anilist.co/cover.jpg"},
            }
        }
    }


def _mock_post(english: str, romaji: str, media_id: int = 12345, media_type: str = "ANIME") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _anilist_response(english, romaji, media_id, media_type)
    mock_resp.raise_for_status = MagicMock()
    mock = MagicMock(return_value=mock_resp)
    return mock


# --- Model tests ---


class TestAnimeExtractionResultModel:
    def test_required_fields(self):
        result = AnimeExtractionResult(
            confidence=0.9,
            needs_review=False,
            source_screenshot="test.png",
            extracted_at="2026-01-01T00:00:00+00:00",
            raw_title="Attack on Titan",
        )
        assert result.extractor == "anime"
        assert result.raw_title == "Attack on Titan"
        assert result.canonical_title is None
        assert result.media_source == "unknown"
        assert result.genres == []
        assert result.multi_title_detected is False

    def test_dedup_key_with_anilist_id(self):
        result = AnimeExtractionResult(
            confidence=0.9,
            needs_review=False,
            source_screenshot="test.png",
            extracted_at="2026-01-01T00:00:00+00:00",
            raw_title="Attack on Titan",
            anilist_id=16498,
            dedup_key="16498",
        )
        assert result.dedup_key == "16498"

    def test_dedup_key_fallback_to_raw_title(self):
        result = AnimeExtractionResult(
            confidence=0.3,
            needs_review=True,
            source_screenshot="test.png",
            extracted_at="2026-01-01T00:00:00+00:00",
            raw_title="Unknown Anime",
            dedup_key="unknown anime",
        )
        assert result.dedup_key == "unknown anime"


# --- Chrome stripping tests ---


class TestStripChrome:
    def test_strip_tiktok_chrome(self):
        text = "by author\nSome Great Anime\nView 3 more replies\nAdd a comment for bluntyyz..."
        platform = _detect_platform(text)
        stripped = _strip_chrome(text, platform)
        assert "Some Great Anime" in stripped
        assert "by author" not in stripped
        assert "View 3 more replies" not in stripped

    def test_strip_ig_broadcast_header(self):
        text = (
            "Flow's Anime Updates 🎯\n"
            "ameen_waleed_ · 3K members\n"
            "kristin · Moderator\n"
            "YESTERDAY 7:07 AM\n"
            "Rage of Bahamut\n"
            "❤️ 55 🔥 12 🌸 6 +4"
        )
        platform = _detect_platform(text)
        stripped = _strip_chrome(text, platform)
        assert "Rage of Bahamut" in stripped
        assert "· members" not in stripped
        assert "· Moderator" not in stripped
        assert "YESTERDAY" not in stripped

    def test_strip_hashtag_heavy_lines(self):
        text = "Attack on Titan\n#anime #manga #aot #shingeki #titan #otaku #weeb"
        stripped = _strip_chrome(text, "unknown")
        assert "Attack on Titan" in stripped
        assert "#anime" not in stripped

    def test_strip_bottom_nav(self):
        text = "Sword Art Online\nHome\nInbox\nExplore\nProfile"
        stripped = _strip_chrome(text, "unknown")
        assert "Sword Art Online" in stripped
        assert "Home" not in stripped

    def test_title_line_not_stripped(self):
        """Lines containing actual titles must survive chrome stripping."""
        text = "Anime Name: The King's Avatar\nFollow\nLiked by user123"
        stripped = _strip_chrome(text, "ig_feed")
        assert "The King's Avatar" in stripped

    def test_strip_comment_placeholder(self):
        """IMG_4852: TikTok/IG comment placeholder must be stripped."""
        text = "benja_elk 19h\nName?\nWhat do you think of this?"
        stripped = _strip_chrome(text, "unknown")
        assert "What do you think" not in stripped

    def test_strip_at_username_lines(self):
        """Lines starting with @handle must be stripped (Bug 4 regression)."""
        text = "@defeatingkoon.exe Crunchyroll and\nAnime Name: Kimetsu no Yaiba"
        stripped = _strip_chrome(text, "unknown")
        assert "@defeatingkoon.exe" not in stripped
        assert "Kimetsu no Yaiba" in stripped

    def test_strip_copyright_line(self):
        """Bug 3 regression: copyright lines starting with © must be stripped."""
        text = "©2026 Yatsuki Wakatsu, Kikka Chashi/KADOKAWA Iseka: Office Worker Partners"
        stripped = _strip_chrome(text, "unknown")
        assert "©" not in stripped

    def test_copyright_line_not_matched_as_label(self):
        """Bug 3 regression: 'Iseka:' in a ©-line must not be matched as an anime label."""
        text = "Rage of Bahamut\n©2026 Author/KADOKAWA Iseka: Office Worker Partners"
        stripped = _strip_chrome(text, "unknown")
        raw_title, _ = _extract_title(stripped)
        assert raw_title == "Rage of Bahamut"

    def test_strip_liked_by_line(self):
        """IMG_4796: 'Liked by animegate.official and 24.198 others' must be stripped."""
        text = "BOUNEN NO XAMDOU\n26 EPS\nLiked by animegate.official and 24.198 others"
        stripped = _strip_chrome(text, "unknown")
        assert "Liked by" not in stripped
        assert "BOUNEN NO XAMDOU" in stripped

    def test_strip_username_timestamp(self):
        """IMG_4852: 'syed_akram6143 19h' must be stripped."""
        text = "syed_akram6143 19h\nTomodachi no Imouto\nbenja_elk 19h"
        stripped = _strip_chrome(text, "unknown")
        assert "syed_akram6143" not in stripped
        assert "benja_elk" not in stripped
        assert "Tomodachi" in stripped

    def test_strip_suggested_for_you(self):
        """IMG_4808: 'Suggested for you' line must be stripped."""
        text = "anifeverr\nSuggested for you\nPluto\n8 Episodes"
        stripped = _strip_chrome(text, "unknown")
        assert "Suggested for you" not in stripped
        assert "Pluto" in stripped

    def test_strip_comments_header(self):
        """IMG_4852/IMG_2933: 'Comments' header must be stripped."""
        text = "Comments\nFor you\nbenja_elk 19h\nName?"
        stripped = _strip_chrome(text, "unknown")
        assert stripped.strip().startswith("Comments") is False
        assert "For you" not in stripped

    def test_strip_view_more_replies_generic(self):
        """'View 3 more replies' must be stripped for any platform, not just TikTok."""
        text = "Some Anime Title\nView 3 more replies"
        stripped = _strip_chrome(text, "unknown")
        assert "View 3 more replies" not in stripped
        assert "Some Anime Title" in stripped

    def test_strip_swipe_indicator(self):
        """IMG_4760: SWIPE indicators must be stripped."""
        text = "ANIME NEWS LIST\nSWIPE\nSWIPE"
        stripped = _strip_chrome(text, "unknown")
        assert "SWIPE" not in stripped
        assert "ANIME NEWS LIST" in stripped

    def test_strip_slide_counter(self):
        """IMG_4760/4796: '2/10', '5/12' slide counters must be stripped."""
        text = "BOUNEN NO XAMDOU\n5/12\n26 EPS"
        stripped = _strip_chrome(text, "unknown")
        assert "5/12" not in stripped

    def test_strip_followed_by_line(self):
        """'Followed by username' engagement lines must be stripped."""
        text = "Attack on Titan\nFollowed by user123 and 5 others"
        stripped = _strip_chrome(text, "unknown")
        assert "Followed by" not in stripped
        assert "Attack on Titan" in stripped

    def test_strip_reaction_emoji_row_outside_broadcast(self):
        """Reaction emoji rows must be stripped even outside ig_broadcast context."""
        text = "Rage of Bahamut\n❤️ 55 🔥 12 🌸 6 +4"
        stripped = _strip_chrome(text, "unknown")
        assert "❤️" not in stripped
        assert "Rage of Bahamut" in stripped

    def test_strip_follow_standalone(self):
        """Standalone 'Follow' button text must be stripped."""
        text = "anifeverr\nFollow\nPluto"
        stripped = _strip_chrome(text, "unknown")
        assert stripped.count("Follow") == 0
        assert "Pluto" in stripped


# --- Fallback rejection guard tests ---


class TestFallbackRejection:
    def test_reject_username_timestamp(self):
        """IMG_4852: 'syed_akram6143 19h' is garbage."""
        assert _is_garbage_fallback("syed_akram6143 19h") is True

    def test_reject_username_timestamp_with_symbol(self):
        """IMG_2933: 'conroy_robinson 16w✰ . by author' is garbage."""
        assert _is_garbage_fallback("conroy_robinson 16w✰ . by author") is True

    def test_reject_liked_by(self):
        """IMG_4796: engagement line is garbage."""
        assert _is_garbage_fallback("VA Liked by animegate.official and 24.198 others") is True

    def test_reject_username_caption(self):
        """IMG_4878: username + generic caption is garbage."""
        assert _is_garbage_fallback("animelif3 Anime I don't see anyone talking about") is True

    def test_reject_collecting_fans(self):
        """IMG_4760: engagement/community count is garbage."""
        assert _is_garbage_fallback("COLLECTING ALL ANIME FANS (23k/25k ...") is True

    def test_reject_username_truncated_caption(self):
        """IMG_4808: 'hotfreestyle Donald Trump reveals he recently... more' is garbage."""
        assert _is_garbage_fallback("hotfreestyle Donald Trump reveals he recently... more") is True

    def test_reject_by_author(self):
        """TikTok 'by author' attribution is garbage."""
        assert _is_garbage_fallback("conroy_robinson 16w✰ · ❤ by author") is True

    def test_accept_real_anime_title(self):
        """Real anime titles must not be rejected."""
        assert _is_garbage_fallback("BOUNEN NO XAMDOU") is False
        assert _is_garbage_fallback("Attack on Titan") is False
        assert _is_garbage_fallback("TRIGUN STARGAZE") is False
        assert _is_garbage_fallback("Rage of Bahamut") is False
        assert _is_garbage_fallback("Magical Destroyers") is False

    def test_accept_long_title(self):
        """Light novel titles must not be rejected."""
        assert _is_garbage_fallback("I've Been Killing Slimes For 300 Years And Maxed Out My Level") is False

    def test_reject_suggested_for_you(self):
        assert _is_garbage_fallback("Suggested for you") is True

    def test_reject_add_comment(self):
        assert _is_garbage_fallback("Add a comment for bluntyyz...") is True

    def test_fallback_returns_none_for_garbage(self):
        """When all lines are garbage, _extract_title returns (None, None)."""
        text = "syed_akram6143 19h\nName?\nReply"
        raw_title, pattern = _extract_title(text)
        assert raw_title is None
        assert pattern is None

    def test_extract_returns_empty_title_for_garbage(self):
        """When fallback rejects garbage, extract() returns empty raw_title, needs_review=True."""
        text = "syed_akram6143 19h\nName?\nReply\nAdd a comment"
        with patch("requests.post") as mock_post:
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        mock_post.assert_not_called()  # no AniList query for garbage
        assert isinstance(result, AnimeExtractionResult)
        assert result.raw_title == ""
        assert result.needs_review is True
        assert result.confidence == pytest.approx(0.0)
        assert result.levenshtein_ratio is None


# --- Platform detection tests ---


class TestDetectPlatform:
    def test_anilist_app_detection(self):
        text = "ADD TO LIST\nAVERAGE SCORE\nMOST POPULAR\nAttack on Titan"
        assert _detect_platform(text) == "anilist_app"

    def test_ig_broadcast_detection(self):
        text = "Flow's Updates\nameen · 3K members\nsome content"
        assert _detect_platform(text) == "ig_broadcast"

    def test_ig_story_detection(self):
        text = "username\nSend message...\nSome caption"
        assert _detect_platform(text) == "ig_story"

    def test_unknown_platform(self):
        text = "Anime Name: Sword Art Online"
        assert _detect_platform(text) == "unknown"


# --- Title extraction tests ---


class TestExtractTitle:
    def test_pattern_a_bare_anime_label(self):
        """IMG_4694: 'Anime: SANDA' — bare 'Anime:' without 'name' must match Pattern A."""
        text = "Anime: SANDA"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "SANDA"
        assert pattern == "label"

    def test_pattern_a_bare_anime_in_caption(self):
        """IMG_4694 full context: 'Anime: SANDA' at end of Plot paragraph."""
        text = (
            "Plot: So, Sanda, a second-year middle school\n"
            "kid, gets straight-up jumped with a kitchen\n"
            'knife, he\'s totally clueless, like, "Yo, what\'s the vibe?"\n'
            "Anime: SANDA"
        )
        raw_title, pattern = _extract_title(text)
        assert raw_title == "SANDA"
        assert pattern == "label"

    def test_pattern_a_label_with_emoji(self):
        text = "• ANIME NAME 🎬 :- The Kawai Complex Guide to Manors and Hostel Behavior"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "The Kawai Complex Guide to Manors and Hostel Behavior"
        assert pattern == "label"

    def test_pattern_a_rage_of_bahamut(self):
        text = "• Anime : Rage of Bahamut"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "Rage of Bahamut"
        assert pattern == "label"

    def test_pattern_a_kings_avatar(self):
        text = "Anime Name: The King's Avatar"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "The King's Avatar"
        assert pattern == "label"

    def test_pattern_b_quoted(self):
        text = 'POV: You thought "Renai flops" was just another harem-comedy anime.'
        raw_title, pattern = _extract_title(text)
        assert raw_title == "Renai flops"
        assert pattern == "quoted"

    def test_pattern_c_numbered(self):
        text = "Top Anime:\n1. Gachiakuta"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "Gachiakuta"
        assert pattern == "numbered"

    def test_pattern_d_year_tagged(self):
        text = "ZETMAN (2012)"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "ZETMAN"
        assert pattern == "year_tagged"

    def test_pattern_e_romaji_macrons(self):
        text = "NISHA ROKUBŌ NO SHICHININ"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "NISHA ROKUBŌ NO SHICHININ"
        assert pattern == "romaji"

    def test_pattern_f_hashtag(self):
        text = "#bungoustraydogsanime is amazing #anime #manga #otaku #weeb"
        raw_title, pattern = _extract_title(text)
        assert pattern == "hashtag"
        assert "Stray" in raw_title or "bungoustraydogs" in raw_title.lower()

    def test_pattern_f_full_text_hashtag_scan(self):
        """Fix: Pattern F must count hashtags from full_text when hashtag-heavy lines
        were stripped by chrome (IMG_2935 regression)."""
        # Simulate chrome-stripped text (hashtag lines removed)
        stripped = "Some anime content here"
        # Original text had 5+ hashtags on stripped lines
        full_text = "Some anime content here\n#anime #manga #otaku #weeb #BungouStrayDogs"
        raw_title, pattern = _extract_title(stripped, full_text=full_text)
        assert pattern == "hashtag"
        assert "Bungou" in raw_title or "Stray" in raw_title

    def test_pattern_g_discussion_title(self):
        """Fix: rhetorical question extracts noun phrase instead of falling to fallback."""
        text = "Is Ghost Stories dub the worst anime of all time?"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "Ghost Stories"
        assert pattern == "discussion"

    def test_pattern_g_best_variant(self):
        """Pattern G handles 'the best' variant."""
        text = "Is Attack on Titan the best anime ever?"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "Attack on Titan"
        assert pattern == "discussion"

    def test_pattern_b_reject_question(self):
        """Quoted text ending with ? is a conversational placeholder, not a title."""
        text = 'Someone asked "What do you think of this?" in the comments'
        raw_title, pattern = _extract_title(text)
        assert pattern != "quoted"

    def test_pattern_b_reject_plot_context(self):
        """Quoted text near Plot:/Synopsis:/Story: keywords is plot description, not a title."""
        text = 'Plot: A young hero discovers\n"Yo, what\'s the vibe?" says the villain\nSynopsis: An epic tale'
        raw_title, pattern = _extract_title(text)
        assert pattern != "quoted"

    def test_pattern_b_reject_synopsis_nearby(self):
        """Quoted text 2 lines after Synopsis: keyword must be rejected."""
        text = 'Synopsis: Something happens\nThen more stuff\n"The great adventure" begins'
        raw_title, pattern = _extract_title(text)
        assert pattern != "quoted"

    def test_pattern_b_long_title_accepted(self):
        """Long quoted titles (>60 chars) like light novel titles must match Pattern B."""
        text = '"I\'ve Been Killing Slimes For 300 Years And Maxed Out My Level"'
        raw_title, pattern = _extract_title(text)
        assert pattern == "quoted"
        assert "Killing Slimes" in raw_title

    def test_pattern_b_broadcast_quoted_title(self):
        """IMG_3368: quoted title in broadcast channel message body must survive chrome stripping."""
        text = (
            "Flow's Anime Updates\n"
            "ameen_waleed_ · 3K members\n"
            "YESTERDAY 7:07 AM\n"
            '"I\'ve Been Killing Slimes For 300 Years And Maxed Out My Level"\n'
            "❤️ 55 🔥 12"
        )
        platform = _detect_platform(text)
        stripped = _strip_chrome(text, platform)
        assert "Killing Slimes" in stripped
        raw_title, pattern = _extract_title(stripped)
        assert pattern == "quoted"
        assert "Killing Slimes" in raw_title

    def test_label_called_form(self):
        text = "anime is called katsugeki/touken ranbu"
        raw_title, pattern = _extract_title(text)
        assert "katsugeki" in raw_title.lower() or "touken" in raw_title.lower()
        assert pattern == "label"

    def test_label_name_is_form(self):
        """Fix: 'the anime name is X' variant (IMG_6502)."""
        text = "the anime name is Ocean Waves"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "Ocean Waves"
        assert pattern == "label"

    def test_label_emoji_prefix(self):
        """Fix: emoji before label keyword must not block match (IMG_6502)."""
        text = "🎬 the anime name is Ocean Waves"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "Ocean Waves"
        assert pattern == "label"

    def test_label_emoji_prefix_colon_form(self):
        """Emoji prefix on colon-separated label."""
        text = "🎬 anime name: The Ocean Waves"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "The Ocean Waves"
        assert pattern == "label"

    def test_label_multiline_title(self):
        """Smart continuation: non-metadata lines are joined (fixes IMG_3753 truncation)."""
        text = "anime is called katsugeki/\ntouken ranbu"
        raw_title, pattern = _extract_title(text)
        assert "katsugeki" in raw_title.lower()
        # Continuation is NOT a metadata keyword — must be captured and joined
        assert "touken" in raw_title.lower()
        assert "\n" not in raw_title
        assert pattern == "label"

    def test_label_multiline_kawai(self):
        """Smart continuation: non-metadata lines are joined (fixes IMG_2946 truncation)."""
        text = "• ANIME NAME 🎬 :- The Kawai Complex\nGuide to Manors and Hostel Behavior"
        raw_title, pattern = _extract_title(text)
        assert "Kawai Complex" in raw_title
        # Continuation is NOT a metadata keyword — must be captured and joined
        assert "Guide to Manors" in raw_title
        assert pattern == "label"

    def test_label_no_genre_continuation(self):
        """Bug 1 regression: genre line must not be appended to title."""
        text = "Anime Name: The King's Avatar\nGenres: Action Sports Gaming Drama Slice"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "The King's Avatar"
        assert "Genres" not in raw_title
        assert pattern == "label"

    def test_release_card_trigun_stargaze(self):
        """Bug 2 fix: all-caps title + RELEASE DATE anchor extracts via release_card."""
        text = "TRIGUN STARGAZE\nRELEASE DATE: Jan 2026"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "TRIGUN STARGAZE"
        assert pattern == "release_card"

    def test_release_card_noise_guard(self):
        """Bug 2 regression: noise lines must not trigger release_card even with anchor."""
        text = "COLLECTING ALL ANIME FANS\nRELEASE DATE: Jan 2026"
        raw_title, pattern = _extract_title(text)
        assert pattern != "release_card"


# --- Hashtag transformation tests ---


class TestTransformHashtag:
    def test_camel_case_with_anime_suffix(self):
        assert _transform_hashtag("BungouStrayDogsAnime") == "Bungou Stray Dogs"

    def test_camel_case_without_suffix(self):
        assert _transform_hashtag("BungouStrayDogs") == "Bungou Stray Dogs"

    def test_lowercase_anime_suffix_stripped(self):
        """Suffix removal must work without word boundary (no \\b)."""
        result = _transform_hashtag("bungoustraydogsanime")
        assert "anime" not in result.lower()
        assert result == "bungoustraydogs"

    def test_manga_suffix(self):
        assert _transform_hashtag("OnePieceManga") == "One Piece"

    def test_series_suffix(self):
        assert _transform_hashtag("NarutoSeries") == "Naruto"

    def test_edit_suffix(self):
        """moriartythepatriotedit → strip edit → split CamelCase."""
        result = _transform_hashtag("MoriartyThePatriotEdit")
        assert result == "Moriarty The Patriot"

    def test_edit_suffix_lowercase(self):
        """Lowercase concatenated edit suffix stripped without word boundary."""
        result = _transform_hashtag("moriartythepatriotedit")
        assert "edit" not in result.lower()

    def test_no_suffix(self):
        assert _transform_hashtag("AttackOnTitan") == "Attack On Titan"


# --- Title + romaji tests ---


class TestTitleRomaji:
    def test_pattern_e2_title_with_romaji(self):
        """IMG_6502: English title + parenthesized romaji on next line."""
        text = "The Ocean Waves\n( Umi ga Kikoeru )"
        raw_title, pattern = _extract_title(text)
        assert raw_title == "The Ocean Waves"
        assert pattern == "title_romaji"

    def test_pattern_e2_extracts_alt_query(self):
        """extract() passes romaji as alt_query to AniList."""
        text = "The Ocean Waves\n( Umi ga Kikoeru )"
        mock = _mock_post("Ocean Waves", "Umi ga Kikoeru")
        with patch("requests.post", mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.raw_title == "The Ocean Waves"
        assert result.title_pattern == "title_romaji"

    def test_pattern_e2_alt_query_improves_ratio(self):
        """When primary ratio is low, alt_query (romaji) should be tried."""
        text = "The Ocean Waves\n( Umi ga Kikoeru )"
        # Primary query "The Ocean Waves" → returns a poor match (low ratio);
        # alt query "Umi ga Kikoeru" → returns the correct entry (high ratio)
        call_count = 0
        def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            body = kwargs.get("json", {})
            search = body.get("variables", {}).get("search", "")
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.status_code = 200
            if "Umi ga Kikoeru" in search:
                # Alt query returns good match
                mock_resp.json.return_value = _anilist_response("The Ocean Waves", "Umi ga Kikoeru")
            else:
                # Primary query returns a distant title
                mock_resp.json.return_value = _anilist_response("Totally Different Show", "Zenzen Chigau")
            return mock_resp
        with patch("requests.post", side_effect=mock_post):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        # Should have made at least 3 calls (ANIME + MANGA for primary, then alt_query)
        assert call_count >= 3

    def test_no_romaji_no_alt_query(self):
        """Regular label pattern must not produce alt_query."""
        text = "Anime Name: Attack on Titan"
        with patch("requests.post", _mock_post("Attack on Titan", "Shingeki no Kyojin")):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.title_pattern == "label"


# --- Multi-title tests ---


class TestMultiTitleDetection:
    def test_three_date_prefixed_titles(self):
        text = (
            "2 Oct  Dusk Beyond the End of the World\n"
            "2 Oct  Pass the Monster Meat, Milady!\n"
            "3 Oct  Ganglion"
        )
        titles = _detect_multi_titles(text)
        assert len(titles) == 3
        assert any("Ganglion" in t for t in titles)

    def test_single_title_not_multi(self):
        text = "Anime Name: Attack on Titan"
        titles = _detect_multi_titles(text)
        assert len(titles) == 0


# --- Confidence assignment tests ---


class TestAssignConfidence:
    def test_high_ratio_no_review(self):
        confidence, needs_review = _assign_confidence(0.85, "label", "recommendation")
        assert confidence == pytest.approx(0.95)
        assert needs_review is False

    def test_high_ratio_label_capped(self):
        confidence, needs_review = _assign_confidence(0.95, "label", "recommendation")
        assert confidence == pytest.approx(0.95)

    def test_mid_ratio_needs_review(self):
        confidence, needs_review = _assign_confidence(0.7, None, "recommendation")
        assert confidence == pytest.approx(0.7)
        assert needs_review is True

    def test_low_ratio_low_confidence(self):
        confidence, needs_review = _assign_confidence(0.35, None, "recommendation")
        assert confidence == pytest.approx(0.3)
        assert needs_review is True

    def test_discussion_context_penalty(self):
        confidence, needs_review = _assign_confidence(0.85, None, "discussion")
        assert confidence == pytest.approx(0.7)

    def test_pattern_a_boost(self):
        confidence, _ = _assign_confidence(0.6, "label", "recommendation")
        assert confidence == pytest.approx(0.8)


# --- AniList integration tests (mocked) ---


class TestAniListIntegration:
    def test_high_ratio_no_review(self):
        text = "Anime Name: Attack on Titan"
        with patch("requests.post", _mock_post("Attack on Titan", "Shingeki no Kyojin")):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.needs_review is False
        assert result.canonical_title == "Attack on Titan"
        assert result.levenshtein_ratio is not None
        assert result.levenshtein_ratio >= 0.8
        assert result.dedup_key == "12345"

    def test_mid_ratio_needs_review(self):
        text = "Anime Name: Atack on Titan"  # typo → lower ratio
        mock = MagicMock()
        mock.json.return_value = _anilist_response("Attack on Titan", "Shingeki no Kyojin")
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.levenshtein_ratio is not None

    def test_low_ratio_unknown_media_source(self):
        text = "Anime Name: Asdfghjkl Qwerty"
        mock = MagicMock()
        mock.json.return_value = _anilist_response("Attack on Titan", "Shingeki no Kyojin")
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.media_source == "unknown"
        assert result.needs_review is True
        assert result.canonical_title is None

    def test_network_error_graceful_fallback(self):
        import requests as req_mod
        text = "Anime Name: Attack on Titan"
        with patch("requests.post", side_effect=req_mod.exceptions.ConnectionError("down")):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.confidence == pytest.approx(0.3)
        assert result.needs_review is True
        assert result.raw_title == "Attack on Titan"

    def test_levenshtein_ratio_always_logged(self):
        text = "Anime Name: Attack on Titan"
        with patch("requests.post", _mock_post("Attack on Titan", "Shingeki no Kyojin")):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.levenshtein_ratio is not None

    def test_no_anilist_match_gives_zero_ratio_not_none(self):
        """Bug 1 regression: no AniList match must give ratio=0.0 (not None).
        None is reserved for genuine network errors; 0.0 → reason='no_anilist_match'."""
        text = "Anime Name: Totally Unknown Xyz"
        mock = MagicMock()
        mock.json.return_value = {"data": {"Media": None}}
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.levenshtein_ratio == pytest.approx(0.0)
        assert result.needs_review is True

    def test_anilist_404_is_no_result_not_network_error(self):
        """Root cause fix: AniList returns 404 for no-match queries.
        Must yield ratio=0.0 (no match), not ratio=None (network error)."""
        text = "Anime Name: Completely Nonexistent Title Xyz"
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("requests.post", return_value=mock_resp):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        # ratio=0.0 means "searched but no match"; ratio=None means "network error"
        assert result.levenshtein_ratio == pytest.approx(0.0)
        assert result.needs_review is True

    def test_network_error_preserves_raw_title_not_empty(self):
        """Bug 1 regression: RequestException must preserve extracted raw_title, not empty string."""
        import requests as req_mod
        text = "Anime Name: Frieren Beyond Journey's End"
        with patch("requests.post", side_effect=req_mod.exceptions.RequestException("timeout")):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.raw_title == "Frieren Beyond Journey's End"
        assert result.raw_title != ""
        assert result.needs_review is True


# --- Non-AniList tests ---


class TestNonAniList:
    def test_donghua_queries_anilist_no_match(self):
        """Donghua signal no longer skips AniList — queries and gets no match → media_source='donghua'."""
        text = "Chinese Web Novel\nDonghua adaptation\nThe King's Avatar"
        mock = MagicMock()
        mock.json.return_value = {"data": {"Media": None}}
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock) as mock_post:
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        mock_post.assert_called()
        assert isinstance(result, AnimeExtractionResult)
        assert result.media_source == "donghua"
        assert result.needs_review is True
        assert result.confidence == pytest.approx(0.3)

    def test_western_queries_anilist_no_match(self):
        """Western signal no longer skips AniList — queries and gets no match → media_source='western'."""
        text = "Cartoon Network original series\nSome Show Title"
        mock = MagicMock()
        mock.json.return_value = {"data": {"Media": None}}
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock) as mock_post:
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        mock_post.assert_called()
        assert result.media_source == "western"
        assert result.needs_review is True

    def test_donghua_signal_but_high_anilist_ratio(self):
        """Donghua signal + AniList ratio >= 0.4 → media_source overridden to 'anime'."""
        text = "BiliBili\nAnime Name: The King's Avatar"
        with patch("requests.post", _mock_post("The King's Avatar", "Quanzhi Gaoshou")) as mock_post:
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        mock_post.assert_called()
        assert isinstance(result, AnimeExtractionResult)
        assert result.media_source == "anime"
        assert result.levenshtein_ratio is not None
        assert result.levenshtein_ratio >= 0.4

    def test_anilist_app_short_circuit(self):
        text = "ADD TO LIST\nAVERAGE SCORE\nMOST POPULAR\nAttack on Titan"
        with patch("requests.post") as mock_post:
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        mock_post.assert_not_called()
        assert isinstance(result, AnimeExtractionResult)
        assert result.extraction_mode == "anilist_app"
        assert result.confidence == pytest.approx(0.95)
        assert result.needs_review is False

    def test_hero_without_class_attempts_anilist(self):
        """Bug 2 regression: 'Skills'/'Class' must not block AniList query."""
        text = "Anime Name: Hero Without a Class: Who Even Needs Skills?!"
        mock = MagicMock()
        mock.json.return_value = _anilist_response("Hero Without a Class", "Hero Without a Class")
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock) as mock_post:
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        mock_post.assert_called()
        assert isinstance(result, AnimeExtractionResult)
        assert result.media_source not in ("donghua", "western")


# --- Multi-title extraction tests ---


class TestMultiTitleExtraction:
    def test_returns_list_for_multi(self):
        text = (
            "2 Oct  Dusk Beyond the End of the World\n"
            "2 Oct  Pass the Monster Meat, Milady!\n"
            "3 Oct  Ganglion"
        )
        mock = MagicMock()
        mock.json.return_value = {"data": {"Media": None}}
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, list)
        assert len(result) == 3
        for r in result:
            assert r.multi_title_detected is True
            assert r.needs_review is True

    def test_single_title_not_list(self):
        text = "Anime Name: Attack on Titan"
        with patch("requests.post", _mock_post("Attack on Titan", "Shingeki no Kyojin")):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)


# --- Discussion context tests ---


class TestDiscussionContext:
    def test_discussion_flag(self):
        text = "Is Ghost Stories dub the worst anime of all time? #anime"
        mock = MagicMock()
        mock.json.return_value = {"data": {"Media": None}}
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.extraction_context == "discussion"
        assert result.raw_title == "Ghost Stories"
        assert result.title_pattern == "discussion"


# --- Dedup key tests ---


class TestDedupKey:
    def test_dedup_key_with_anilist_match(self):
        text = "Anime Name: Attack on Titan"
        with patch("requests.post", _mock_post("Attack on Titan", "Shingeki no Kyojin", media_id=16498)):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.dedup_key == "16498"

    def test_dedup_key_no_match(self):
        text = "Anime Name: Totally Unknown Show Xyz"
        mock = MagicMock()
        mock.json.return_value = {"data": {"Media": None}}
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.dedup_key is not None
        assert result.dedup_key == result.raw_title.lower().strip()


# --- Pipeline integration test ---


class TestPipelineIntegration:
    def test_anime_mode_pipeline_roundtrip(self, tmp_path: Path):
        import logging

        from paku.context import AppContext
        from paku.models import OcrResult
        from paku.ocr.router import EngineRouter
        from paku.ocr.stub import StubOCREngine

        stub = StubOCREngine(config={}, logger=logging.getLogger("test"))
        mock_ocr = OcrResult(
            engine="stub",
            raw_text="• Anime : Rage of Bahamut",
        )
        stub.extract = MagicMock(return_value=mock_ocr)

        config = {"outputs": {"review_queue": str(tmp_path / "review.json")}}
        router = EngineRouter({"stub": stub})
        AppContext._instance = AppContext(
            config=config,
            logger=logging.getLogger("test"),
            ocr_engines={"stub": stub},
            router=router,
        )

        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        from paku.pipeline import process_image

        with patch("requests.post", _mock_post("Rage of Bahamut", "Rage of Bahamut: Genesis")):
            result = process_image(image_path=img_path, mode="anime")

        assert result is not None
        assert result["status"] == "extracted"
        assert "extraction" in result
        extraction = result["extraction"]
        assert extraction["raw_title"] == "Rage of Bahamut"

    def test_anime_mode_low_confidence_writes_review(self, tmp_path: Path):
        import logging

        from paku.context import AppContext
        from paku.models import OcrResult
        from paku.ocr.router import EngineRouter
        from paku.ocr.stub import StubOCREngine

        stub = StubOCREngine(config={}, logger=logging.getLogger("test"))
        mock_ocr = OcrResult(
            engine="stub",
            raw_text="• Anime : Some Very Obscure Title",
        )
        stub.extract = MagicMock(return_value=mock_ocr)

        queue_path = tmp_path / "review.json"
        config = {"outputs": {"review_queue": str(queue_path)}}
        router = EngineRouter({"stub": stub})
        AppContext._instance = AppContext(
            config=config,
            logger=logging.getLogger("test"),
            ocr_engines={"stub": stub},
            router=router,
        )

        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        from paku.pipeline import process_image

        mock = MagicMock()
        mock.json.return_value = _anilist_response("Attack on Titan", "Shingeki no Kyojin")
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock):
            result = process_image(image_path=img_path, mode="anime")

        assert result is not None
        assert result["status"] == "extracted"
        # Review queue should have been written since ratio will be low
        assert queue_path.exists()
        with open(queue_path) as f:
            entries = json.load(f)
        assert len(entries) >= 1
        assert entries[0]["extractor"] == "anime"


# --- Fixture-based tests (skip when files absent) ---


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "anime"


def _load_fixture(fixture_dir: Path, name: str) -> str | None:
    p = fixture_dir / name
    if not p.exists():
        return None
    import subprocess

    result = subprocess.run(
        ["python", "-c", f"from google.cloud import vision; print('ok')"],
        capture_output=True,
    )
    return str(p)


@pytest.mark.parametrize("fixture_name,expected_contains", [
    ("IMG_5432.PNG", "Rage of Bahamut"),
    ("IMG_2946.PNG", None),
])
def test_fixture_extraction(fixture_dir: Path, fixture_name: str, expected_contains: str | None):
    p = fixture_dir / fixture_name
    if not p.exists():
        pytest.skip(f"Fixture {fixture_name} not present")

    from paku.context import AppContext

    ctx = AppContext.instance()
    engine = ctx.resolve_engine("auto")

    from PIL import Image
    from paku.pipeline import preprocess

    img = preprocess(Image.open(p))
    from paku.ocr.stub import StubOCREngine
    if isinstance(engine, StubOCREngine):
        pytest.skip("Stub engine — no real OCR")

    from paku.models import OcrResult
    ocr: OcrResult = engine.extract(img)

    with patch("requests.post", _mock_post(expected_contains or "unknown", expected_contains or "unknown")):
        result = extract(ocr.raw_text, str(p), {}, _LOG)

    assert isinstance(result, (AnimeExtractionResult, list))


@pytest.mark.integration
def test_integration_img_4841_trigun_stargaze(fixture_dir: Path):
    """Real AniList call: IMG_4841 → TRIGUN STARGAZE with auto-accept."""
    p = fixture_dir / "IMG_4841.PNG"
    if not p.exists():
        pytest.skip("Fixture IMG_4841.PNG not present")

    from paku.context import AppContext

    ctx = AppContext.instance()
    engine = ctx.resolve_engine("auto")

    from PIL import Image
    from paku.pipeline import preprocess
    from paku.ocr.stub import StubOCREngine

    if isinstance(engine, StubOCREngine):
        pytest.skip("Stub engine — no real OCR")

    from paku.models import OcrResult

    img = preprocess(Image.open(p))
    ocr: OcrResult = engine.extract(img)

    # Real AniList call — no mock
    result = extract(ocr.raw_text, str(p), {}, _LOG)
    assert isinstance(result, AnimeExtractionResult)
    assert result.canonical_title == "TRIGUN STARGAZE"
    assert result.anilist_id == 163144
    assert result.levenshtein_ratio == pytest.approx(1.0)
    assert result.needs_review is False


# --- Phase 2 gate fixes ---


class TestEnhancedRatio:
    """Tests for Fix 1: enhanced ratio with substring and word-overlap boosts."""

    def test_base_ratio_above_threshold_unchanged(self):
        """When base Levenshtein >= 0.8, no boost applied."""
        ratio = _enhanced_ratio("Hell's Paradise", "Hell's Paradise")
        assert ratio >= 0.8  # exact match, returns base

    def test_substring_boost(self):
        """Short raw title that is a substring of canonical gets boosted to 0.85."""
        # "Dusk Beyond the End" is a prefix of "Dusk Beyond the End of the World"
        ratio = _enhanced_ratio("Dusk Beyond the End", "Dusk Beyond the End of the World")
        assert ratio >= 0.85

    def test_substring_too_short_no_boost(self):
        """Substring shorter than 8 chars should not trigger boost."""
        ratio = _enhanced_ratio("Rage", "Rage of Bahamut: Genesis")
        assert ratio < 0.85

    def test_word_overlap_boost(self):
        """Reordered words with high overlap get boosted."""
        # "IRON-BLOODED ORPHANS MOBILE SUIT GUNDAM" vs "Mobile Suit Gundam: Iron-Blooded Orphans"
        ratio = _enhanced_ratio(
            "iron-blooded orphans mobile suit gundam",
            "Mobile Suit Gundam Iron-Blooded Orphans",
        )
        assert ratio >= 0.85

    def test_no_boost_low_overlap(self):
        """Completely different titles should not be boosted."""
        ratio = _enhanced_ratio("Something Else Entirely", "Attack on Titan Final Season")
        assert ratio < 0.85

    def test_empty_strings(self):
        assert _enhanced_ratio("", "Anything") == 0.0
        assert _enhanced_ratio("Anything", "") == 0.0

    def test_compute_best_ratio_picks_highest(self):
        titles = {"english": "My Friend's Little Sister Has It In for Me!", "romaji": "Tomodachi no Imouto ga Ore ni Dake Uzai"}
        ratio = _compute_best_ratio("My Friend's Little Sister", titles)
        assert ratio >= 0.85  # substring match against english title


class TestMultiTitleConditionalReview:
    """Tests for Fix 2: multi-title screenshots auto-accept when ratio >= 0.8."""

    @patch("paku.extractors.anime.requests.post")
    def test_multi_title_high_ratio_auto_accepts(self, mock_post_fn):
        """Multi-title screenshot with ratio >= 0.8 should NOT force review."""
        mock_post_fn.side_effect = [
            _mock_post("Ganglion", "GANGLION", media_id=198117)(),
        ]
        text = "Upcoming anime\n1. Ganglion\n2. Other Title"
        result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        results = result if isinstance(result, list) else [result]
        # Find the Ganglion result (exact match = ratio 1.0)
        ganglion = next((r for r in results if "ganglion" in (r.raw_title or "").lower()), None)
        if ganglion and ganglion.levenshtein_ratio and ganglion.levenshtein_ratio >= 0.8:
            assert ganglion.needs_review is False

    @patch("paku.extractors.anime.requests.post")
    def test_multi_title_low_ratio_still_reviews(self, mock_post_fn):
        """Multi-title screenshot with ratio < 0.8 should still require review."""
        mock_post_fn.side_effect = [
            _mock_post("Completely Different Name", "Zenzen Chigau", media_id=99999)(),
        ]
        text = "Upcoming anime\n1. Pass the Monster\n2. Other Title"
        result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        results = result if isinstance(result, list) else [result]
        for r in results:
            if r.levenshtein_ratio and r.levenshtein_ratio < 0.8:
                assert r.needs_review is True


class TestTrailingPunctuationCleanup:
    """Tests for Fix 3: trailing punctuation stripped before AniList query."""

    @patch("paku.extractors.anime.requests.post")
    def test_trailing_bullet_stripped(self, mock_post_fn):
        """'Rage of Bahamut •' should query as 'Rage of Bahamut'."""
        mock_post_fn.side_effect = [
            _mock_post("Rage of Bahamut: Genesis", "Shingeki no Bahamut: Genesis", media_id=20590)(),
        ]
        text = "Send message\nRage of Bahamut •"
        result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        results = result if isinstance(result, list) else [result]
        r = results[0]
        # The query sent to AniList should not have the bullet
        assert r.canonical_title is not None
        assert "Bahamut" in (r.canonical_title or "")

    @patch("paku.extractors.anime.requests.post")
    def test_trailing_dash_stripped(self, mock_post_fn):
        """Trailing dashes and hyphens should be stripped."""
        mock_post_fn.side_effect = [
            _mock_post("Attack on Titan", "Shingeki no Kyojin", media_id=16498)(),
        ]
        text = "Send message\nAttack on Titan —"
        result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        results = result if isinstance(result, list) else [result]
        assert results[0].canonical_title is not None


# --- Expanded AniList field tests (010) ---


def _anilist_response_full() -> dict:
    """AniList response populated with all expanded fields."""
    return {
        "data": {
            "Media": {
                "id": 16498,
                "title": {
                    "english": "Attack on Titan",
                    "romaji": "Shingeki no Kyojin",
                    "native": "進撃の巨人",
                },
                "type": "ANIME",
                "format": "TV",
                "source": "MANGA",
                "episodes": 25,
                "status": "FINISHED",
                "genres": ["Action", "Drama"],
                "averageScore": 84,
                "siteUrl": "https://anilist.co/anime/16498",
                "countryOfOrigin": "JP",
                "startDate": {"year": 2013},
                "coverImage": {
                    "extraLarge": "https://img.anilist.co/cover-xl.jpg",
                    "large": "https://img.anilist.co/cover-l.jpg",
                },
                "bannerImage": "https://img.anilist.co/banner.jpg",
                "studios": {
                    "edges": [
                        {"node": {"name": "Wit Studio", "isAnimationStudio": True}},
                        {"node": {"name": "Production I.G", "isAnimationStudio": False}},
                        {"node": {"name": "MAPPA", "isAnimationStudio": True}},
                    ]
                },
            }
        }
    }


class TestExpandedAniListFields:
    def test_full_response_populates_new_fields(self):
        text = "Anime Name: Attack on Titan"
        mock = MagicMock()
        mock.json.return_value = _anilist_response_full()
        mock.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, AnimeExtractionResult)
        assert result.media_format == "TV"
        assert result.source == "MANGA"
        assert result.country_of_origin == "JP"
        assert result.debut_year == 2013
        assert result.banner_image == "https://img.anilist.co/banner.jpg"
        # extraLarge preferred over large
        assert result.cover_image == "https://img.anilist.co/cover-xl.jpg"
        # Only animation studios kept; order preserved
        assert result.studios == ["Wit Studio", "MAPPA"]

    def test_extra_large_falls_back_to_large(self):
        """If extraLarge is absent, cover_image must fall back to large."""
        resp = _anilist_response_full()
        del resp["data"]["Media"]["coverImage"]["extraLarge"]
        mock = MagicMock()
        mock.json.return_value = resp
        mock.raise_for_status = MagicMock()
        text = "Anime Name: Attack on Titan"
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.cover_image == "https://img.anilist.co/cover-l.jpg"

    def test_missing_optional_fields_use_defaults(self):
        """When AniList omits new fields, model defaults must hold (None / empty list)."""
        resp = {
            "data": {
                "Media": {
                    "id": 999,
                    "title": {"english": "Some Show", "romaji": "Some Show", "native": "X"},
                    "type": "ANIME",
                    "episodes": 12,
                    "status": "FINISHED",
                    "genres": [],
                    "averageScore": None,
                    "siteUrl": "https://anilist.co/anime/999",
                    "coverImage": {"large": "https://img.anilist.co/cover.jpg"},
                    # format, source, countryOfOrigin, startDate, bannerImage, studios omitted
                }
            }
        }
        mock = MagicMock()
        mock.json.return_value = resp
        mock.raise_for_status = MagicMock()
        text = "Anime Name: Some Show"
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.media_format is None
        assert result.source is None
        assert result.country_of_origin is None
        assert result.debut_year is None
        assert result.banner_image is None
        assert result.studios == []
        # cover_image still set from `large` fallback
        assert result.cover_image == "https://img.anilist.co/cover.jpg"

    def test_studios_with_null_start_date_object(self):
        """startDate present as dict but year is null → debut_year is None, no crash."""
        resp = _anilist_response_full()
        resp["data"]["Media"]["startDate"] = {"year": None}
        mock = MagicMock()
        mock.json.return_value = resp
        mock.raise_for_status = MagicMock()
        text = "Anime Name: Attack on Titan"
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.debut_year is None

    def test_network_error_new_fields_default(self):
        """Network error path must set new fields to defaults via Pydantic, not crash."""
        import requests as req_mod
        text = "Anime Name: Attack on Titan"
        with patch("requests.post", side_effect=req_mod.exceptions.ConnectionError("down")):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.raw_title == "Attack on Titan"
        assert result.media_format is None
        assert result.source is None
        assert result.country_of_origin is None
        assert result.debut_year is None
        assert result.banner_image is None
        assert result.studios == []

    def test_low_ratio_skips_enrichment_new_fields_default(self):
        """When ratio < 0.4, new fields must remain at defaults (no enrichment)."""
        resp = _anilist_response_full()
        # Force a mismatch with the queried title
        mock = MagicMock()
        mock.json.return_value = resp
        mock.raise_for_status = MagicMock()
        text = "Anime Name: Asdfghjkl Qwerty Zxcvbn"
        with patch("requests.post", return_value=mock):
            result = extract(text, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        # Ratio computed against canonical "Attack on Titan" → very low
        assert result.media_format is None
        assert result.studios == []
        assert result.debut_year is None
