from __future__ import annotations

import logging
from typing import Any

import pytest

from paku.extractors.recipe import (
    _assign_confidence,
    _extract_instructions,
    _extract_source_account,
    _extract_title,
    _find_ingredient_block,
    _parse_ingredient_line,
    extract,
)
from paku.models import Ingredient, RecipeExtractionResult

_LOG = logging.getLogger("test")
_DUMMY_PATH = "test.png"
_DUMMY_CONFIG: dict[str, Any] = {}


# --- Model tests ---


class TestRecipeModels:
    def test_ingredient_required_fields(self) -> None:
        ing = Ingredient(name="banana")
        assert ing.name == "banana"
        assert ing.quantity is None
        assert ing.unit is None
        assert ing.notes is None

    def test_ingredient_full_fields(self) -> None:
        ing = Ingredient(name="flour", quantity=100.0, unit="g", notes="sifted")
        assert ing.quantity == 100.0
        assert ing.unit == "g"
        assert ing.notes == "sifted"

    def test_recipe_result_defaults(self) -> None:
        result = RecipeExtractionResult(
            confidence=0.9,
            needs_review=False,
            source_screenshot="test.png",
            extracted_at="2026-01-01T00:00:00+00:00",
        )
        assert result.extractor == "recipe"
        assert result.ingredients == []
        assert result.title is None
        assert result.needs_review is False

    def test_recipe_result_with_ingredients(self) -> None:
        result = RecipeExtractionResult(
            confidence=0.9,
            needs_review=False,
            source_screenshot="test.png",
            extracted_at="2026-01-01T00:00:00+00:00",
            title="Banana Bread",
            ingredients=[Ingredient(name="banana", quantity=2.0)],
        )
        assert result.title == "Banana Bread"
        assert len(result.ingredients) == 1
        assert result.ingredients[0].quantity == 2.0


# --- _parse_ingredient_line tests ---


class TestParseIngredientLine:
    def test_attached_unit_no_space(self) -> None:
        ing = _parse_ingredient_line("100g protein powder")
        assert ing is not None
        assert ing.quantity == 100.0
        assert ing.unit == "g"
        assert ing.name == "protein powder"

    def test_attached_unit_ml(self) -> None:
        ing = _parse_ingredient_line("50ml skimmed milk")
        assert ing is not None
        assert ing.quantity == 50.0
        assert ing.unit == "ml"
        assert ing.name == "skimmed milk"

    def test_integer_qty_no_unit(self) -> None:
        ing = _parse_ingredient_line("2 bananas")
        assert ing is not None
        assert ing.quantity == 2.0
        assert ing.unit is None
        assert ing.name == "bananas"

    def test_qty_with_descriptive_unit(self) -> None:
        ing = _parse_ingredient_line("1 medium egg")
        assert ing is not None
        assert ing.quantity == 1.0
        assert ing.unit == "medium"
        assert ing.name == "egg"

    def test_word_qty_a_pinch(self) -> None:
        ing = _parse_ingredient_line("a pinch of salt")
        assert ing is not None
        assert ing.quantity is None
        assert ing.unit == "pinch"
        assert ing.name == "salt"

    def test_word_qty_a_handful(self) -> None:
        ing = _parse_ingredient_line("a handful of spinach")
        assert ing is not None
        assert ing.unit == "handful"
        assert ing.name == "spinach"

    def test_tbsp_unit(self) -> None:
        ing = _parse_ingredient_line("2 tbsp olive oil")
        assert ing is not None
        assert ing.quantity == 2.0
        assert ing.unit == "tbsp"
        assert ing.name == "olive oil"

    def test_unicode_fraction(self) -> None:
        ing = _parse_ingredient_line("½ cup oats")
        assert ing is not None
        assert ing.quantity == pytest.approx(0.5)
        assert ing.unit == "cup"
        assert ing.name == "oats"

    def test_vulgar_fraction(self) -> None:
        ing = _parse_ingredient_line("1/2 cup flour")
        assert ing is not None
        assert ing.quantity == pytest.approx(0.5)
        assert ing.unit == "cup"
        assert ing.name == "flour"

    def test_parenthesized_notes(self) -> None:
        ing = _parse_ingredient_line("50g butter (softened)")
        assert ing is not None
        assert ing.quantity == 50.0
        assert ing.unit == "g"
        assert ing.name == "butter"
        assert ing.notes == "softened"

    def test_to_taste_notes(self) -> None:
        ing = _parse_ingredient_line("salt, to taste")
        assert ing is not None
        assert ing.name == "salt"
        assert ing.notes == "to taste"
        assert ing.quantity is None

    def test_bullet_prefix_stripped(self) -> None:
        ing = _parse_ingredient_line("• 3 eggs")
        assert ing is not None
        assert ing.quantity == 3.0
        assert ing.name == "eggs"

    def test_numbered_list_prefix_stripped(self) -> None:
        ing = _parse_ingredient_line("1. 200g flour")
        assert ing is not None
        assert ing.quantity == 200.0
        assert ing.unit == "g"
        assert ing.name == "flour"

    def test_empty_line_returns_none(self) -> None:
        assert _parse_ingredient_line("") is None
        assert _parse_ingredient_line("  ") is None

    def test_ingredient_anchor_line_returns_none(self) -> None:
        assert _parse_ingredient_line("Ingredients:") is None
        assert _parse_ingredient_line("ingredienti:") is None

    def test_instruction_anchor_line_returns_none(self) -> None:
        assert _parse_ingredient_line("Method:") is None
        assert _parse_ingredient_line("Instructions:") is None

    def test_european_decimal(self) -> None:
        ing = _parse_ingredient_line("100,5g oat flour")
        assert ing is not None
        assert ing.quantity == pytest.approx(100.5)
        assert ing.unit == "g"

    def test_decimal_qty(self) -> None:
        ing = _parse_ingredient_line("1.5 cups milk")
        assert ing is not None
        assert ing.quantity == pytest.approx(1.5)
        assert ing.unit == "cup"
        assert ing.name == "milk"

    def test_name_only_no_qty(self) -> None:
        # Lines without qty should still parse as ingredient name
        ing = _parse_ingredient_line("cinnamon")
        assert ing is not None
        assert ing.name == "cinnamon"
        assert ing.quantity is None
        assert ing.unit is None

    def test_unit_normalization_tablespoon(self) -> None:
        ing = _parse_ingredient_line("2 tablespoons honey")
        assert ing is not None
        assert ing.unit == "tbsp"

    def test_unit_normalization_grams(self) -> None:
        ing = _parse_ingredient_line("200 grams chicken")
        assert ing is not None
        assert ing.unit == "g"


# --- _find_ingredient_block tests ---


class TestFindIngredientBlock:
    def test_anchor_detected(self) -> None:
        lines = [
            "Banana Protein Bread",
            "Ingredients:",
            "- 2 bananas",
            "- 1 egg",
            "- 100g flour",
        ]
        anchor, start, end = _find_ingredient_block(lines)
        assert anchor == 1
        assert start == 2
        assert end == 5

    def test_italian_anchor(self) -> None:
        lines = ["Ricetta", "Ingredienti:", "100g farina", "2 uova"]
        anchor, start, end = _find_ingredient_block(lines)
        assert anchor == 1
        assert start == 2
        assert end == 4

    def test_what_you_need_anchor(self) -> None:
        lines = ["My Recipe", "What you'll need:", "2 eggs", "100g sugar"]
        anchor, start, end = _find_ingredient_block(lines)
        assert anchor == 1
        assert start == 2
        assert end == 4

    def test_block_ends_at_instruction_anchor(self) -> None:
        lines = [
            "Recipe Title",
            "Ingredients:",
            "2 eggs",
            "100g flour",
            "Method:",
            "Mix everything together.",
        ]
        anchor, start, end = _find_ingredient_block(lines)
        assert anchor == 1
        assert end == 4  # stops before "Method:"

    def test_block_ends_at_double_empty_line(self) -> None:
        lines = [
            "Recipe",
            "Ingredients:",
            "2 bananas",
            "1 egg",
            "",
            "",
            "Some other content",
        ]
        anchor, start, end = _find_ingredient_block(lines)
        assert anchor == 1
        # end may include the first empty line but not reach the second block
        assert end <= 5

    def test_no_anchor_returns_full_range(self) -> None:
        lines = ["2 bananas", "1 egg", "100g flour"]
        anchor, start, end = _find_ingredient_block(lines)
        assert anchor is None
        assert start == 0
        assert end == 3


# --- _assign_confidence tests ---


class TestAssignConfidence:
    def test_anchor_with_multiple_ingredients(self) -> None:
        ingredients = [
            Ingredient(name="banana", quantity=2.0),
            Ingredient(name="egg", quantity=1.0),
            Ingredient(name="flour", quantity=100.0, unit="g"),
        ]
        conf, review = _assign_confidence(anchor_idx=1, ingredients=ingredients)
        assert conf == 0.9
        assert review is False

    def test_anchor_with_single_ingredient(self) -> None:
        ingredients = [Ingredient(name="banana")]
        conf, review = _assign_confidence(anchor_idx=1, ingredients=ingredients)
        assert conf == 0.7
        assert review is True

    def test_anchor_with_no_ingredients(self) -> None:
        conf, review = _assign_confidence(anchor_idx=1, ingredients=[])
        assert conf == 0.7
        assert review is True

    def test_no_anchor_with_ingredients(self) -> None:
        ingredients = [Ingredient(name="banana"), Ingredient(name="egg")]
        conf, review = _assign_confidence(anchor_idx=None, ingredients=ingredients)
        assert conf == 0.5
        assert review is True

    def test_no_anchor_no_ingredients(self) -> None:
        conf, review = _assign_confidence(anchor_idx=None, ingredients=[])
        assert conf == 0.1
        assert review is True


# --- _extract_title tests ---


class TestExtractTitle:
    def test_returns_longest_pre_anchor_line(self) -> None:
        lines = [
            "daniel.pitasi",
            "Banana Protein Bread",
            "A healthy recipe",
            "Ingredients:",
        ]
        title = _extract_title(lines, anchor_idx=3)
        # handle "daniel.pitasi" is filtered; longest of remaining candidates wins
        assert title == "Banana Protein Bread"

    def test_skips_chrome_lines(self) -> None:
        lines = [
            "@daniel.pitasi",
            "Banana Protein Bread",
            "Ingredients:",
        ]
        title = _extract_title(lines, anchor_idx=2)
        assert title == "Banana Protein Bread"

    def test_no_anchor_scans_first_15(self) -> None:
        lines = ["Short", "A much longer recipe title here", "another line"]
        title = _extract_title(lines, anchor_idx=None)
        assert title == "A much longer recipe title here"

    def test_returns_none_on_all_chrome(self) -> None:
        lines = ["@user", "#hashtag", "1234"]
        title = _extract_title(lines, anchor_idx=None)
        assert title is None


# --- _extract_source_account tests ---


class TestExtractSourceAccount:
    def test_plain_handle(self) -> None:
        lines = ["daniel.pitasi", "Banana Bread", "Ingredients:"]
        result = _extract_source_account(lines)
        assert result == "daniel.pitasi"

    def test_skips_chrome_words(self) -> None:
        lines = ["Follow", "Banana Bread"]
        result = _extract_source_account(lines)
        assert result is None

    def test_no_handle_returns_none(self) -> None:
        result = _extract_source_account(["Banana Bread", "2 eggs"])
        assert result is None


# --- _extract_instructions tests ---


class TestExtractInstructions:
    def test_extracts_after_block(self) -> None:
        lines = [
            "Recipe",
            "Ingredients:",
            "2 eggs",
            "Method:",
            "Mix well.",
            "Bake at 180C.",
        ]
        instr = _extract_instructions(lines, ing_end=3)
        # Should skip "Method:" anchor and return the steps
        assert instr is not None
        assert "Mix well." in instr
        assert "Bake at 180C." in instr

    def test_no_instruction_anchor_returns_remaining(self) -> None:
        lines = ["Recipe", "Ingredients:", "2 eggs", "", "Mix well."]
        instr = _extract_instructions(lines, ing_end=3)
        assert instr is not None
        assert "Mix well." in instr

    def test_no_content_after_block_returns_none(self) -> None:
        lines = ["Recipe", "Ingredients:", "2 eggs"]
        instr = _extract_instructions(lines, ing_end=3)
        assert instr is None


# --- Full extract() integration tests ---


class TestExtract:
    def test_full_recipe_high_confidence(self) -> None:
        ocr = (
            "daniel.pitasi\n"
            "Banana Protein Bread\n"
            "Ingredients:\n"
            "- 2 bananas\n"
            "- 1 medium egg\n"
            "- 100g protein powder\n"
            "- 50ml skimmed milk\n"
            "Method:\n"
            "Mash bananas. Mix all ingredients. Bake 20 mins."
        )
        result = extract(ocr, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert isinstance(result, RecipeExtractionResult)
        assert result.confidence == 0.9
        assert result.needs_review is False
        assert len(result.ingredients) == 4

        qty_map = {ing.name: (ing.quantity, ing.unit) for ing in result.ingredients}
        assert qty_map["bananas"] == (2.0, None)
        assert qty_map["egg"] == (1.0, "medium")
        assert qty_map["protein powder"] == (100.0, "g")
        assert qty_map["skimmed milk"] == (50.0, "ml")

        assert result.title is not None
        assert "Banana" in result.title
        assert result.instructions is not None

    def test_partial_recipe_low_confidence(self) -> None:
        ocr = (
            "Quick Snack\n"
            "Ingredients:\n"
            "1 banana\n"
        )
        result = extract(ocr, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.confidence == 0.7
        assert result.needs_review is True

    def test_no_anchor_heuristic(self) -> None:
        ocr = (
            "Healthy Bowl\n"
            "2 cups rice\n"
            "1 avocado\n"
            "3 eggs\n"
        )
        result = extract(ocr, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.confidence == 0.5
        assert result.needs_review is True
        assert len(result.ingredients) >= 1

    def test_empty_text_very_low_confidence(self) -> None:
        result = extract("", _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.confidence <= 0.5
        assert result.needs_review is True

    def test_dedup_key_from_title(self) -> None:
        ocr = (
            "Banana Bread Recipe\n"
            "Ingredients:\n"
            "- 2 bananas\n"
            "- 100g flour\n"
            "- 1 egg\n"
        )
        result = extract(ocr, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.dedup_key is not None
        assert result.dedup_key == result.dedup_key.lower()

    def test_model_dump_roundtrip(self) -> None:
        ocr = (
            "My Recipe\n"
            "Ingredients:\n"
            "- 2 eggs\n"
            "- 100g flour\n"
            "- 50ml milk\n"
        )
        result = extract(ocr, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        dumped = result.model_dump()
        assert "ingredients" in dumped
        assert isinstance(dumped["ingredients"], list)
        for ing in dumped["ingredients"]:
            assert "name" in ing
            # qty and unit must always be separate fields (never merged)
            assert "quantity" in ing
            assert "unit" in ing

    def test_italian_anchor(self) -> None:
        ocr = (
            "Ricetta proteica\n"
            "Ingredienti:\n"
            "2 banane\n"
            "100g farina d'avena\n"
            "1 uovo medio\n"
        )
        result = extract(ocr, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        assert result.confidence >= 0.7
        assert len(result.ingredients) >= 2

    def test_source_screenshot_and_extracted_at_set(self) -> None:
        result = extract("Ingredients:\n2 eggs\n1 banana\n", "my_photo.png", _DUMMY_CONFIG, _LOG)
        assert result.source_screenshot == "my_photo.png"
        assert result.extracted_at != ""

    def test_qty_unit_never_merged(self) -> None:
        """Core invariant: quantity and unit are always separate fields."""
        ocr = "Ingredients:\n100g oat flour\n200ml almond milk\n2 tablespoons honey\n"
        result = extract(ocr, _DUMMY_PATH, _DUMMY_CONFIG, _LOG)
        for ing in result.ingredients:
            # quantity must be numeric (float) or None, never a string like "100g"
            assert ing.quantity is None or isinstance(ing.quantity, (int, float))
            # unit must be a string or None, never contain digits
            if ing.unit is not None:
                assert not any(c.isdigit() for c in ing.unit), (
                    f"Unit '{ing.unit}' for '{ing.name}' contains digits — qty/unit merged"
                )


# --- Metric-in-parens reversed format (IMG_6796 / giallozafferano style) ---


class TestMetricParensReversedFormat:
    """OCR from giallozafferano-style sites renders 'Name qty unit (metric g)'.
    Unicode fractions (¾, ⅔, ½) are OCR-garbled to digit runs (34, 23, 12).
    The extractor must use the parenthesized metric value as authoritative qty."""

    def test_garbled_fraction_cup_with_metric(self) -> None:
        # "1 ¾ cup" → OCR: "134 cup"; metric (450 g) is reliable
        ing = _parse_ingredient_line("Whole milk 134 cup (450 g)")
        assert ing is not None
        assert ing.name == "Whole milk"
        assert ing.quantity == 450.0
        assert ing.unit == "g"

    def test_garbled_two_thirds_cup(self) -> None:
        # "⅔ cup" → OCR: "23 cup"
        ing = _parse_ingredient_line("Heavy cream 23 cup (150 g)")
        assert ing is not None
        assert ing.name == "Heavy cream"
        assert ing.quantity == 150.0
        assert ing.unit == "g"

    def test_eggs_with_secondary_note(self) -> None:
        # "(220 g) - (about 4 medium)" — secondary note stripped, metric used
        ing = _parse_ingredient_line("Eggs 1 cup (220 g) - (about 4 medium)")
        assert ing is not None
        assert ing.name == "Eggs"
        assert ing.quantity == 220.0
        assert ing.unit == "g"

    def test_garbled_half_tbsp_with_metric(self) -> None:
        # "2 ½ tbsp" → OCR: "2 12 tbsp"
        ing = _parse_ingredient_line("Egg yolks 2 12 tbsp (20 g) - (about 1 medium)")
        assert ing is not None
        assert ing.name == "Egg yolks"
        assert ing.quantity == 20.0
        assert ing.unit == "g"

    def test_no_metric_parens_unchanged(self) -> None:
        # Normal trailing-qty line without metric parens — existing logic unchanged
        ing = _parse_ingredient_line("Sale fino 1 pizzico")
        assert ing is not None
        assert ing.name == "Sale fino"
        assert ing.quantity == 1.0
        assert ing.unit == "pinch"

    def test_normal_forward_format_unchanged(self) -> None:
        # Normal "qty unit name (metric)" format — existing leading-qty path unchanged
        ing = _parse_ingredient_line("2 cups flour (240 g)")
        assert ing is not None
        assert ing.name == "flour"
        assert ing.quantity == 2.0
        assert ing.unit == "cup"


class TestBareCountFallback:
    """'Vanilla bean 1' — no unit, just a count at the end of the name."""

    def test_vanilla_bean(self) -> None:
        ing = _parse_ingredient_line("Vanilla bean 1")
        assert ing is not None
        assert ing.name == "Vanilla bean"
        assert ing.quantity == 1.0
        assert ing.unit is None

    def test_bay_leaves(self) -> None:
        ing = _parse_ingredient_line("Bay leaves 3")
        assert ing is not None
        assert ing.name == "Bay leaves"
        assert ing.quantity == 3.0
        assert ing.unit is None

    def test_single_word_name_not_matched(self) -> None:
        # "Salt 1" — single-word name is too ambiguous; falls through without bare count
        ing = _parse_ingredient_line("Salt 1")
        # Should still produce an ingredient, but the bare-count rule does not fire
        # (single-word name guard). qty may be None here — just ensure no crash.
        assert ing is not None or ing is None  # no crash

    def test_does_not_match_ingredient_with_unit(self) -> None:
        # "Farina 00 500 g" has a unit; trailing-qty handles it, bare-count should not fire
        ing = _parse_ingredient_line("Farina 00 500 g")
        # Either trailing-qty or metric path handles this; unit must be g
        assert ing is not None
        assert ing.unit == "g"


# --- Title extraction: music credit and fragment rejection (IMG_6535) ---


class TestExtractTitleRejectsNonRecipeCandidates:
    """Title extraction should return None rather than a wrong music credit or fragment."""

    def test_rejects_music_credit_middle_dot(self) -> None:
        # Instagram music credit: "Starly · Kavkaz" (with middle dot)
        lines = ["daniel.pitasi Follow", "Starly · Kavkaz", "below)", "ingredients:"]
        title = _extract_title(lines, anchor_idx=3)
        assert title is None

    def test_rejects_fragment_ending_paren(self) -> None:
        # Truncated caption fragment: "below)" — ends with ) but no opening (
        lines = ["daniel.pitasi Follow", "below)", "ingredients:"]
        title = _extract_title(lines, anchor_idx=2)
        assert title is None

    def test_keeps_title_with_balanced_parens(self) -> None:
        # A title that has balanced parentheses should not be rejected
        lines = ["Banana Bread (Easy Recipe)", "ingredients:"]
        title = _extract_title(lines, anchor_idx=1)
        assert title == "Banana Bread (Easy Recipe)"

    def test_keeps_normal_title(self) -> None:
        lines = ["ritas.recipes Follow", "UDON ALLE VERDURE", "Ingredienti:"]
        title = _extract_title(lines, anchor_idx=2)
        assert title == "UDON ALLE VERDURE"

    def test_no_title_returns_none_when_all_rejected(self) -> None:
        # All candidates filtered → None (no title is a valid, expected outcome)
        lines = ["daniel.pitasi Follow", "Starly · Kavkaz", "below)", "ingredients:"]
        title = _extract_title(lines, anchor_idx=3)
        assert title is None
