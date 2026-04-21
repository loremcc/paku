from __future__ import annotations

from pathlib import Path

import pytest

from paku.outputs.txt_out import write_batch_txt


class TestWriteBatchTxt:
    def test_writes_entries_sorted(self, tmp_path):
        entries = ["Zelda", "Alpha", "Mario"]
        out = write_batch_txt(entries, tmp_path / "out.txt")
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines == ["Alpha", "Mario", "Zelda"]

    def test_deduplicates_entries(self, tmp_path):
        entries = ["Alpha", "Beta", "Alpha", "Gamma", "Beta"]
        out = write_batch_txt(entries, tmp_path / "out.txt")
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines == ["Alpha", "Beta", "Gamma"]
        assert len(lines) == 3

    def test_5_entries_1_duplicate_gives_4_lines(self, tmp_path):
        entries = ["E", "D", "C", "B", "B"]
        out = write_batch_txt(entries, tmp_path / "out.txt")
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4

    def test_empty_list_creates_file(self, tmp_path):
        out = write_batch_txt([], tmp_path / "out.txt")
        assert out.exists()
        assert out.read_text(encoding="utf-8") == ""

    def test_empty_list_zero_lines(self, tmp_path):
        out = write_batch_txt([], tmp_path / "out.txt")
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines == []

    def test_filters_empty_strings(self, tmp_path):
        entries = ["Alpha", "", "Beta", None]  # type: ignore[list-item]
        out = write_batch_txt(entries, tmp_path / "out.txt")
        lines = out.read_text(encoding="utf-8").splitlines()
        assert "" not in lines
        assert None not in lines
        assert "Alpha" in lines
        assert "Beta" in lines

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        out = write_batch_txt(["X"], nested / "out.txt")
        assert out.exists()

    def test_atomic_no_tmp_file_left(self, tmp_path):
        out_path = tmp_path / "out.txt"
        write_batch_txt(["A"], out_path)
        tmp = tmp_path / "out.txt.tmp"
        assert not tmp.exists()
        assert out_path.exists()

    def test_returns_output_path(self, tmp_path):
        out_path = tmp_path / "result.txt"
        returned = write_batch_txt(["A"], out_path)
        assert returned == out_path

    def test_single_entry(self, tmp_path):
        out = write_batch_txt(["OnlyOne"], tmp_path / "out.txt")
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines == ["OnlyOne"]
