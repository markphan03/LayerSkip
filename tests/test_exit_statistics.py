"""Exit distributions include zero-count layers and use draft denominators."""

import pytest

from self_speculation.exit_statistics import format_draft_exit_statistics, summarize_draft_exits


def test_layer_percentages_and_depth_bands():
    stats = summarize_draft_exits([1, 1, 2, 0, 0, 0, 1, 3])
    assert stats["total_drafts"] == 8
    assert [row["layer"] for row in stats["layers"]] == list(range(1, 9))
    assert [row["percentage"] for row in stats["layers"]] == [12.5, 12.5, 25, 0, 0, 0, 12.5, 37.5]
    summary = stats["depth_summary"]
    assert [row["last_layer"] for row in summary] == [2, 4, 6, 8]
    assert [row["exit_count"] for row in summary] == [2, 2, 0, 4]
    assert [row["percentage"] for row in summary] == [25, 25, 0, 50]
    assert [row["cumulative_percentage"] for row in summary] == [25, 50, 50, 100]
    table = format_draft_exit_statistics(stats)
    assert "8 drafts" in table and "Cumulative" in table
    assert "0–25%" in table and "75–100%" in table
    assert "Layer 8 is full depth" in table


@pytest.mark.parametrize("layers,cutoffs", [(35, [9, 18, 27, 35]), (3, [1, 2, 3, 3]), (1, [1, 1, 1, 1])])
def test_depth_cutoffs_round_up_and_bands_do_not_double_count(layers, cutoffs):
    stats = summarize_draft_exits([1] * layers)
    assert [row["last_layer"] for row in stats["depth_summary"]] == cutoffs
    assert sum(row["exit_count"] for row in stats["depth_summary"]) == layers
    assert stats["depth_summary"][-1]["cumulative_percentage"] == 100


def test_no_drafts_has_all_layers_and_undefined_percentages():
    stats = summarize_draft_exits([0] * 4)
    assert stats["total_drafts"] == 0 and len(stats["layers"]) == 4
    for row in stats["layers"] + stats["depth_summary"]:
        assert row["exit_count"] == 0
        assert row["percentage"] is None and row["cumulative_percentage"] is None
    assert "N/A" in format_draft_exit_statistics(stats)
    assert summarize_draft_exits([]) is None
    assert format_draft_exit_statistics(None) == ""
