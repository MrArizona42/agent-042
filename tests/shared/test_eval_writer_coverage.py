"""Tests for shared.db.eval_writer's evaluation-coverage decision logic.

`check_evaluation_coverage` itself needs a live Postgres connection (this
repo has none in its test environment, same limitation noted in phase 3);
`_coverage_from_rows` is the pure decision logic factored out specifically
so it can be tested without one.
"""

from __future__ import annotations

from shared.db.eval_writer import _coverage_from_rows


class TestCoverageFromRows:
    def test_no_required_datasets_is_never_covered(self):
        assert _coverage_from_rows([], required_dataset_names=[]) is False

    def test_single_completed_dataset_is_covered(self):
        rows = [("kb.benchmark", "completed", "pass")]
        assert _coverage_from_rows(rows, required_dataset_names=["kb.benchmark"]) is True

    def test_missing_dataset_is_not_covered(self):
        rows = [("kb.benchmark_a", "completed", "pass")]
        assert (
            _coverage_from_rows(rows, required_dataset_names=["kb.benchmark_a", "kb.benchmark_b"])
            is False
        )

    def test_all_required_datasets_completed_is_covered(self):
        rows = [
            ("kb.benchmark_a", "completed", "pass"),
            ("kb.benchmark_b", "completed", "unscored"),
        ]
        assert (
            _coverage_from_rows(rows, required_dataset_names=["kb.benchmark_a", "kb.benchmark_b"])
            is True
        )

    def test_any_failing_verdict_blocks_coverage(self):
        rows = [
            ("kb.benchmark_a", "completed", "pass"),
            ("kb.benchmark_a", "completed", "fail"),
        ]
        assert _coverage_from_rows(rows, required_dataset_names=["kb.benchmark_a"]) is False

    def test_unscored_verdict_does_not_block_coverage(self):
        rows = [("kb.benchmark_a", "completed", "unscored")]
        assert _coverage_from_rows(rows, required_dataset_names=["kb.benchmark_a"]) is True

    def test_running_status_does_not_count_as_completed(self):
        rows = [("kb.benchmark_a", "running", None)]
        assert _coverage_from_rows(rows, required_dataset_names=["kb.benchmark_a"]) is False

    def test_failed_status_row_for_unrelated_dataset_does_not_block_others(self):
        rows = [
            ("kb.benchmark_a", "completed", "pass"),
            ("kb.benchmark_b", "failed", None),
        ]
        # kb.benchmark_b never completed, so it's still not covered overall,
        # but a non-'fail' verdict (None here) must not itself block.
        assert _coverage_from_rows(rows, required_dataset_names=["kb.benchmark_a"]) is True
