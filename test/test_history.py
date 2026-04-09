"""Tests for q2mm.diagnostics.history module."""

from __future__ import annotations

import json
from pathlib import Path

from q2mm.diagnostics.history import RunSummary, generate_run_id, load_history


class TestRunSummary:
    """Round-trip serialization and field validation."""

    def _make_summary(self) -> RunSummary:
        return RunSummary(
            run_id="ch3f_abc12345_20260409T120000",
            system="ch3f",
            git_sha="abc12345deadbeef",
            git_dirty=False,
            timestamp="2026-04-09T12:00:00+00:00",
            environment={"python": "3.12.0", "gpu": "RTX 5090"},
            config={"backends": ["jax"], "forms": ["harmonic"], "n_combos": 2},
            combos={
                "ch3f_jax_harmonic_gpu_powell": {
                    "stem": "ch3f_jax_harmonic_gpu_powell",
                    "status": "converged",
                    "rmsd": 0.0,
                    "mae": 0.0,
                    "time_s": 1.5,
                    "n_eval": 42,
                },
                "ch3f_jax_harmonic_gpu_lbfgsb_auto": {
                    "stem": "ch3f_jax_harmonic_gpu_lbfgsb_auto",
                    "status": "converged",
                    "rmsd": 550.0,
                    "mae": 270.0,
                    "time_s": 10.2,
                    "n_eval": 300,
                },
            },
        )

    def test_round_trip(self, tmp_path: Path) -> None:
        """to_json then from_json should reproduce all fields."""
        original = self._make_summary()
        path = tmp_path / "run.json"
        original.to_json(path)

        loaded = RunSummary.from_json(path)
        assert loaded.run_id == original.run_id
        assert loaded.system == original.system
        assert loaded.git_sha == original.git_sha
        assert loaded.git_dirty == original.git_dirty
        assert loaded.timestamp == original.timestamp
        assert loaded.environment == original.environment
        assert loaded.config == original.config
        assert loaded.combos == original.combos

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """to_json should create missing parent directories."""
        summary = self._make_summary()
        path = tmp_path / "nested" / "dir" / "run.json"
        summary.to_json(path)
        assert path.exists()

    def test_json_is_valid(self, tmp_path: Path) -> None:
        """Output file should be valid JSON with expected top-level keys."""
        summary = self._make_summary()
        path = tmp_path / "run.json"
        summary.to_json(path)

        with open(path) as fh:
            data = json.load(fh)
        assert set(data.keys()) == {
            "run_id",
            "system",
            "git_sha",
            "git_dirty",
            "timestamp",
            "environment",
            "config",
            "combos",
        }


class TestLoadHistory:
    """Tests for load_history()."""

    def test_empty_dir(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        assert load_history(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        """Missing directory returns empty list."""
        assert load_history(tmp_path / "does_not_exist") == []

    def test_sorted_by_timestamp(self, tmp_path: Path) -> None:
        """Runs should be sorted by timestamp, oldest first."""
        for i, ts in enumerate(["2026-04-09", "2026-04-07", "2026-04-08"]):
            s = RunSummary(
                run_id=f"run_{i}",
                system="ch3f",
                git_sha=None,
                git_dirty=None,
                timestamp=f"{ts}T00:00:00+00:00",
            )
            s.to_json(tmp_path / f"run_{i}.json")

        runs = load_history(tmp_path)
        assert len(runs) == 3
        assert runs[0].run_id == "run_1"  # 04-07
        assert runs[1].run_id == "run_2"  # 04-08
        assert runs[2].run_id == "run_0"  # 04-09

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON files should be skipped, not crash."""
        (tmp_path / "bad.json").write_text("not valid json")
        s = RunSummary(
            run_id="good",
            system="ch3f",
            git_sha=None,
            git_dirty=None,
            timestamp="2026-04-09T00:00:00+00:00",
        )
        s.to_json(tmp_path / "good.json")

        runs = load_history(tmp_path)
        assert len(runs) == 1
        assert runs[0].run_id == "good"


class TestGenerateRunId:
    """Tests for generate_run_id()."""

    def test_format(self) -> None:
        """Run ID should contain system name and timestamp parts."""
        rid = generate_run_id("ch3f")
        assert rid.startswith("ch3f_")
        parts = rid.split("_")
        assert len(parts) >= 3
        # Timestamp part should start with a year
        assert parts[2].startswith("20")

    def test_unique(self) -> None:
        """Two calls should produce different IDs (different timestamps)."""
        import time

        r1 = generate_run_id("ch3f")
        time.sleep(1.1)
        r2 = generate_run_id("ch3f")
        assert r1 != r2
