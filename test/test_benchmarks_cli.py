"""Smoke tests for the ``q2mm-benchmark`` CLI (:mod:`q2mm.benchmarks.cli`).

``list`` and ``preflight`` must be side-effect-free/probing-only and run
without any backend.  ``single``/``matrix``/``load`` are exercised against
an unavailable backend (graceful skip) so the command wiring is covered
without requiring JAX/OpenMM; a JAX-marked test runs the real accepted
path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from q2mm.benchmarks.cli import main


class TestListAndPreflight:
    def test_list_runs_and_lists_sections(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "Systems:" in out
        assert "Backends" in out
        assert "Optimizers:" in out
        assert "ch3f" in out

    def test_preflight_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["preflight"]) == 0
        assert "Pre-flight" in capsys.readouterr().out

    def test_no_command_errors(self) -> None:
        with pytest.raises(SystemExit):
            main([])


class TestSingleAndMatrixSkipPath:
    """Exercise command wiring against a registered-but-unavailable backend (graceful skip)."""

    @staticmethod
    def _force_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the ``jax`` descriptor report its dependency missing (skip path)."""
        import q2mm.backends.registry as registry

        real = registry.get_descriptor("jax")

        class _Desc:
            name = real.name
            backend_api_version = real.backend_api_version
            factory = real.factory
            role = real.role
            capability_ceiling = real.capability_ceiling
            functional_form_ceiling = real.functional_form_ceiling
            probe = real.probe

            def is_available(self) -> tuple[bool, str]:
                return False, "synthetic missing dependency"

        monkeypatch.setattr(registry, "get_descriptor", lambda key: _Desc() if key == "jax" else real)

    def test_single_skipped_backend_writes_candidate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._force_unavailable(monkeypatch)
        rc = main(
            [
                "single",
                "--system",
                "ch3f",
                "--backend",
                "jax",
                "--optimizer",
                "scipy-lbfgsb-jax",
                "--output",
                str(tmp_path),
            ]
        )
        assert rc == 0
        assert len(list((tmp_path / "candidates").glob("*.json"))) == 1

    def test_matrix_skipped_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._force_unavailable(monkeypatch)
        rc = main(
            [
                "matrix",
                "--system",
                "ch3f",
                "--backend",
                "jax",
                "--optimizer",
                "scipy-lbfgsb-jax",
                "--form",
                "harmonic",
                "--output",
                str(tmp_path),
            ]
        )
        assert rc == 0
        assert (tmp_path / "candidates").is_dir()

    def test_single_unknown_backend_errors(self, tmp_path: Path) -> None:
        # An unknown backend is a configuration error -> nonzero exit.
        rc = main(
            [
                "single",
                "--system",
                "ch3f",
                "--backend",
                "does-not-exist",
                "--optimizer",
                "scipy-lbfgsb",
                "--output",
                str(tmp_path),
            ]
        )
        assert rc == 1

    def test_load_reads_written_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._force_unavailable(monkeypatch)
        main(
            [
                "single",
                "--system",
                "ch3f",
                "--backend",
                "jax",
                "--optimizer",
                "scipy-lbfgsb-jax",
                "--output",
                str(tmp_path),
            ]
        )
        capsys.readouterr()
        assert main(["load", str(tmp_path)]) == 0
        assert "candidate record" in capsys.readouterr().out

    def test_load_missing_directory_errors(self, tmp_path: Path) -> None:
        assert main(["load", str(tmp_path / "does-not-exist")]) == 1

    def test_matrix_unknown_system_errors(self, tmp_path: Path) -> None:
        assert main(["matrix", "--system", "nope", "--output", str(tmp_path)]) == 1

    def test_batch_unknown_system_errors(self, tmp_path: Path) -> None:
        assert main(["batch", "--system", "nope", "--output", str(tmp_path)]) == 1


class TestKnobThreading:
    """The restored optimizer/profile knobs must reach the built RunProfile."""

    def _parse(self, argv: list[str]) -> object:
        from q2mm.benchmarks.cli import _build_parser

        return _build_parser().parse_args(argv)

    def test_optimizer_knobs_thread_into_profile(self) -> None:
        from q2mm.benchmarks.cli import _optimizer_knobs
        from q2mm.benchmarks.profiles import RunProfile

        args = self._parse(
            [
                "single",
                "--system",
                "ch3f",
                "--learning-rate",
                "0.05",
                "--max-params",
                "7",
                "--max-cycles",
                "42",
                "--convergence",
                "0.002",
                "--regularization",
                "0.25",
                "--data-root",
                "ch3f=/tmp/ch3f-data",
            ]
        )
        knobs = _optimizer_knobs(args)  # type: ignore[arg-type]
        profile = RunProfile(system="ch3f", **knobs)  # type: ignore[arg-type]
        assert profile.learning_rate == 0.05
        assert profile.max_params == 7
        assert profile.max_cycles == 42
        assert profile.convergence == 0.002
        assert profile.regularization == 0.25
        assert profile.data_roots["ch3f"] == "/tmp/ch3f-data"

    def test_regularization_defaults_none(self) -> None:
        from q2mm.benchmarks.cli import _optimizer_knobs

        args = self._parse(["single", "--system", "ch3f"])
        assert _optimizer_knobs(args)["regularization"] is None  # type: ignore[arg-type]

    def test_objective_profile_threads_into_run_profile(self) -> None:
        from q2mm.benchmarks.publications import FERROCENE_SEVEN_STRUCTURE_PROFILE
        from q2mm.benchmarks.profiles import RunProfile

        args = self._parse(
            [
                "single",
                "--system",
                "ferrocene",
                "--starting-point",
                "published",
                "--objective-profile",
                FERROCENE_SEVEN_STRUCTURE_PROFILE,
            ]
        )
        profile = RunProfile(
            system=args.system,  # type: ignore[attr-defined]
            starting_point=args.starting_point,  # type: ignore[attr-defined]
            objective_profile=args.objective_profile,  # type: ignore[attr-defined]
        )
        assert profile.objective_profile == FERROCENE_SEVEN_STRUCTURE_PROFILE

    def test_data_root_rejects_unknown_key(self) -> None:
        import argparse

        from q2mm.benchmarks.cli import _data_root_pair

        with pytest.raises(argparse.ArgumentTypeError):
            _data_root_pair("bogus=/x")

    def test_new_knobs_change_candidate_identity(self) -> None:
        from q2mm.benchmarks.profiles import RunProfile

        base = RunProfile(system="ch3f")
        for override in (
            {"learning_rate": 0.9},
            {"max_params": 9},
            {"max_cycles": 99},
            {"convergence": 0.5},
        ):
            assert RunProfile(system="ch3f", **override).candidate_id() != base.candidate_id()  # type: ignore[arg-type]


@pytest.mark.jax
class TestSingleRealRun:
    def test_single_ch3f_accepted(self, tmp_path: Path) -> None:
        rc = main(
            [
                "single",
                "--system",
                "ch3f",
                "--backend",
                "jax",
                "--form",
                "harmonic",
                "--optimizer",
                "scipy-lbfgsb-jax",
                "--maxiter",
                "3",
                "--n-evals",
                "0",
                "--output",
                str(tmp_path),
                "--no-analyze",
            ]
        )
        assert rc == 0
        accepted = list((tmp_path / "accepted").glob("*.json"))
        assert len(accepted) == 1
        assert list((tmp_path / "forcefields").glob("*.frcmod"))
