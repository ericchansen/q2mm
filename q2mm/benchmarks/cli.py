"""The one ``q2mm-benchmark`` command-line entry point.

Explicit, current-only subcommands compose
:class:`~q2mm.benchmarks.profiles.RunProfile` objects and hand them to the
single :func:`~q2mm.benchmarks.runner.run_profiles` execution path:

- ``list`` — enumerate systems, backends, functional forms, and optimizers
  using only side-effect-free catalog probes (no device initialization).
- ``preflight`` — explicitly probe GPU/JAX/OpenMM device availability.
- ``single`` — run one (system, backend, form, optimizer) profile.
- ``batch`` — run the convergence workflow across one or more systems.
- ``matrix`` — run a backend x form x optimizer matrix for a system.
- ``load`` — load and summarise previously persisted candidate records.

Every run persists one immutable record per candidate (accepted, rejected,
skipped, or errored) and promotes only accepted candidates to canonical
result and force-field names.
"""

from __future__ import annotations

import logging
import platform
import sys
from itertools import product
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, computed_field
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliPositionalArg,
    CliSubCommand,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
    get_subcommand,
)

from q2mm.benchmarks import artifacts
from q2mm.benchmarks.acceptance import CandidateStatus
from q2mm.benchmarks.profiles import (
    FUNCTIONAL_FORMS,
    OPTIMIZER_CATALOG,
    BenchmarkSelection,
    BenchmarkSettings,
    RunProfile,
    catalog_settings,
)
from q2mm.benchmarks.publications import KNOWN_OBJECTIVE_PROFILES, PUBLICATION_SYSTEM_KEYS
from q2mm.benchmarks.records import RunOutcome
from q2mm.benchmarks.runner import run_profiles
from q2mm.settings import BackendSettings, Key, MethodE2Settings, Settings, WorkflowSettings

__all__ = ["main"]

logger = logging.getLogger("q2mm.benchmarks.cli")


# ---------------------------------------------------------------------------
# list (side-effect free)
# ---------------------------------------------------------------------------


def _cmd_list() -> int:
    from q2mm.backends.contracts import BackendRole
    from q2mm.backends.registry import catalog, discovery_report
    from q2mm.benchmarks.systems import SYSTEM_KEYS, system_metadata

    print("\nSystems:")
    for key in SYSTEM_KEYS:
        meta = system_metadata(key)
        print(f"  {key:<14} {meta.description}  [forms: {', '.join(meta.default_forms)}]")

    print("\nBackends (cheap probe, no device init):")
    for status in catalog(role=BackendRole.MM):
        desc = status.descriptor
        forms = ", ".join(sorted(desc.functional_form_ceiling)) or "n/a"
        health = "available" if status.healthy else f"unavailable ({status.reason})"
        print(f"  {desc.name:<14} {health}  [forms: {forms}]")

    # Surface isolated plugin-discovery issues (rejected/unavailable plugins)
    # without letting them break the listing above.  Cheap: probes only.
    issues = [record for record in discovery_report().issues if not record.registered]
    if issues:
        print("\nDiscovery issues (isolated; healthy backends unaffected):")
        for record in issues:
            source = record.entry_point or record.distribution or record.source.value
            label = record.name or source
            print(f"  {label:<14} {record.issue.value if record.issue else 'unknown'}: {record.message}")

    print("\nFunctional forms:")
    for form in FUNCTIONAL_FORMS:
        print(f"  {form}")

    print("\nPublication objective profiles:")
    for profile in sorted(KNOWN_OBJECTIVE_PROFILES):
        print(f"  {profile}")

    print("\nOptimizers:")
    print(f"  {'KEY':<20} {'LABEL':<28} METHOD (evaluator)")
    for spec in OPTIMIZER_CATALOG.values():
        print(f"  {spec.key:<20} {spec.label:<28} {spec.method} ({spec.evaluator})")
    print()
    return 0


# ---------------------------------------------------------------------------
# preflight (explicit device probe)
# ---------------------------------------------------------------------------


def _cmd_preflight() -> int:
    from q2mm import __version__

    print("=== Q2MM Benchmark Pre-flight Check ===\n")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"OS:       {platform.platform()}")
    print(f"q2mm:     {__version__}\n")

    has_openmm_cuda = has_openmm = jax_installed = has_jax_cuda = probe_failed = False
    try:
        import openmm as mm

        names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
        has_openmm = True
        has_openmm_cuda = "CUDA" in names
        print(f"OpenMM platforms: {', '.join(names)}")
        if has_openmm_cuda:
            print("OpenMM CUDA: available")
        elif "OpenCL" in names:
            print("OpenMM CUDA: NOT available (only OpenCL - slow, ~14% GPU utilization)")
        else:
            print("OpenMM GPU: NOT available")
    except ImportError:
        print("OpenMM: not installed")
    except (RuntimeError, OSError) as exc:  # pragma: no cover - environment dependent
        has_openmm = True
        probe_failed = True
        print(f"OpenMM: installed but platform probe failed: {exc}")
    print()

    try:
        import jax

        jax_installed = True
        device_strs = [str(d) for d in jax.devices()]
        has_jax_cuda = any("cuda" in s.lower() for s in device_strs)
        print(f"JAX devices: {', '.join(device_strs)}")
        print("JAX CUDA: available" if has_jax_cuda else "JAX CUDA: NOT available (CPU only)")
    except ImportError:
        print("JAX: not installed")
    except (RuntimeError, OSError) as exc:  # pragma: no cover - environment dependent
        jax_installed = True
        probe_failed = True
        print(f"JAX: installed but device probe failed: {exc}")
    print()

    if has_openmm_cuda and has_jax_cuda:
        print("Ready for GPU benchmarks.")
    else:
        missing = []
        if has_openmm and not has_openmm_cuda:
            missing.append("OpenMM CUDA")
        if jax_installed and not has_jax_cuda:
            missing.append("JAX CUDA")
        note = f" ({', '.join(missing)} missing)" if missing else ""
        print(f"GPU benchmarks will be slow or unavailable{note}. Consider WSL2.")
        if sys.platform == "win32" and jax_installed and not has_jax_cuda:
            print("Note: JAX CUDA is not available on Windows; use WSL2 for the full GPU stack.")
    print()
    return 1 if probe_failed else 0


# ---------------------------------------------------------------------------
# Shared profile construction
# ---------------------------------------------------------------------------


def _summarise(outcome: RunOutcome) -> None:
    counts = {status: len(outcome.by_status(status)) for status in CandidateStatus}
    print("\nCandidate outcomes:")
    for status in CandidateStatus:
        print(f"  {status.value:<9} {counts[status]}")
    for candidate in outcome.candidates:
        summary = candidate.summary
        line = f"  [{candidate.status.value:<8}] {candidate.candidate_id}"
        if "improvement_pct" in summary:
            line += f"  improvement={summary['improvement_pct']:.2f}%"
        if candidate.status is not CandidateStatus.ACCEPTED:
            line += f"  ({candidate.reason})"
        print(line)


def _cmd_load(args: LoadCommand) -> int:
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1
    candidates = artifacts.load_candidates(directory)
    if not candidates:
        print(f"No candidate records found under {directory}", file=sys.stderr)
        return 1
    print(f"Loaded {len(candidates)} candidate record(s) from {directory}\n")
    for cand in candidates:
        summary = cand.summary
        line = f"  [{cand.status.value:<8}] {cand.candidate_id}"
        if "improvement_pct" in summary:
            line += f"  improvement={summary['improvement_pct']}"
        if cand.reason:
            line += f"  ({cand.reason})"
        print(line)
    return 0


# ---------------------------------------------------------------------------
# Native model-driven command wiring
# ---------------------------------------------------------------------------


class ListCommand(Settings):
    """List systems, backend/plugin probes, forms, and optimizer presets."""

    def cli_cmd(self) -> int:
        """Present the cheap catalogs."""
        return _cmd_list()


class PreflightCommand(Settings):
    """Explicitly probe OpenMM platforms and JAX devices."""

    def cli_cmd(self) -> int:
        """Probe devices only when requested."""
        return _cmd_preflight()


class RunCommandOptions(Settings):
    """Presentation and persistence controls, not numerical policy."""

    output: Path = Field(default=Path("results"), strict=False)
    analyze: bool = True
    promote: bool = True
    show_config: bool = Field(default=False, description="Dump typed requested policy without executing or saving.")

    def run(self, profiles: list[RunProfile], command: str) -> int:
        """Use the single benchmark coordinator and propagate its outcome."""
        outcome = run_profiles(
            profiles,
            output_dir=self.output,
            generator=f"q2mm-benchmark {command}",
            analyze=self.analyze,
            promote=self.promote,
        )
        _summarise(outcome)
        print(f"\nResults saved under: {self.output}")
        return 0 if outcome.ok else 1


class SingleCommand(RunProfile, RunCommandOptions):
    """Run one system with shared typed backend, optimizer, objective, and workflow settings."""

    def cli_cmd(self) -> int:
        """Inspect or execute one profile."""
        if self.show_config:
            print(self.model_dump_json(indent=2))
            return 0
        profile = RunProfile(**self.model_dump(exclude=set(RunCommandOptions.model_fields)))
        return self.run([profile], "single")


class BatchCommand(BenchmarkSelection, BenchmarkSettings, RunCommandOptions):
    """Run the same policy across repeated systems (default: all registered systems)."""

    system: list[Key] = Field(default_factory=list)
    workflow: WorkflowSettings = MethodE2Settings()
    analyze: bool = False

    def model_post_init(self, context: Any) -> None:
        """Validate the requested publication selection before dispatch."""
        super().model_post_init(context)
        if self.objective_profile is not None:
            from q2mm.benchmarks.systems import SYSTEM_KEYS

            if set(self.system or SYSTEM_KEYS) - PUBLICATION_SYSTEM_KEYS:
                raise ValueError("objective_profile applies only to registered publication systems.")

    def cli_cmd(self) -> int:
        """Inspect or execute a convergence batch."""
        if self.show_config:
            print(self.model_dump_json(indent=2))
            return 0
        from q2mm.benchmarks.systems import SYSTEM_KEYS

        systems = self.system or list(SYSTEM_KEYS)
        unknown = set(systems) - set(SYSTEM_KEYS)
        if unknown:
            print(f"Error: unknown systems {sorted(unknown)}. Available: {sorted(SYSTEM_KEYS)}", file=sys.stderr)
            return 1
        values = self.model_dump(exclude=set(RunCommandOptions.model_fields) | {"system"})
        return self.run([RunProfile(system=system, **values) for system in systems], "batch")


class MatrixCommand(BenchmarkSelection, RunCommandOptions):
    """Run catalog presets across repeated backends and forms; omitted selectors expand at execution."""

    system: Key
    backends: list[Key | BackendSettings] = Field(default_factory=list)
    forms: list[Literal["harmonic", "mm3"]] = Field(default_factory=list)
    presets: list[str] = Field(default_factory=lambda: list(OPTIMIZER_CATALOG))

    def model_post_init(self, context: Any) -> None:
        """Reject unsupported selectors without backend construction."""
        super().model_post_init(context)
        if self.objective_profile is not None and self.system not in PUBLICATION_SYSTEM_KEYS:
            raise ValueError("objective_profile applies only to registered publication systems.")
        if self.functional_form is not None:
            raise ValueError("matrix uses repeated --forms, not --functional-form.")
        if not self.presets:
            raise ValueError("matrix requires at least one preset.")
        for key in self.presets:
            catalog_settings(key)
        if self.backends:
            self._profiles(self.backends, list(self.forms) or [None])

    def _profiles(self, backends: list[str | BackendSettings], forms: list[str | None]) -> list[RunProfile]:
        selection = self.model_dump(include=set(BenchmarkSelection.model_fields) - {"functional_form"})
        return [
            RunProfile(
                system=self.system,
                **{
                    **policy.model_dump(),
                    "backend": BackendSettings(key=backend) if isinstance(backend, str) else backend,
                    "functional_form": form,
                },
                **selection,
            )
            for backend, form, policy in product(backends, forms, self.policies)
        ]

    def cli_cmd(self) -> int:
        """Inspect unresolved policy or expand and execute the catalog matrix."""
        if self.show_config:
            print(self.model_dump_json(indent=2))
            return 0
        from q2mm.backends.registry import available_mm_backends
        from q2mm.benchmarks.systems import SYSTEM_KEYS, system_metadata

        if self.system not in SYSTEM_KEYS:
            print(f"Error: unknown system {self.system!r}. Available: {sorted(SYSTEM_KEYS)}", file=sys.stderr)
            return 1
        backends = [
            BackendSettings(key=value) if isinstance(value, str) else value
            for value in (self.backends or available_mm_backends())
        ]
        if not backends:
            print("Error: no MM backends available.", file=sys.stderr)
            return 1
        forms: list[str] = list(self.forms) or list(system_metadata(self.system).default_forms)
        try:
            profiles = self._profiles(list(backends), list(forms))
        except ValidationError as exc:
            print(f"Error: incoherent matrix settings; no candidates executed: {exc}", file=sys.stderr)
            return 2
        print(f"Matrix: system={self.system} backends={backends} forms={forms} -> {len(profiles)} candidates")
        return self.run(profiles, "matrix")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def policies(self) -> list[BenchmarkSettings]:
        """The same actual catalog policies used for execution and inspection."""
        return [catalog_settings(key) for key in self.presets]


class LoadCommand(Settings):
    """Load persisted candidate records, including unchanged historical schemas."""

    directory: CliPositionalArg[Path] = Field(strict=False)

    def cli_cmd(self) -> int:
        """Present persisted outcomes."""
        return _cmd_load(self)


class BenchmarkCLI(BaseSettings):
    """Q2MM benchmark commands generated from the actual shared settings models."""

    model_config = SettingsConfigDict(
        extra="forbid",
        strict=True,
        cli_prog_name="q2mm-benchmark",
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_enforce_required=True,
        cli_parse_none_str="none",
        nested_model_default_partial_update=True,
        cli_use_class_docs_for_groups=True,
    )
    list: CliSubCommand[ListCommand]
    preflight: CliSubCommand[PreflightCommand]
    single: CliSubCommand[SingleCommand]
    batch: CliSubCommand[BatchCommand]
    matrix: CliSubCommand[MatrixCommand]
    load: CliSubCommand[LoadCommand]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use explicit CLI/init input only; data-root fallbacks belong to the loader."""
        return (init_settings,)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``q2mm-benchmark`` console script."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if stream is not None and hasattr(stream, "reconfigure") and (stream.encoding or "").lower() != "utf-8":
                stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    try:
        app = CliApp.run(BenchmarkCLI, cli_args=argv, cli_exit_on_error=False)
        command = get_subcommand(app, cli_exit_on_error=False)
    except (SettingsError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    if not isinstance(
        command, (ListCommand, PreflightCommand, SingleCommand, BatchCommand, MatrixCommand, LoadCommand)
    ):
        raise TypeError("Unexpected benchmark subcommand.")
    # CliApp ignores cli_cmd returns; explicit dispatch preserves the process status.
    return command.cli_cmd()


if __name__ == "__main__":
    raise SystemExit(main())
