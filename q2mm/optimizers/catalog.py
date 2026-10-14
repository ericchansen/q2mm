"""Pure named optimizer presets and explicit, lazy component construction."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from q2mm.settings import (
    BasinHoppingSettings,
    CyclingSettings,
    JaxMultiStartSettings,
    JaxOptSettings,
    MultiStartSettings,
    OptaxSettings,
    OptimizerOverrides,
    OptimizerSettings,
    ScipySettings,
    Settings,
    Positive,
    NonNegative,
)

if TYPE_CHECKING:
    from q2mm.optimizers.protocols import _Optimizer

EVALUATORS = frozenset({"python", "jax"})
GRADIENT_MODES = frozenset({"analytical", "finite_difference", "none"})


def optimizer_method(settings: OptimizerSettings) -> str:
    """Return the explicit algorithm label, without importing implementations."""
    if isinstance(settings, ScipySettings):
        return settings.method
    if isinstance(settings, OptaxSettings):
        return f"optax:{settings.optimizer}"
    if isinstance(settings, JaxOptSettings):
        return f"jaxopt:{settings.method}"
    if isinstance(settings, MultiStartSettings):
        return f"multi:{optimizer_method(settings.optimizer)}"
    return settings.kind


class OptimizerSpec(Settings):
    """One immutable named policy; the solver choices are actual settings."""

    key: str
    label: str
    settings: OptimizerSettings
    evaluator: Literal["python", "jax"]
    gradient_mode: Literal["none", "analytical", "finite_difference"] = "none"
    fd_step: Positive = 1e-4
    regularization: NonNegative = 0.0

    @property
    def method(self) -> str:
        """Algorithm label used by existing records and command listings."""
        return optimizer_method(self.settings)

    def model_post_init(self, context: Any) -> None:
        """Validate numerical executor policy independently of construction."""
        from q2mm.settings import ObjectiveSettings

        ObjectiveSettings(
            executor=self.evaluator,
            gradient_mode=self.gradient_mode,
            fd_step=self.fd_step if self.gradient_mode == "finite_difference" else None,
            regularization=self.regularization,
        )
        if not self.key or not self.label:
            raise ValueError("OptimizerSpec key and label must be non-empty.")


def _entry(
    key: str,
    label: str,
    settings: OptimizerSettings,
    evaluator: Literal["python", "jax"] = "python",
    gradient_mode: Literal["none", "analytical", "finite_difference"] = "none",
    regularization: float = 0.0,
) -> OptimizerSpec:
    return OptimizerSpec(
        key=key,
        label=label,
        settings=settings,
        evaluator=evaluator,
        gradient_mode=gradient_mode,
        regularization=regularization,
    )


OPTIMIZER_CATALOG: Mapping[str, OptimizerSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            _entry("scipy-lbfgsb", "SciPy L-BFGS-B (SciPy FD)", ScipySettings(verbose=False)),
            _entry("scipy-lbfgsb-jax", "SciPy L-BFGS-B (JAX grad)", ScipySettings(verbose=False), "jax", "analytical"),
            _entry(
                "scipy-lbfgsb-fd",
                "SciPy L-BFGS-B (executor FD)",
                ScipySettings(verbose=False),
                "python",
                "finite_difference",
            ),
            _entry("scipy-nm", "Nelder-Mead", ScipySettings(method="Nelder-Mead", verbose=False)),
            _entry("scipy-powell", "Powell", ScipySettings(method="Powell", verbose=False)),
            _entry("grad-simp", "Grad-Simp", CyclingSettings(verbose=False)),
            _entry("grad-simp-auto", "Grad-Simp (JAX grad)", CyclingSettings(verbose=False), "jax", "analytical"),
            _entry("optax-adam", "Optax Adam", OptaxSettings(max_steps=2000, verbose=False), "jax", "analytical"),
            _entry(
                "optax-adam-cosine",
                "Optax Adam+cosine",
                OptaxSettings(max_steps=2000, schedule="cosine", verbose=False),
                "jax",
                "analytical",
            ),
            _entry(
                "optax-adagrad",
                "Optax AdaGrad",
                OptaxSettings(optimizer="adagrad", max_steps=2000, verbose=False),
                "jax",
                "analytical",
            ),
            _entry(
                "optax-sgd",
                "Optax SGD",
                OptaxSettings(optimizer="sgd", max_steps=2000, verbose=False),
                "jax",
                "analytical",
            ),
            _entry("basinhopping", "Basin-hopping (T=1.0)", BasinHoppingSettings(niter=25, seed=0, verbose=False)),
            _entry(
                "basinhopping-cold",
                "Basin-hopping (T=0.5)",
                BasinHoppingSettings(niter=25, T=0.5, seed=0, verbose=False),
            ),
            _entry(
                "multi-lbfgsb-5",
                "Multi-start n=5",
                MultiStartSettings(optimizer=ScipySettings(verbose=False), seed=0, verbose=False),
            ),
            _entry(
                "multi-lbfgsb-10",
                "Multi-start n=10",
                MultiStartSettings(optimizer=ScipySettings(verbose=False), n_starts=10, seed=0, verbose=False),
            ),
            _entry(
                "scipy-lbfgsb-l2", "SciPy L-BFGS-B+L2 (SciPy FD)", ScipySettings(verbose=False), regularization=0.01
            ),
            _entry(
                "optax-adam-l2",
                "Optax Adam+L2",
                OptaxSettings(max_steps=2000, verbose=False),
                "jax",
                "analytical",
                0.01,
            ),
            _entry("jaxopt-lbfgs", "JaxOpt L-BFGS", JaxOptSettings(verbose=False), "jax", "analytical", 0.01),
            _entry(
                "jaxopt-lbfgsb", "JaxOpt L-BFGS-B", JaxOptSettings(method="lbfgsb", verbose=False), "jax", "analytical"
            ),
            _entry(
                "grad-simp-multi",
                "Grad-Simp (multi inner)",
                CyclingSettings(
                    full_optimizer=MultiStartSettings(
                        optimizer=ScipySettings(maxiter=200, verbose=False), verbose=False
                    ),
                    verbose=False,
                ),
            ),
        )
    }
)


def optimizer_spec(key: str) -> OptimizerSpec:
    """Read a named preset without importing any optional algorithm."""
    try:
        return OPTIMIZER_CATALOG[key]
    except KeyError:
        raise ValueError(f"Unknown optimizer {key!r}; choose one of {sorted(OPTIMIZER_CATALOG)}.") from None


def optimizer_override_fields(settings: OptimizerSettings) -> frozenset[str]:
    """Shared command overrides applicable to a concrete algorithm family."""
    return {
        "scipy": frozenset({"maxiter", "ftol", "fc_fraction", "eq_fraction"}),
        "cycling": frozenset({"maxiter", "max_params", "max_cycles", "convergence"}),
        "optax": frozenset({"maxiter", "learning_rate"}),
        "jaxopt": frozenset({"maxiter"}),
        "basinhopping": frozenset({"maxiter", "seed"}),
        "multistart": frozenset({"maxiter", "seed"}),
        "jax-multistart": frozenset({"maxiter", "seed"}),
    }[settings.kind]


def _updated(settings: OptimizerSettings, changes: Mapping[str, Any]) -> OptimizerSettings:
    return type(settings).model_validate({**settings.model_dump(mode="python"), **changes})


def optimizer_settings(
    key: str,
    overrides: OptimizerOverrides = OptimizerOverrides(),
) -> OptimizerSettings:
    """Resolve a pure named policy and explicit overrides, with no runtime work."""
    settings = optimizer_spec(key).settings
    supplied = overrides.model_dump(include=overrides.model_fields_set)
    unknown = set(supplied) - optimizer_override_fields(settings)
    if unknown:
        raise ValueError(f"Unknown options for optimizer {key!r}: {sorted(unknown)}.")
    maxiter = supplied.pop("maxiter", None)
    if maxiter is not None:
        if isinstance(settings, CyclingSettings):
            supplied["full_optimizer"] = _with_iterations(settings.full_optimizer, maxiter)
            supplied["simplex_optimizer"] = _with_iterations(settings.simplex_optimizer, maxiter)
        else:
            settings = _with_iterations(settings, maxiter)
    return _updated(settings, supplied)


def _with_iterations(settings: OptimizerSettings, maxiter: int) -> OptimizerSettings:
    if isinstance(settings, (MultiStartSettings, JaxMultiStartSettings)):
        return _updated(settings, {"optimizer": _with_iterations(settings.optimizer, maxiter)})
    field = (
        "max_steps"
        if isinstance(settings, OptaxSettings)
        else ("local_maxiter" if isinstance(settings, BasinHoppingSettings) else "maxiter")
    )
    return _updated(settings, {field: maxiter})


def build_optimizer(settings: OptimizerSettings) -> _Optimizer:
    """Construct exactly the declared algorithm, importing only its implementation."""
    if isinstance(settings, ScipySettings):
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        return ScipyOptimizer(settings)
    if isinstance(settings, OptaxSettings):
        from q2mm.optimizers.optax import OptaxOptimizer

        return OptaxOptimizer(settings)
    if isinstance(settings, JaxOptSettings):
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        return JaxOptOptimizer(settings)
    if isinstance(settings, BasinHoppingSettings):
        from q2mm.optimizers.basinhopping import BasinHoppingOptimizer

        return BasinHoppingOptimizer(settings)
    if isinstance(settings, MultiStartSettings):
        from q2mm.optimizers.multistart import MultiStartOptimizer

        return MultiStartOptimizer(settings)
    if isinstance(settings, JaxMultiStartSettings):
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer

        return JaxMultiStartOptimizer(settings)
    if isinstance(settings, CyclingSettings):
        from q2mm.optimizers.cycling import CyclingOptimizer

        return CyclingOptimizer(settings)
    raise TypeError(f"Unsupported optimizer settings: {type(settings).__name__}.")


def expected_result_gradient(spec: OptimizerSpec) -> str:
    """Gradient provenance required by the declared named policy."""
    if isinstance(spec.settings, ScipySettings) and spec.settings.method in {"Nelder-Mead", "Powell"}:
        return "none"
    return "finite_difference" if spec.gradient_mode == "none" else spec.gradient_mode


__all__ = [
    "EVALUATORS",
    "GRADIENT_MODES",
    "OPTIMIZER_CATALOG",
    "OptimizerSpec",
    "build_optimizer",
    "expected_result_gradient",
    "optimizer_method",
    "optimizer_override_fields",
    "optimizer_settings",
    "optimizer_spec",
]
