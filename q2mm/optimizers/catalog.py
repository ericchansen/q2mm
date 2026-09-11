"""Generic optimizer catalog and strict construction.

The catalog is dependency-light: optional optimizer implementations are
imported only when their entry is explicitly resolved. Effective settings
include the Q2MM constructors' bound defaults and nested Q2MM solvers;
they do not inspect arbitrary custom optimizer objects or runtime state.
Catalog and cycling policies supply explicit options to one bound
construction graph used for both object creation and settings capture.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from q2mm.models.results import OptimizationResult, deep_freeze

if TYPE_CHECKING:
    from q2mm.optimizers.protocols import _Optimizer

EVALUATORS = frozenset({"python", "jax"})
GRADIENT_MODES = frozenset({"analytical", "finite_difference", "none"})


@dataclass(frozen=True, eq=False)
class OptimizerSpec:
    """One immutable registered optimizer configuration."""

    key: str
    label: str
    method: str
    evaluator: str
    gradient_mode: str = "none"
    fd_step: float = 1e-4
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.label or not self.method:
            raise ValueError("OptimizerSpec key, label, and method must be non-empty.")
        if self.evaluator not in EVALUATORS:
            raise ValueError(f"OptimizerSpec.evaluator must be one of {sorted(EVALUATORS)}, got {self.evaluator!r}.")
        if self.gradient_mode not in GRADIENT_MODES:
            raise ValueError(f"OptimizerSpec.gradient_mode must be one of {sorted(GRADIENT_MODES)}.")
        if self.evaluator == "jax" and self.gradient_mode != "analytical":
            raise ValueError("A JAX-executor optimizer must declare gradient_mode='analytical'.")
        if not (math.isfinite(self.fd_step) and self.fd_step > 0.0):
            raise ValueError(f"OptimizerSpec.fd_step must be positive and finite, got {self.fd_step!r}.")
        object.__setattr__(self, "extra", deep_freeze(dict(self.extra)))

    @property
    def regularization(self) -> float:
        """L2 regularization requested by this catalog entry."""
        return float(self.extra.get("regularization", 0.0))


def _spec(key: str, label: str, method: str, evaluator: str, *, gradient_mode: str, **extra: Any) -> OptimizerSpec:
    return OptimizerSpec(
        key=key, label=label, method=method, evaluator=evaluator, gradient_mode=gradient_mode, extra=extra
    )


OPTIMIZER_CATALOG: Mapping[str, OptimizerSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            _spec("scipy-lbfgsb", "SciPy L-BFGS-B (SciPy FD)", "L-BFGS-B", "python", gradient_mode="none"),
            _spec("scipy-lbfgsb-jax", "SciPy L-BFGS-B (JAX grad)", "L-BFGS-B", "jax", gradient_mode="analytical"),
            _spec(
                "scipy-lbfgsb-fd",
                "SciPy L-BFGS-B (executor FD)",
                "L-BFGS-B",
                "python",
                gradient_mode="finite_difference",
            ),
            _spec("scipy-nm", "Nelder-Mead", "Nelder-Mead", "python", gradient_mode="none"),
            _spec("scipy-powell", "Powell", "Powell", "python", gradient_mode="none"),
            _spec("grad-simp", "Grad-Simp", "cycling", "python", gradient_mode="none"),
            _spec("grad-simp-auto", "Grad-Simp (JAX grad)", "cycling", "jax", gradient_mode="analytical"),
            _spec("optax-adam", "Optax Adam", "optax:adam", "jax", gradient_mode="analytical"),
            _spec(
                "optax-adam-cosine",
                "Optax Adam+cosine",
                "optax:adam",
                "jax",
                gradient_mode="analytical",
                schedule="cosine",
            ),
            _spec("optax-adagrad", "Optax AdaGrad", "optax:adagrad", "jax", gradient_mode="analytical"),
            _spec("optax-sgd", "Optax SGD", "optax:sgd", "jax", gradient_mode="analytical"),
            _spec(
                "basinhopping",
                "Basin-hopping (T=1.0)",
                "basinhopping",
                "python",
                gradient_mode="none",
                niter=25,
            ),
            _spec(
                "basinhopping-cold",
                "Basin-hopping (T=0.5)",
                "basinhopping",
                "python",
                gradient_mode="none",
                niter=25,
                T=0.5,
            ),
            _spec(
                "multi-lbfgsb-5",
                "Multi-start n=5",
                "multi:L-BFGS-B",
                "python",
                gradient_mode="none",
                n_starts=5,
            ),
            _spec(
                "multi-lbfgsb-10",
                "Multi-start n=10",
                "multi:L-BFGS-B",
                "python",
                gradient_mode="none",
                n_starts=10,
            ),
            _spec(
                "scipy-lbfgsb-l2",
                "SciPy L-BFGS-B+L2 (SciPy FD)",
                "L-BFGS-B",
                "python",
                gradient_mode="none",
                regularization=0.01,
            ),
            _spec(
                "optax-adam-l2",
                "Optax Adam+L2",
                "optax:adam",
                "jax",
                gradient_mode="analytical",
                regularization=0.01,
            ),
            _spec(
                "jaxopt-lbfgs",
                "JaxOpt L-BFGS",
                "jaxopt:lbfgs",
                "jax",
                gradient_mode="analytical",
                regularization=0.01,
            ),
            _spec("jaxopt-lbfgsb", "JaxOpt L-BFGS-B", "jaxopt:lbfgsb", "jax", gradient_mode="analytical"),
            _spec(
                "grad-simp-multi",
                "Grad-Simp (multi inner)",
                "cycling",
                "python",
                gradient_mode="none",
                full_method="multi:L-BFGS-B",
            ),
        )
    }
)


class _CyclingOptimizer:
    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def optimize(self, evaluator: Any, space: Any) -> OptimizationResult:
        from q2mm.optimizers.cycling import OptimizationLoop

        return OptimizationLoop(evaluator, space, **self._kwargs).run()


def _constructor_arguments(constructor: Callable[..., object], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Bind known built-in constructor arguments, including their declared defaults."""
    bound = inspect.signature(constructor).bind_partial(**overrides)
    bound.apply_defaults()
    return dict(bound.arguments)


def _constructor_settings(constructor: Callable[..., object], kind: str, **overrides: Any) -> dict[str, Any]:
    return {"kind": kind, **_constructor_arguments(constructor, overrides)}


def _scipy_scaling_settings(settings: dict[str, Any], gradient_mode: str) -> dict[str, Any]:
    applicable = settings["method"] == "L-BFGS-B" and settings["use_bounds"] and gradient_mode != "none"
    return {
        **settings,
        "analytical_parameter_scaling": "bound-normalized" if applicable else "none",
        "parameter_scaling_requires": "finite nondegenerate active bounds" if applicable else None,
    }


@dataclass(frozen=True)
class _Construction:
    """One bound constructor graph, shared by execution and settings capture."""

    constructor: Callable[..., _Optimizer]
    kind: str
    arguments: Mapping[str, object]
    extra_settings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(self, "extra_settings", MappingProxyType(dict(self.extra_settings)))

    def build(self) -> _Optimizer:
        """Construct this graph without consulting another resolver."""
        arguments = {
            name: value.build() if isinstance(value, _Construction) else value for name, value in self.arguments.items()
        }
        return self.constructor(**arguments)

    def settings(self, gradient_mode: str = "none") -> dict[str, object]:
        """Describe the same bound arguments without constructing any object."""
        settings = {
            "kind": self.kind,
            **{
                name: value.settings(gradient_mode) if isinstance(value, _Construction) else value
                for name, value in {**self.arguments, **self.extra_settings}.items()
            },
        }
        return _scipy_scaling_settings(settings, gradient_mode) if self.kind == "scipy" else settings


def _leaf(constructor: Callable[..., _Optimizer], kind: str, **arguments: object) -> _Construction:
    return _Construction(constructor, kind, _constructor_arguments(constructor, arguments))


_Syntax = Literal["catalog", "cycling", "scipy"]


def _method_family(method: str, syntax: _Syntax = "catalog") -> str:
    if syntax == "scipy":
        return "scipy"
    if method == "cycling" and syntax == "catalog":
        return "cycling"
    for prefix, family in (
        ("optax:", "optax"),
        ("jaxopt:", "jaxopt"),
        ("basinhopping", "basinhopping"),
        ("multi:", "multistart"),
    ):
        if method.startswith(prefix):
            return family
    return "scipy"


def _optimizer_construction(
    method: str,
    options: Mapping[str, Any],
    *,
    syntax: _Syntax = "catalog",
    inner_options: Mapping[str, Any] | None = None,
) -> _Construction:
    """Own method parsing, concrete constructor choice, and nested construction."""
    family = _method_family(method, syntax)
    if family == "cycling":
        from q2mm.optimizers.cycling import OptimizationLoop

        arguments = _constructor_arguments(OptimizationLoop, options)
        return _Construction(_CyclingOptimizer, family, arguments, _cycling_constructions(arguments))
    if family == "optax":
        from q2mm.optimizers.optax import OptaxOptimizer

        name = method.split(":", 1)[1]
        parsed: dict[str, Any] = {"optimizer": name}
        if syntax == "cycling" and "+" in name:
            parsed["optimizer"], parsed["schedule"] = name.split("+", 1)
        return _leaf(OptaxOptimizer, family, **{**parsed, **options})
    if family == "jaxopt":
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        return _leaf(JaxOptOptimizer, family, **{"method": method.split(":", 1)[1], **options})
    if family == "basinhopping":
        from q2mm.optimizers.basinhopping import BasinHoppingOptimizer

        parsed = {}
        if syntax == "cycling":
            parsed["local_method"] = (method.split(":", 1)[1].strip() or "L-BFGS-B") if ":" in method else "L-BFGS-B"
        return _leaf(BasinHoppingOptimizer, family, **{**parsed, **options})
    if family == "multistart":
        from q2mm.optimizers.multistart import MultiStartOptimizer

        inner = _optimizer_construction(
            method.split(":", 1)[1], {} if inner_options is None else inner_options, syntax="scipy"
        )
        plan = _leaf(MultiStartOptimizer, family, **{"optimizer": inner, **options})
        if syntax == "catalog":
            return _Construction(
                plan.constructor,
                plan.kind,
                plan.arguments,
                {"inner_method": inner.arguments["method"], "inner_maxiter": inner.arguments["maxiter"]},
            )
        return plan
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    return _leaf(ScipyOptimizer, family, **{"method": method, **options})


def _cycling_constructions(settings: Mapping[str, Any]) -> dict[str, _Construction]:
    """Supply cycling's explicit phase policies to the shared construction owner."""
    maxiter = settings["full_maxiter"]
    scipy_options = {"maxiter": maxiter, "eps": settings["eps"], "verbose": False}
    policies = {
        "scipy": scipy_options,
        "optax": {"max_steps": maxiter, "verbose": False},
        "jaxopt": {"maxiter": maxiter, "verbose": False},
        "basinhopping": {"local_maxiter": maxiter, "verbose": False},
        "multistart": {"n_starts": 5, "verbose": False},
    }
    family = _method_family(settings["full_method"], "cycling")
    return {
        "full_optimizer": _optimizer_construction(
            settings["full_method"], policies[family], syntax="cycling", inner_options=scipy_options
        ),
        "simplex_optimizer": _optimizer_construction(
            settings["simp_method"],
            {"maxiter": settings["simp_maxiter"], "eps": settings["eps"], "verbose": False},
            syntax="scipy",
        ),
    }


def _cycling_nested_settings(settings: Mapping[str, Any], gradient_mode: str) -> dict[str, Any]:
    """Serialize the same phase plans used by cycling execution."""
    return {name: plan.settings(gradient_mode) for name, plan in _cycling_constructions(settings).items()}


_COMMON_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "maxiter": None,
        "ftol": 1e-8,
        "fc_fraction": None,
        "eq_fraction": None,
        "learning_rate": 1e-3,
        "max_params": 3,
        "max_cycles": 10,
        "convergence": 0.01,
        "seed": 0,
    }
)


def optimizer_spec(value: str | OptimizerSpec) -> OptimizerSpec:
    """Resolve a catalog key or return a validated spec."""
    if isinstance(value, OptimizerSpec):
        return value
    try:
        return OPTIMIZER_CATALOG[value]
    except KeyError:
        raise ValueError(f"Unknown optimizer {value!r}; choose one of {sorted(OPTIMIZER_CATALOG)}.") from None


def _allowed_options(method: str) -> frozenset[str]:
    family = _method_family(method)
    if family == "scipy" and method not in ("L-BFGS-B", "Nelder-Mead", "Powell"):
        return frozenset()
    return {
        "scipy": frozenset({"maxiter", "ftol", "fc_fraction", "eq_fraction"}),
        "cycling": frozenset({"maxiter", "max_params", "max_cycles", "convergence"}),
        "optax": frozenset({"maxiter", "learning_rate"}),
        "jaxopt": frozenset({"maxiter"}),
        "basinhopping": frozenset({"maxiter", "seed"}),
        "multistart": frozenset({"maxiter", "seed"}),
    }[family]


def optimizer_option_names(value: str | OptimizerSpec) -> frozenset[str]:
    """Return the accepted option keys for a catalog entry."""
    return _allowed_options(optimizer_spec(value).method)


def _catalog_options(spec: OptimizerSpec, cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Supply named-catalog policy values, not a second constructor dispatch."""
    method = spec.method
    extra = spec.extra
    maxiter = cfg["maxiter"]
    family = _method_family(method)
    if family == "scipy":
        if method not in ("L-BFGS-B", "Nelder-Mead", "Powell"):
            raise ValueError(f"Unknown optimizer method {method!r}.")
        return {
            "maxiter": 500 if maxiter is None else int(maxiter),
            "ftol": float(cfg["ftol"]),
            "verbose": False,
            "fc_fraction": cfg["fc_fraction"],
            "eq_fraction": cfg["eq_fraction"],
        }, None
    if family == "cycling":
        arguments: dict[str, Any] = {
            "max_params": int(cfg["max_params"]),
            "convergence": float(cfg["convergence"]),
            "max_cycles": int(cfg["max_cycles"]),
            "verbose": False,
        }
        if maxiter is not None:
            arguments["full_maxiter"] = int(maxiter)
            arguments["simp_maxiter"] = int(maxiter)
        if "full_method" in extra:
            arguments["full_method"] = extra["full_method"]
        return arguments, None
    if family == "optax":
        arguments = {
            "max_steps": 2000 if maxiter is None else int(maxiter),
            "learning_rate": float(cfg["learning_rate"]),
            "verbose": False,
        }
        if "schedule" in extra:
            arguments["schedule"] = extra["schedule"]
        return arguments, None
    if family == "jaxopt":
        return {"maxiter": 200 if maxiter is None else int(maxiter), "verbose": False}, None
    if family == "basinhopping":
        arguments = {
            "verbose": False,
            "local_maxiter": 200 if maxiter is None else int(maxiter),
            "seed": int(cfg["seed"]),
        }
        if "niter" in extra:
            arguments["niter"] = int(extra["niter"])
        if "T" in extra:
            arguments["T"] = float(extra["T"])
        return arguments, None
    if family == "multistart":
        arguments = {"verbose": False, "seed": int(cfg["seed"])}
        if "n_starts" in extra:
            arguments["n_starts"] = int(extra["n_starts"])
        return arguments, {"maxiter": 500 if maxiter is None else int(maxiter), "verbose": False}
    raise ValueError(f"Unknown optimizer method {method!r}.")


def resolve_optimizer(
    value: str | OptimizerSpec,
    options: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve explicit catalog policy through the shared construction owner."""
    spec = optimizer_spec(value)
    supplied = dict(options or {})
    unknown = set(supplied) - _allowed_options(spec.method)
    if unknown:
        raise ValueError(f"Unknown options for optimizer {spec.key!r}: {sorted(unknown)}.")
    cfg = {**_COMMON_DEFAULTS, **supplied}
    arguments, inner_arguments = _catalog_options(spec, cfg)
    plan = _optimizer_construction(spec.method, arguments, inner_options=inner_arguments)
    return plan.build(), plan.settings(spec.gradient_mode)


def expected_result_gradient(spec: OptimizerSpec) -> str:
    """Return the gradient provenance an optimizer result must report."""
    if spec.evaluator == "jax":
        return "analytical"
    if spec.gradient_mode == "finite_difference":
        return "finite_difference"
    if spec.method in ("Nelder-Mead", "Powell"):
        return "none"
    return "finite_difference"


__all__ = [
    "EVALUATORS",
    "GRADIENT_MODES",
    "OPTIMIZER_CATALOG",
    "OptimizerSpec",
    "expected_result_gradient",
    "optimizer_option_names",
    "optimizer_spec",
    "resolve_optimizer",
]
