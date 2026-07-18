"""Generic optimizer catalog and strict construction.

The catalog is dependency-light: optional optimizer implementations are
imported only when their entry is explicitly resolved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from q2mm.models.results import OptimizationResult, deep_freeze

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

        return OptimizationLoop(evaluator, space, verbose=False, **self._kwargs).run()


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
    if method in ("L-BFGS-B", "Nelder-Mead", "Powell"):
        return frozenset({"maxiter", "ftol", "fc_fraction", "eq_fraction"})
    if method == "cycling":
        return frozenset({"maxiter", "max_params", "max_cycles", "convergence"})
    if method.startswith("optax:"):
        return frozenset({"maxiter", "learning_rate"})
    if method.startswith("jaxopt:"):
        return frozenset({"maxiter"})
    if method.startswith("basinhopping"):
        return frozenset({"maxiter", "seed"})
    if method.startswith("multi:"):
        return frozenset({"maxiter", "seed"})
    return frozenset()


def optimizer_option_names(value: str | OptimizerSpec) -> frozenset[str]:
    """Return the accepted option keys for a catalog entry."""
    return _allowed_options(optimizer_spec(value).method)


def resolve_optimizer(
    value: str | OptimizerSpec,
    options: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct an optimizer and return its exact effective settings."""
    spec = optimizer_spec(value)
    supplied = dict(options or {})
    unknown = set(supplied) - _allowed_options(spec.method)
    if unknown:
        raise ValueError(f"Unknown options for optimizer {spec.key!r}: {sorted(unknown)}.")
    cfg = {**_COMMON_DEFAULTS, **supplied}
    method = spec.method
    extra = spec.extra
    maxiter = cfg["maxiter"]

    if method in ("L-BFGS-B", "Nelder-Mead", "Powell"):
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        effective = 500 if maxiter is None else int(maxiter)
        opt: Any = ScipyOptimizer(
            method=method,
            maxiter=effective,
            ftol=float(cfg["ftol"]),
            verbose=False,
            fc_fraction=cfg["fc_fraction"],
            eq_fraction=cfg["eq_fraction"],
        )
        return opt, {
            "kind": "scipy",
            "method": method,
            "maxiter": effective,
            "ftol": float(cfg["ftol"]),
            "gtol": opt.gtol,
            "maxls": opt.maxls,
            "eps": opt.eps,
            "fc_fraction": cfg["fc_fraction"],
            "eq_fraction": cfg["eq_fraction"],
            "use_bounds": opt.use_bounds,
            "analytical_parameter_scaling": "bound-normalized",
        }
    if method == "cycling":
        effective_cfg: dict[str, Any] = {
            "max_params": int(cfg["max_params"]),
            "convergence": float(cfg["convergence"]),
            "max_cycles": int(cfg["max_cycles"]),
        }
        if maxiter is not None:
            effective_cfg["full_maxiter"] = int(maxiter)
            effective_cfg["simp_maxiter"] = int(maxiter)
        if "full_method" in extra:
            effective_cfg["full_method"] = extra["full_method"]
        return _CyclingOptimizer(**effective_cfg), {"kind": "cycling", **effective_cfg}
    if method.startswith("optax:"):
        from q2mm.optimizers.optax import OptaxOptimizer

        steps = 2000 if maxiter is None else int(maxiter)
        kwargs: dict[str, Any] = {
            "optimizer": method.split(":", 1)[1],
            "max_steps": steps,
            "learning_rate": float(cfg["learning_rate"]),
            "verbose": False,
        }
        if "schedule" in extra:
            kwargs["schedule"] = extra["schedule"]
        return OptaxOptimizer(**kwargs), {
            "kind": "optax",
            "optimizer": kwargs["optimizer"],
            "max_steps": steps,
            "schedule": extra.get("schedule"),
            "learning_rate": float(cfg["learning_rate"]),
        }
    if method.startswith("jaxopt:"):
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        effective = 200 if maxiter is None else int(maxiter)
        name = method.split(":", 1)[1]
        return JaxOptOptimizer(method=name, maxiter=effective, verbose=False), {
            "kind": "jaxopt",
            "method": name,
            "maxiter": effective,
        }
    if method.startswith("basinhopping"):
        from q2mm.optimizers.basinhopping import BasinHoppingOptimizer

        local_maxiter = 200 if maxiter is None else int(maxiter)
        kwargs = {"verbose": False, "local_maxiter": local_maxiter, "seed": int(cfg["seed"])}
        if "niter" in extra:
            kwargs["niter"] = int(extra["niter"])
        if "T" in extra:
            kwargs["T"] = float(extra["T"])
        return BasinHoppingOptimizer(**kwargs), {
            "kind": "basinhopping",
            "local_maxiter": local_maxiter,
            "niter": extra.get("niter"),
            "T": extra.get("T"),
            "seed": int(cfg["seed"]),
        }
    if method.startswith("multi:"):
        from q2mm.optimizers.multistart import MultiStartOptimizer
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        inner_maxiter = 500 if maxiter is None else int(maxiter)
        inner_name = method.split(":", 1)[1]
        inner = ScipyOptimizer(method=inner_name, maxiter=inner_maxiter, verbose=False)
        kwargs = {"optimizer": inner, "verbose": False, "seed": int(cfg["seed"])}
        if "n_starts" in extra:
            kwargs["n_starts"] = int(extra["n_starts"])
        return MultiStartOptimizer(**kwargs), {
            "kind": "multistart",
            "inner_method": inner_name,
            "inner_maxiter": inner_maxiter,
            "n_starts": extra.get("n_starts"),
            "seed": int(cfg["seed"]),
        }
    raise ValueError(f"Unknown optimizer method {method!r}.")


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
