"""Pure named optimizer presets and explicit, lazy component construction."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from q2mm.settings import (
    BasinHoppingSettings,
    CyclingSettings,
    JaxMultiStartSettings,
    JaxOptSettings,
    MultiStartSettings,
    OptaxSettings,
    ObjectiveSettings,
    OptimizerSettings,
    ScipySettings,
    Settings,
    Key,
)

if TYPE_CHECKING:
    from q2mm.optimizers.protocols import _Optimizer


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

    key: Key
    label: Key
    settings: OptimizerSettings
    objective: ObjectiveSettings = ObjectiveSettings()

    @property
    def method(self) -> str:
        """Algorithm label used by existing records and command listings."""
        return optimizer_method(self.settings)


def _entry(
    key: str,
    label: str,
    settings: OptimizerSettings,
    objective: ObjectiveSettings = ObjectiveSettings(),
) -> OptimizerSpec:
    return OptimizerSpec(
        key=key,
        label=label,
        settings=settings,
        objective=objective,
    )


OPTIMIZER_CATALOG: Mapping[str, OptimizerSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            _entry("scipy-lbfgsb", "SciPy L-BFGS-B (SciPy FD)", ScipySettings(verbose=False)),
            _entry(
                "scipy-lbfgsb-jax",
                "SciPy L-BFGS-B (JAX grad)",
                ScipySettings(verbose=False),
                ObjectiveSettings(executor="jax"),
            ),
            _entry(
                "scipy-lbfgsb-fd",
                "SciPy L-BFGS-B (executor FD)",
                ScipySettings(verbose=False),
                ObjectiveSettings(gradient_mode="finite_difference"),
            ),
            _entry("scipy-nm", "Nelder-Mead", ScipySettings(method="Nelder-Mead", verbose=False)),
            _entry("scipy-powell", "Powell", ScipySettings(method="Powell", verbose=False)),
            _entry("grad-simp", "Grad-Simp", CyclingSettings(verbose=False)),
            _entry(
                "grad-simp-auto",
                "Grad-Simp (JAX grad)",
                CyclingSettings(verbose=False),
                ObjectiveSettings(executor="jax"),
            ),
            _entry(
                "optax-adam",
                "Optax Adam",
                OptaxSettings(max_steps=2000, verbose=False),
                ObjectiveSettings(executor="jax"),
            ),
            _entry(
                "optax-adam-cosine",
                "Optax Adam+cosine",
                OptaxSettings(max_steps=2000, schedule="cosine", verbose=False),
                ObjectiveSettings(executor="jax"),
            ),
            _entry(
                "optax-adagrad",
                "Optax AdaGrad",
                OptaxSettings(optimizer="adagrad", max_steps=2000, verbose=False),
                ObjectiveSettings(executor="jax"),
            ),
            _entry(
                "optax-sgd",
                "Optax SGD",
                OptaxSettings(optimizer="sgd", max_steps=2000, verbose=False),
                ObjectiveSettings(executor="jax"),
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
                "scipy-lbfgsb-l2",
                "SciPy L-BFGS-B+L2 (SciPy FD)",
                ScipySettings(verbose=False),
                ObjectiveSettings(regularization=0.01),
            ),
            _entry(
                "optax-adam-l2",
                "Optax Adam+L2",
                OptaxSettings(max_steps=2000, verbose=False),
                ObjectiveSettings(executor="jax", regularization=0.01),
            ),
            _entry(
                "jaxopt-lbfgs",
                "JaxOpt L-BFGS",
                JaxOptSettings(verbose=False),
                ObjectiveSettings(executor="jax", regularization=0.01),
            ),
            _entry(
                "jaxopt-lbfgsb",
                "JaxOpt L-BFGS-B",
                JaxOptSettings(method="lbfgsb", verbose=False),
                ObjectiveSettings(executor="jax"),
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


__all__ = [
    "OPTIMIZER_CATALOG",
    "OptimizerSpec",
    "build_optimizer",
    "optimizer_method",
    "optimizer_spec",
]
