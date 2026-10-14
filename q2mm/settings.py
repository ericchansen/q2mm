"""Immutable declarative choices shared by Python applications and commands.

Settings describe requested or effective policy, never loaded engines,
scientific arrays, or numerical sessions. Scientific fingerprints use their
existing explicit projections, not arbitrary model dumps.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from q2mm.constants import KCAL_TO_KJ, MDYNA_TO_KJMOLA2, MM3_STR


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


JsonOptions = Annotated[
    Mapping[str, JsonValue],
    BeforeValidator(_thaw),
    AfterValidator(_freeze),
    PlainSerializer(_thaw, return_type=dict),
]
PathOptions = Annotated[
    Mapping[str, str],
    AfterValidator(_freeze),
    PlainSerializer(_thaw, return_type=dict),
]
Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]
Fraction = Annotated[float, Field(gt=0, le=1)]
Iterations = Annotated[int, Field(ge=0)]
Count = Annotated[int, Field(ge=1)]
Key = Annotated[str, Field(min_length=1)]


class Settings(BaseModel):
    """Strict, finite, deeply immutable declarative data."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False, validate_default=True)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        """Copy the model shell while safely sharing its immutable nested data."""
        return self.__copy__()


class QFuerzaSettings(Settings):
    """Hessian-informed initialization controls (Farrugia et al., 2025)."""

    strategy: Literal["fuerza", "qfuerza"] = "qfuerza"
    zero_torsions: bool = True
    au_hessian: bool = True
    invalid_policy: Literal["keep", "skip"] = "keep"
    replace_with: Positive = 1.0


class PreparationSettings(Settings):
    """Scientific construction choices; no backend work occurs in prepare."""

    stationary_point: Literal["ground_state", "transition_state"]
    functional_form: Literal["harmonic", "mm3"] | None = None
    initialize: Literal["provided", "qfuerza"] | None = None
    qfuerza: QFuerzaSettings | None = Field(
        default_factory=lambda data: None if data.get("initialize") == "provided" else QFuerzaSettings()
    )

    def model_post_init(self, context: Any) -> None:
        """Validate effective initialization controls, independently of field origin."""
        if self.initialize == "provided" and self.qfuerza is not None:
            raise ValueError("qfuerza settings cannot be applied with initialize='provided'.")
        if self.initialize != "provided" and self.qfuerza is None:
            raise ValueError("QFUERZA initialization requires qfuerza settings.")


class FrequencyMatchingSettings(Settings):
    """Real-mode threshold and observation weight for explicit matching."""

    threshold: float = 50.0
    weight: Positive = 0.001


class BackendSettings(Settings):
    """An open registered backend key and finite JSON factory options."""

    key: Key = "jax"
    options: JsonOptions = Field(default_factory=dict)

    def factory_options(self) -> dict[str, Any]:
        """Return ordinary recursive JSON containers for the backend factory."""
        return dict(self.model_dump(mode="json")["options"])


class ObjectiveSettings(Settings):
    """Concrete objective executor and explicit gradient policy."""

    executor: Literal["python", "jax"] = "python"
    gradient_mode: Literal["none", "finite_difference", "analytical"] = Field(
        default_factory=lambda data: "analytical" if data.get("executor") == "jax" else "none"
    )
    fd_step: Positive | None = Field(
        default_factory=lambda data: 1e-4 if data.get("gradient_mode") == "finite_difference" else None
    )
    regularization: NonNegative = 0.0

    def model_post_init(self, context: Any) -> None:
        """Reject gradient policies the selected executor cannot provide."""
        if self.executor == "jax" and self.gradient_mode != "analytical":
            raise ValueError("JAX execution requires analytical gradients.")
        if self.executor == "python" and self.gradient_mode == "analytical":
            raise ValueError("Python execution cannot provide analytical gradients.")
        if self.gradient_mode == "finite_difference" and self.fd_step is None:
            raise ValueError("Finite-difference execution requires an fd_step.")
        if self.gradient_mode != "finite_difference" and self.fd_step is not None:
            raise ValueError("fd_step applies only to finite-difference executor configurations.")


class ScipySettings(Settings):
    """SciPy solver controls, including explicit sanity/fractional bounds."""

    kind: Literal["scipy"] = "scipy"
    method: Key = "L-BFGS-B"
    maxiter: Iterations = 500
    ftol: NonNegative = 1e-8
    gtol: Positive = 1e-5
    maxls: Count = 100
    eps: Positive = 1e-3
    use_bounds: bool = True
    verbose: bool = True
    divergence_factor: NonNegative | None = 3.0
    divergence_patience: Iterations = 5
    fc_fraction: Fraction | None = None
    eq_fraction: Fraction | None = None


class OptaxSettings(Settings):
    """Optax training controls; gradients must be supplied explicitly."""

    kind: Literal["optax"] = "optax"
    optimizer: Literal["adam", "adamw", "adagrad", "sgd"] = "adam"
    learning_rate: NonNegative = 1e-3
    max_steps: Iterations = 1000
    ftol: NonNegative = 1e-8
    grad_norm_tol: NonNegative = 1e-6
    patience: Iterations = 20
    momentum: Annotated[float, Field(ge=0, lt=1)] = 0.9
    b1: Annotated[float, Field(ge=0, lt=1)] = 0.9
    b2: Annotated[float, Field(ge=0, lt=1)] = 0.999
    schedule: Literal["cosine", "exponential"] | None = None
    decay_rate: Positive = 0.99
    use_bounds: bool = True
    divergence_factor: NonNegative | None = 3.0
    divergence_patience: Iterations = 10
    verbose: bool = True
    log_interval: Count = 50


class JaxOptSettings(Settings):
    """JaxOpt per-case-JIT solver controls."""

    kind: Literal["jaxopt"] = "jaxopt"
    method: Literal["lbfgs", "lbfgsb", "gradient_descent"] = "lbfgs"
    maxiter: Iterations = 200
    tol: NonNegative = 1e-6
    verbose: bool = True


class BasinHoppingSettings(Settings):
    """Basin-hopping and local SciPy controls."""

    kind: Literal["basinhopping"] = "basinhopping"
    niter: Iterations = 50
    T: NonNegative = 1.0
    stepsize: NonNegative = 0.5
    local_method: Key = "L-BFGS-B"
    local_maxiter: Iterations = 200
    seed: int | None = None
    verbose: bool = True


class MultiStartSettings(Settings):
    """Sequential multi-start controls and its configured inner algorithm."""

    kind: Literal["multistart"] = "multistart"
    optimizer: ScipySettings | OptaxSettings | JaxOptSettings | BasinHoppingSettings = Field(
        default_factory=ScipySettings
    )
    n_starts: Count = 5
    perturbation_pct: NonNegative = 0.1
    seed: int | None = None
    verbose: bool = True


class JaxMultiStartSettings(Settings):
    """JaxOpt-specific construction of the shared multi-start algorithm."""

    kind: Literal["jax-multistart"] = "jax-multistart"
    optimizer: JaxOptSettings = Field(default_factory=lambda: JaxOptSettings(verbose=False))
    n_starts: Count = 10
    perturbation_pct: NonNegative = 0.1
    seed: int | None = None
    verbose: bool = True


class CyclingSettings(Settings):
    """Sensitivity cycling with explicit full and subspace phase choices."""

    kind: Literal["cycling"] = "cycling"
    max_params: Count = 3
    convergence: NonNegative = 0.01
    max_cycles: Iterations = 10
    sensitivity_metric: Literal["simp_var", "abs_d1"] = "simp_var"
    verbose: bool = True
    full_optimizer: ScipySettings | OptaxSettings | JaxOptSettings | BasinHoppingSettings | MultiStartSettings = Field(
        default_factory=lambda: ScipySettings(maxiter=200, verbose=False)
    )
    simplex_optimizer: ScipySettings = Field(
        default_factory=lambda: ScipySettings(method="Nelder-Mead", maxiter=200, verbose=False)
    )


OptimizerSettings = Annotated[
    ScipySettings
    | OptaxSettings
    | JaxOptSettings
    | BasinHoppingSettings
    | MultiStartSettings
    | JaxMultiStartSettings
    | CyclingSettings,
    Field(discriminator="kind"),
]


class SingleStageSettings(Settings):
    """One optimizer pass with configured endpoint sampling."""

    name: Literal["single-stage"] = "single-stage"


_NearZero = Annotated[
    Mapping[Literal["bond_k", "angle_k", "ub_k"], NonNegative],
    AfterValidator(_freeze),
    PlainSerializer(_thaw, return_type=dict),
]


def _near_zero_defaults() -> Mapping[Literal["bond_k", "angle_k", "ub_k"], float]:
    return {
        "bond_k": 5.0 * (0.5 * MDYNA_TO_KJMOLA2 / KCAL_TO_KJ),
        "angle_k": 0.5 * (0.5 * MM3_STR / KCAL_TO_KJ),
        "ub_k": 5.0 * (0.5 * MDYNA_TO_KJMOLA2 / KCAL_TO_KJ),
    }


class MethodE2Settings(Settings):
    """Existing Method E2 thresholds and physical force-constant replacements."""

    name: Literal["method-e2"] = "method-e2"
    negative_fc_threshold: NonNegative = 1e-3
    replace_with_round2: Positive = 1.0
    allow_negative: bool = False
    near_zero_replace_with: _NearZero = Field(default_factory=_near_zero_defaults)


WorkflowSettings = Annotated[SingleStageSettings | MethodE2Settings, Field(discriminator="name")]


class OptimizationSettings(Settings, BaseSettings):
    """Final declarative execution choices, not runtime objects or verified facts.

    Python input uses Pydantic's native nested partial-default updates. Ambient
    environment, dotenv and secret-file sources are deliberately disabled.
    Supplying a complete typed nested model replaces it; partial dictionaries
    are validated against and update the declared typed default.
    """

    model_config = SettingsConfigDict(nested_model_default_partial_update=True)

    recipe_id: Key = "explicit-v1"
    backend: BackendSettings = BackendSettings()
    optimizer: OptimizerSettings | None = None
    workflow: WorkflowSettings = SingleStageSettings()
    objective: ObjectiveSettings = ObjectiveSettings()
    n_evals: Iterations = 1
    ratio_tol: NonNegative | None = None

    @model_validator(mode="after")
    def validate_objective_backend(self) -> Self:
        """Reject typed combinations that cannot execute, before runtime construction."""
        if self.objective.executor == "jax" and self.backend.key != "jax":
            raise ValueError(
                "The JAX objective requires backend='jax'; explicitly select the Python objective for other backends."
            )
        pending = [self.optimizer] if self.optimizer is not None else []
        while pending:
            optimizer = pending.pop()
            if isinstance(optimizer, (JaxOptSettings, JaxMultiStartSettings)):
                if self.objective.executor != "jax" or self.objective.gradient_mode != "analytical":
                    raise ValueError(f"{optimizer.kind} requires a JAX analytical objective.")
            elif isinstance(optimizer, OptaxSettings):
                if self.objective.gradient_mode == "none":
                    raise ValueError("Optax requires a gradient-capable objective.")
            elif isinstance(optimizer, MultiStartSettings):
                pending.append(optimizer.optimizer)
            elif isinstance(optimizer, CyclingSettings):
                pending.extend((optimizer.full_optimizer, optimizer.simplex_optimizer))
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keep source precedence library-owned and isolate ambient configuration."""
        return (init_settings,)


class GroundStateOptimizationSettings(OptimizationSettings):
    """Measured SDK ground-state JAX policy, with native validated partial updates."""

    recipe_id: Key = "recommended-jax-gs-v1"
    optimizer: OptimizerSettings = ScipySettings(verbose=False)
    objective: ObjectiveSettings = ObjectiveSettings(executor="jax", gradient_mode="analytical")
    ratio_tol: NonNegative | None = 10.0


class TransitionStateOptimizationSettings(OptimizationSettings):
    """SDK TS JAX policy; publication/Heck caller policies remain separate."""

    recipe_id: Key = "recommended-jax-ts-v1"
    optimizer: OptimizerSettings = ScipySettings(
        ftol=1e-12,
        fc_fraction=0.20,
        eq_fraction=0.05,
        verbose=False,
    )
    objective: ObjectiveSettings = ObjectiveSettings(executor="jax", gradient_mode="analytical")


__all__ = [
    "BackendSettings",
    "BasinHoppingSettings",
    "CyclingSettings",
    "FrequencyMatchingSettings",
    "JaxMultiStartSettings",
    "JaxOptSettings",
    "MethodE2Settings",
    "MultiStartSettings",
    "ObjectiveSettings",
    "OptaxSettings",
    "OptimizationSettings",
    "GroundStateOptimizationSettings",
    "TransitionStateOptimizationSettings",
    "OptimizerSettings",
    "PreparationSettings",
    "QFuerzaSettings",
    "ScipySettings",
    "Settings",
    "SingleStageSettings",
    "WorkflowSettings",
]
