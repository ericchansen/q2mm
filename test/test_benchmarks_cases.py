"""Tests for :class:`q2mm.benchmarks.cases.BenchmarkCase` deep immutability.

Domain-review Finding #3: ``BenchmarkCase`` must be deeply immutable —
``metadata``/``normal_modes`` as read-only mapping views (never the
caller's own dict), every reachable ``numpy.ndarray`` a read-only
defensive copy (never the caller's own array), and every sequence field
a tuple. These tests prove the source objects passed to the constructor
*and* the objects stored on the instance cannot be used to mutate it.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from test._shared import make_water

from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import OptimizationProblem, StationaryPointKind, TrainingCase


def _simple_ff() -> ForceField:
    return ForceField(
        bonds=(BondParam(elements=("H", "O"), force_constant=500.0, equilibrium=0.96),),
        functional_form=FunctionalForm.HARMONIC,
    )


def _simple_problem() -> OptimizationProblem:
    ff = _simple_ff()
    layout = ParameterLayout.from_force_field(ff)
    case = TrainingCase(
        case_id="water-000", molecule=make_water(name="water-000"), stationary_point=StationaryPointKind.GROUND_STATE
    )
    return OptimizationProblem(
        cases=(case,),
        starting_force_field=ff,
        layout=layout,
        active_space=ActiveParameterSpace.all_active(layout, ff),
        observations=ObservationSet(),
    )


def _make_case(**overrides: object) -> BenchmarkCase:
    kwargs: dict[str, object] = {
        "key": "test-system",
        "name": "Test System",
        "problem": _simple_problem(),
    }
    kwargs.update(overrides)
    return BenchmarkCase(**kwargs)  # type: ignore[arg-type]


class TestBenchmarkCaseIsFrozen:
    def test_uses_identity_equality_for_array_backed_metadata(self) -> None:
        first = _make_case(qm_freqs_per_mol=[np.array([100.0, 200.0])])
        second = _make_case(qm_freqs_per_mol=[np.array([100.0, 200.0])])

        assert first is not second
        assert first != second
        assert len({first, second}) == 2

    def test_cannot_assign_key(self) -> None:
        case = _make_case()
        with pytest.raises((AttributeError, TypeError)):
            case.key = "changed"  # type: ignore[misc]

    def test_cannot_assign_metadata(self) -> None:
        case = _make_case()
        with pytest.raises((AttributeError, TypeError)):
            case.metadata = {}  # type: ignore[misc]


class TestMetadataImmutability:
    def test_metadata_is_a_mapping_proxy(self) -> None:
        case = _make_case(metadata={"doi": "10.1000/xyz"})
        assert isinstance(case.metadata, MappingProxyType)

    def test_default_metadata_is_a_mapping_proxy(self) -> None:
        """Even the ``field(default_factory=dict)`` default must be frozen."""
        case = _make_case()
        assert isinstance(case.metadata, MappingProxyType)
        assert dict(case.metadata) == {}

    def test_metadata_rejects_item_assignment(self) -> None:
        case = _make_case(metadata={"doi": "10.1000/xyz"})
        with pytest.raises(TypeError):
            case.metadata["doi"] = "changed"  # type: ignore[index]

    def test_mutating_source_dict_after_construction_does_not_leak(self) -> None:
        """The stored metadata must be a snapshot, not a view of the caller's dict."""
        source = {"doi": "10.1000/xyz", "paper": "Original"}
        case = _make_case(metadata=source)

        source["paper"] = "Mutated after construction"
        source["new_key"] = "should not appear"

        assert case.metadata["paper"] == "Original"
        assert "new_key" not in case.metadata

    def test_nested_dict_within_metadata_is_recursively_frozen(self) -> None:
        """A dict-of-dicts must be frozen at every level, not just the top."""
        case = _make_case(metadata={"starting_point_audit": {"by_type": {"bond_k": {"frozen": 5}}}})

        assert isinstance(case.metadata["starting_point_audit"], MappingProxyType)
        assert isinstance(case.metadata["starting_point_audit"]["by_type"], MappingProxyType)
        with pytest.raises(TypeError):
            case.metadata["starting_point_audit"]["by_type"] = {}  # type: ignore[index]


class TestNormalModesImmutability:
    def test_normal_modes_is_a_mapping_proxy(self) -> None:
        case = _make_case(normal_modes={"eigenvalues": np.array([1.0, 2.0, 3.0])})
        assert isinstance(case.normal_modes, MappingProxyType)

    def test_none_normal_modes_stays_none(self) -> None:
        case = _make_case(normal_modes=None)
        assert case.normal_modes is None

    def test_normal_modes_rejects_item_assignment(self) -> None:
        case = _make_case(normal_modes={"eigenvalues": np.array([1.0, 2.0])})
        with pytest.raises(TypeError):
            case.normal_modes["eigenvalues"] = np.zeros(2)  # type: ignore[index]

    def test_normal_modes_array_is_read_only(self) -> None:
        case = _make_case(normal_modes={"eigenvalues": np.array([1.0, 2.0, 3.0])})
        stored = case.normal_modes["eigenvalues"]
        assert not stored.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            stored[0] = 999.0

    def test_mutating_source_array_after_construction_does_not_leak(self) -> None:
        source_array = np.array([1.0, 2.0, 3.0])
        case = _make_case(normal_modes={"eigenvalues": source_array})

        source_array[0] = 999.0  # mutate the caller's own array in place

        np.testing.assert_array_equal(case.normal_modes["eigenvalues"], [1.0, 2.0, 3.0])

    def test_mutating_source_dict_after_construction_does_not_leak(self) -> None:
        source_dict = {"eigenvalues": np.array([1.0, 2.0])}
        case = _make_case(normal_modes=source_dict)

        source_dict["eigenvalues"] = np.array([999.0, 999.0])
        source_dict["new_key"] = np.zeros(1)

        np.testing.assert_array_equal(case.normal_modes["eigenvalues"], [1.0, 2.0])
        assert "new_key" not in case.normal_modes


class TestQmFreqsPerMolImmutability:
    def test_is_a_tuple(self) -> None:
        case = _make_case(qm_freqs_per_mol=[np.array([100.0, 200.0])])
        assert isinstance(case.qm_freqs_per_mol, tuple)

    def test_elements_are_read_only_arrays(self) -> None:
        case = _make_case(qm_freqs_per_mol=[np.array([100.0, 200.0])])
        assert not case.qm_freqs_per_mol[0].flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            case.qm_freqs_per_mol[0][0] = 0.0

    def test_mutating_source_array_after_construction_does_not_leak(self) -> None:
        source_array = np.array([100.0, 200.0, 300.0])
        case = _make_case(qm_freqs_per_mol=[source_array])

        source_array[0] = -1.0

        np.testing.assert_array_equal(case.qm_freqs_per_mol[0], [100.0, 200.0, 300.0])

    def test_mutating_source_list_after_construction_does_not_leak(self) -> None:
        source_list = [np.array([100.0]), np.array([200.0])]
        case = _make_case(qm_freqs_per_mol=source_list)

        source_list.append(np.array([300.0]))
        source_list[0] = np.array([999.0])

        assert len(case.qm_freqs_per_mol) == 2
        np.testing.assert_array_equal(case.qm_freqs_per_mol[0], [100.0])


class TestDefaultFormsIsATuple:
    def test_default_forms_default_is_a_tuple(self) -> None:
        case = _make_case()
        assert case.default_forms == ("mm3",)
        assert isinstance(case.default_forms, tuple)

    def test_default_forms_list_input_is_converted_to_tuple(self) -> None:
        case = _make_case(default_forms=["mm3", "harmonic"])
        assert isinstance(case.default_forms, tuple)
        assert case.default_forms == ("mm3", "harmonic")

    def test_mutating_source_list_after_construction_does_not_leak(self) -> None:
        source_list = ["mm3"]
        case = _make_case(default_forms=source_list)

        source_list.append("harmonic")

        assert case.default_forms == ("mm3",)
