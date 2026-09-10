"""Native .hes parser proofs with mocked subprocess output, not native execution.

Section labels and packed row order follow Tinker's testhess.f (formats
370-440); wrapped lines contain scalars, not separate matrix sections:
https://github.com/TinkerTools/tinker/blob/543d4de6b6d723ace70b62858c6b432beb06e0c0/source/testhess.f
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess

import numpy as np
import pytest

from q2mm.backends.contracts import (
    EvaluationError,
    HessianRequest,
    HessianResult,
    HessianUnit,
    PreparationRequest,
)
from q2mm.backends.mm import tinker
from q2mm.constants import KCALMOLA2_TO_HESSIAN_AU
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.units import hessian_kcalmola2_to_au
from test._shared import make_diatomic


_DIAGONAL = " Diagonal Hessian Elements  (3 per Atom)\n\n 200 0 0 200 0 0\n"
_BLOCKS = (
    " Off-diagonal Hessian Elements for Atom     1 X\n\n 1 -2 3 0 5\n",
    " Off-diagonal Hessian Elements for Atom     1 Y\n\n 6 7 8 9\n",
    " Off-diagonal Hessian Elements for Atom     1 Z\n\n 10 11 12\n",
    " Off-diagonal Hessian Elements for Atom     2 X\n\n 13 14\n",
    " Off-diagonal Hessian Elements for Atom     2 Y\n\n 15\n",
)
_EXPECTED = np.array(
    [
        [200, 1, -2, 3, 0, 5],
        [1, 0, 6, 7, 8, 9],
        [-2, 6, 0, 10, 11, 12],
        [3, 7, 10, 200, 13, 14],
        [0, 8, 11, 13, 0, 15],
        [5, 9, 12, 14, 15, 0],
    ]
)


@pytest.fixture
def evaluate_hessian(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[str], HessianResult]:
    backend = tinker.TinkerBackend(tinker_dir=str(tmp_path), params_file=str(tmp_path / "unused.prm"))
    ff = ForceField(functional_form=FunctionalForm.MM3, bonds=(BondParam(("H", "H"), 0.74, 100.0),))
    prepared = backend.prepare(PreparationRequest(case_id="h2-parser", molecule=make_diatomic(), force_field=ff))
    monkeypatch.setattr(tinker, "_exe", lambda directory, name: str(Path(directory) / name))

    def evaluate(content: str) -> HessianResult:
        def run(
            cmd: list[str], *, capture_output: bool, text: bool, timeout: int, input: str, cwd: str
        ) -> subprocess.CompletedProcess[str]:
            assert Path(cmd[0]).name == "testhess"
            assert input == "Y\nN\n"
            assert capture_output and text
            assert timeout == 300
            assert Path(cmd[1]).parent == Path(cwd)
            Path(cmd[1]).with_suffix(".hes").write_text(content, encoding="ascii")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with monkeypatch.context() as process_patch:
            process_patch.setattr(tinker.subprocess, "run", run)
            return prepared.hessian(HessianRequest(parameters=prepared.layout.vector(ff)))

    return evaluate


def test_complete_native_hessian(evaluate_hessian: Callable[[str], HessianResult]) -> None:
    result = evaluate_hessian("\n" + _DIAGONAL + "\n".join(_BLOCKS))
    assert result.unit is HessianUnit.HARTREE_PER_BOHR2
    np.testing.assert_allclose(result.hessian, _EXPECTED * KCALMOLA2_TO_HESSIAN_AU)
    np.testing.assert_array_equal(result.hessian, result.hessian.T)
    assert not result.hessian.flags.writeable


def test_parser_uses_canonical_hessian_conversion(evaluate_hessian: Callable[[str], HessianResult]) -> None:
    result = evaluate_hessian("\n" + _DIAGONAL + "\n".join(_BLOCKS))
    np.testing.assert_allclose(result.hessian, _EXPECTED * hessian_kcalmola2_to_au(1.0), rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("width", [4, 5, 6])
@pytest.mark.parametrize("exponent", [None, "e", "D", "d"])
def test_wrapping_whitespace_and_exponents(
    evaluate_hessian: Callable[[str], HessianResult], width: int, exponent: str | None
) -> None:
    sections = []
    for section in (_DIAGONAL, *_BLOCKS):
        header, values = section.split("\n", maxsplit=1)
        if exponent is None:
            field_width, precision = {4: (16, 8), 5: (14, 6), 6: (12, 4)}[width]
            tokens = [f"{float(v):{field_width}.{precision}f}" for v in values.split()]
            separator = ""
        else:
            tokens = [f"{float(v):.8e}".replace("e", exponent) for v in values.split()]
            separator = " \t"
        wrapped = "\r\n\t".join(separator.join(tokens[i : i + width]) for i in range(0, len(tokens), width))
        sections.append(f"{header}\r\n\r\n\t{wrapped}\r\n")
    result = evaluate_hessian("\r\n".join(sections))
    np.testing.assert_allclose(result.hessian, _EXPECTED * KCALMOLA2_TO_HESSIAN_AU)


def test_complete_unlabeled_sections(evaluate_hessian: Callable[[str], HessianResult]) -> None:
    blocks = [" Off-diagonal Hessian Elements\n" + section.split("\n", maxsplit=1)[1] for section in _BLOCKS]
    result = evaluate_hessian("\n" + _DIAGONAL + "".join(blocks))
    np.testing.assert_allclose(result.hessian, _EXPECTED * KCALMOLA2_TO_HESSIAN_AU)


def test_complete_zero_off_diagonals(evaluate_hessian: Callable[[str], HessianResult]) -> None:
    blocks = []
    for row, section in enumerate(_BLOCKS):
        header = section.split("\n", maxsplit=1)[0]
        blocks.append(header + "\n" + " 0" * (5 - row) + "\n")
    result = evaluate_hessian("\n" + _DIAGONAL + "\n".join(blocks))
    np.testing.assert_allclose(result.hessian, np.diag([200, 0, 0, 200, 0, 0]) * KCALMOLA2_TO_HESSIAN_AU)


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        pytest.param("", "no diagonal section found", id="empty-file"),
        pytest.param("\n" + _DIAGONAL, "expected 5 off-diagonal blocks, got 0", id="diagonal-only"),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS[:-1]),
            "expected 5 off-diagonal blocks, got 4",
            id="missing-last-block",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS[:-1]) + _BLOCKS[-1].split("\n")[0],
            "expected 1 values, got 0",
            id="truncated-last-header",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS).replace("6 7 8 9", ""),
            "expected 4 values, got 0",
            id="empty-middle-block",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS).replace("6 7 8 9", "6 7 8"),
            "expected 4 values, got 3",
            id="missing-scalar",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS).replace("13 14", "13 14 99"),
            "expected 2 values, got 3",
            id="extra-scalar",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS) + _BLOCKS[-1],
            "expected 5 off-diagonal blocks, got 6",
            id="extra-block",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS) + _BLOCKS[-1].split("\n")[0] + "\n\n",
            "expected 5 off-diagonal blocks, got 6",
            id="extra-empty-block",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS).replace("1 Y", "1 X"),
            "expected Atom 1 Y",
            id="duplicate-row-label",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS).replace("2 X", "2 Z"),
            "expected Atom 2 X",
            id="out-of-order-row-label",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS).replace("2 Y", "3 Y"),
            "expected Atom 2 Y",
            id="out-of-range-atom",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS).replace(_BLOCKS[2], _DIAGONAL),
            "unexpected diagonal section",
            id="duplicate-diagonal",
        ),
        pytest.param("\n" + "".join(_BLOCKS), "no diagonal section found", id="missing-diagonal"),
        pytest.param(
            "\n" + _DIAGONAL.replace("200 0 0 200 0 0", "200 0 0 200 0") + "".join(_BLOCKS),
            "expected 6 diagonal values, got 5",
            id="short-diagonal",
        ),
        pytest.param(
            "\n" + _DIAGONAL.replace("200 0 0 200 0 0", "200 0 0 200 0 0 0") + "".join(_BLOCKS),
            "expected 6 diagonal values, got 7",
            id="long-diagonal",
        ),
        pytest.param(
            "\n" + _DIAGONAL.replace("200 0 0 200 0 0", "") + "".join(_BLOCKS),
            "expected 6 diagonal values, got 0",
            id="empty-diagonal",
        ),
        pytest.param(
            "\n" + _DIAGONAL + "".join(_BLOCKS).replace("13 14", "13 invalid"),
            "could not convert string to float",
            id="invalid-scalar",
        ),
    ],
)
def test_rejects_incomplete_or_misidentified_hessian(
    evaluate_hessian: Callable[[str], HessianResult], content: str, reason: str
) -> None:
    with pytest.raises(EvaluationError) as caught:
        evaluate_hessian(content)
    message = str(caught.value)
    assert "Tinker" in message
    assert "molecule.hes" in message
    assert reason in message
