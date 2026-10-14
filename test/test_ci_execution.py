"""A required test job cannot succeed with an empty or wholly skipped report."""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.check_test_execution import check_execution


@pytest.mark.parametrize(
    ("cases", "expected"),
    [
        ("", 0),
        ("<testcase><skipped message='tier disabled'/></testcase>", 0),
        ("<testcase/><testcase><skipped/></testcase>", 1),
        ("<testcase/><testcase/>", 2),
    ],
)
def test_executed_test_guard(tmp_path: Path, cases: str, expected: int) -> None:
    report = tmp_path / "results.xml"
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>", encoding="utf-8")
    if expected:
        assert check_execution(report) == expected
    else:
        with pytest.raises(ValueError, match="no tests executed"):
            check_execution(report)


def test_missing_and_invalid_reports_fail_closed(tmp_path: Path) -> None:
    report = tmp_path / "missing.xml"
    with pytest.raises(FileNotFoundError):
        check_execution(report)
    report.write_text("not XML", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        check_execution(report)
