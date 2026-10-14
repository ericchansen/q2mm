"""Reject a test responsibility whose JUnit report contains no executed cases."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def check_execution(report: Path) -> int:
    """Return the executed case count, rejecting empty or wholly skipped reports."""
    executed = sum(case.find("skipped") is None for case in ET.parse(report).getroot().iter("testcase"))
    if not executed:
        raise ValueError(f"{report}: no tests executed; check selectors, tier flags and backend configuration.")
    return executed


def main() -> None:
    """Check an existing JUnit report; missing or invalid reports fail closed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(f"Executed {check_execution(args.report)} test cases.")


if __name__ == "__main__":
    main()
