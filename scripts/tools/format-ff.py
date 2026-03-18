#!/usr/bin/env python3
"""
I use this to clean up FFs I find in the literature or wherever with poor
formatting.
"""

import argparse
import re
import sys


def format_ff(filename):
    with open(filename) as f:
        lines = f.readlines()
    formatted_lines = []
    for line in lines:
        if line.startswith(" C") or line.startswith(" 9") or line.startswith("-"):
            print(line.strip("\n"))
        elif re.match("[a-z][A-Z]", line[:2]):
            # I don't think this section is general enough.
            cols = line.split()
            lbl = f"{cols[0]:>2}"
            a1 = f"{cols[1]:>2}"
            a2 = f"{cols[2]:>2}"
            a3 = f"{cols[3]:>2}"
            p1 = f"{float(cols[4]):>9.4f}"
            p2 = f"{float(cols[5]):>9.4f}"
            print(f"{lbl}  {a1}  {a2}  {a3}          {p1} {p2}")
        elif re.match(r"[a-z\s]1", line[:2]):
            cols = line.split()
            lbl = f"{cols[0]:>2}"
            a1 = f"{cols[1]:>2}"
            a2 = f"{cols[2]:>2}"
            p1 = f"{float(cols[3]):>9.4f}"
            p2 = f"{float(cols[4]):>9.4f}"
            p3 = f"{float(cols[5]):>9.4f}"
            print(f"{lbl}  {a1}  {a2}              {p1} {p2} {p3}")
        elif re.match(r"[a-z\s]2", line[:2]):
            cols = line.split()
            lbl = f"{cols[0]:>2}"
            a1 = f"{cols[1]:>2}"
            a2 = f"{cols[2]:>2}"
            a3 = f"{cols[3]:>2}"
            p1 = f"{float(cols[4]):>9.4f}"
            p2 = f"{float(cols[5]):>9.4f}"
            print(f"{lbl}  {a1}  {a2}  {a3}          {p1} {p2}")
        elif re.match(r"[a-z\s]4", line[:2]):
            cols = line.split()
            lbl = f"{cols[0]:>2}"
            a1 = f"{cols[1]:>2}"
            a2 = f"{cols[2]:>2}"
            a3 = f"{cols[3]:>2}"
            a4 = f"{cols[4]:>2}"
            p1 = f"{float(cols[5]):>9.4f}"
            p2 = f"{float(cols[6]):>9.4f}"
            p3 = f"{float(cols[7]):>9.4f}"
            print(f"{lbl}  {a1}  {a2}  {a3}  {a4}      {p1} {p2} {p3}")


def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="File to parse and format.")
    return parser


def main(opts):
    format_ff(opts.input)


if __name__ == "__main__":
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
