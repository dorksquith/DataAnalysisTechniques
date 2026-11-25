#!/usr/bin/env python3
"""
collusion_check.py — template-adjusted diff similarity between two submissions.

Usage:
    python3 collusion_check.py <cand_A> <cand_B>

Example for checking candidate number 123456 versus 654321
    python3 collusion_check.py 123456 654321

The script searches the directory it lives in for files whose names contain
the candidate number, strips lines shared with the template (Coursework2/Report2.md),
runs a line-level diff on the residuals, and reports the similarity ratio.
"""

import sys
import os
import difflib
import glob

# ── locate this script's directory as the submission folder ──────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "Coursework2", "Report2.md")


def find_submission(cand_no: str) -> str:
    """Return the path of the submission file for the given candidate number."""
    pattern = os.path.join(SCRIPT_DIR, f"candidateno{cand_no}_*.md")
    matches = glob.glob(pattern)
    if not matches:
        sys.exit(f"Error: no file found matching candidateno{cand_no}_*.md "
                 f"in {SCRIPT_DIR}")
    if len(matches) > 1:
        print(f"Warning: multiple files found for {cand_no}, using first:\n"
              f"  {matches[0]}", file=sys.stderr)
    return matches[0]


def read(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


def strip_template(lines: list[str], template_lines: set[str]) -> list[str]:
    """Remove any line that appears verbatim in the template."""
    return [l for l in lines if l not in template_lines]


def similarity_ratio(lines_a: list[str], lines_b: list[str]) -> float:
    """
    2 * matched_lines / (len_a + len_b)  — identical to difflib ratio().
    Returns 0.0 if both lists are empty.
    """
    total = len(lines_a) + len(lines_b)
    if total == 0:
        return 0.0
    return difflib.SequenceMatcher(None, lines_a, lines_b, autojunk=False).ratio()


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 collusion_check.py <cand_A> <cand_B>")

    cand_a, cand_b = sys.argv[1], sys.argv[2]

    path_a = find_submission(cand_a)
    path_b = find_submission(cand_b)

    if not os.path.exists(TEMPLATE_PATH):
        sys.exit(f"Error: template not found at {TEMPLATE_PATH}")

    template_lines = set(read(TEMPLATE_PATH))
    lines_a_raw    = read(path_a)
    lines_b_raw    = read(path_b)
    lines_a        = strip_template(lines_a_raw, template_lines)
    lines_b        = strip_template(lines_b_raw, template_lines)

    raw_ratio  = similarity_ratio(lines_a_raw, lines_b_raw)
    adj_ratio  = similarity_ratio(lines_a, lines_b)

    # ── diff output ──────────────────────────────────────────────────────────
    diff = list(difflib.unified_diff(
        lines_a, lines_b,
        fromfile=f"{cand_a} (stripped)",
        tofile=f"{cand_b} (stripped)",
        lineterm="",
    ))

    # ── report ───────────────────────────────────────────────────────────────
    print(f"\nSubmission A : {os.path.basename(path_a)}")
    print(f"Submission B : {os.path.basename(path_b)}")
    print(f"Template     : {os.path.relpath(TEMPLATE_PATH, SCRIPT_DIR)}")
    print()
    print(f"Lines in A (raw / after template strip) : "
          f"{len(lines_a_raw):4d} / {len(lines_a):4d}")
    print(f"Lines in B (raw / after template strip) : "
          f"{len(lines_b_raw):4d} / {len(lines_b):4d}")
    print()
    print(f"Raw similarity (before strip)  : {raw_ratio:.4f}  "
          f"({raw_ratio*100:.1f}%)")
    print(f"Adjusted similarity (post-strip): {adj_ratio:.4f}  "
          f"({adj_ratio*100:.1f}%)")
    print()
    print("Equivalent shell commands:")
    print(f"  grep -vxFf Coursework2/Report2.md "
          f"'{os.path.basename(path_a)}' > stripped_a.txt")
    print(f"  grep -vxFf Coursework2/Report2.md "
          f"'{os.path.basename(path_b)}' > stripped_b.txt")
    print(f"  diff --unchanged-line-format='' --new-line-format='+%L' "
          f"--old-line-format='-%L' stripped_a.txt stripped_b.txt")
    print()

    if diff:
        print(f"Unified diff ({len(diff)} lines)  "
              f"[only lines absent from template are shown]:")
        print("-" * 60)
        # cap at 200 diff lines to avoid overwhelming output
        for line in diff[:200]:
            print(line)
        if len(diff) > 200:
            print(f"  ... {len(diff) - 200} further diff lines omitted ...")
    else:
        print("Diff: files are identical after template stripping.")


if __name__ == "__main__":
    main()
