#!/usr/bin/env python3
"""
Validator for MLE-Bench Python solution files.

Now assumes:
- This script lives in a `validator/` folder.
- All baseline .py files are in a sibling `baselines/` folder.

Checks:
1) File contains CO_DATASCIENTIST_BLOCK markers.
2) File contains KPI print statement.
3) Every pd.read_csv(...) must point to an existing file:
   - If it uses DATA_DIR → rewrite path to `.cache/mle-bench/data/<script-name>/prepared/public/<filename>`
   - If it uses quoted string → check file exists (relative to cwd if needed)
   - If it uses os.path.join with quoted parts → join them and check
   - Otherwise skip
4) Must save submission to SUBMISSION_DIR/submission.csv
5) Matches expected filenames from split file
"""

import re
import sys
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# Directory setup
# ────────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BASELINES_DIR = SCRIPT_DIR.parent / "baselines"
SELF_FILE = Path(__file__).name
SPLIT_FILE = Path("/home/harry/mle-bench/experiments/splits/low.txt")


def _base_data_dir_for(script_name: str) -> Path:
    comp = script_name.rsplit(".", 1)[0]
    return Path(f"/home/harry/.cache/mle-bench/data/{comp}/prepared/public")


def _extract_quoted_parts(s: str) -> list[str]:
    """Find quoted strings inside an expression."""
    import re
    return re.findall(r'["\']([^"\']+)["\']', s)


def _extract_read_csv_args(text: str) -> list[str]:
    """Find all argument strings passed into pd.read_csv(...), handling nested () properly."""
    results = []
    pattern = re.compile(r'pd\s*\.\s*read_csv\s*\(')
    for m in pattern.finditer(text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i-1])
    return results


def resolve_read_csv_path(arg: str, script_name: str) -> Path | None:
    """Resolve a filesystem path from pd.read_csv argument."""
    base = _base_data_dir_for(script_name)
    arg_stripped = arg.strip()

    # Case 1: Uses DATA_DIR → place inside .cache/mle-bench path
    if "DATA_DIR" in arg_stripped:
        parts = _extract_quoted_parts(arg_stripped)
        parts = [p for p in parts if p != "DATA_DIR"]
        if parts:
            resolved_path = base.joinpath(*parts)
            if resolved_path.exists() or resolved_path.with_suffix(".zip").exists():
                return resolved_path
        return base

    # Case 2: os.path.join(...) with quoted parts
    if "os.path.join" in arg_stripped:
        parts = _extract_quoted_parts(arg_stripped)
        if parts:
            resolved_path = Path(*parts).resolve()
            if resolved_path.exists() or resolved_path.with_suffix(".zip").exists():
                return resolved_path

    # Case 3: plain quoted path
    parts = _extract_quoted_parts(arg_stripped)
    if parts:
        p = Path(parts[0])
        if not p.is_absolute():
            p = (BASELINES_DIR / p).resolve()
        if p.exists() or p.with_suffix(".zip").exists():
            return p

    return None


def main() -> int:
    issues: list[str] = []
    passes: list[str] = []

    # Load expected files
    if SPLIT_FILE.exists():
        expected_ids = [ln.strip() for ln in SPLIT_FILE.read_text().splitlines() if ln.strip()]
        expected_files = {f"{_id}.py" for _id in expected_ids}
    else:
        issues.append(f"❌ Missing split file: {SPLIT_FILE}")
        expected_files = set()

    py_files = {p.name for p in BASELINES_DIR.glob("*.py")}

    missing = expected_files - py_files
    extra = py_files - expected_files

    if missing:
        issues.append("❌ Missing expected files:\n   " + "\n   ".join(sorted(missing)))
    if extra:
        issues.append("⚠️ Unexpected extra files:\n   " + "\n   ".join(sorted(extra)))

    # Check each Python file
    for fname in sorted(py_files & expected_files):
        fpath = BASELINES_DIR / fname
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            issues.append(f"--- {fname} ---\n  - Could not read file: {e}")
            continue

        file_issues: list[str] = []

        # 1) CO_DATASCIENTIST_BLOCK markers
        if "CO_DATASCIENTIST_BLOCK_START" not in content or "CO_DATASCIENTIST_BLOCK_END" not in content:
            file_issues.append("Missing CO_DATASCIENTIST_BLOCK markers")

        # 2) KPI print
        if not re.search(r'print\s*\(\s*f?[\'"]KPI:', content):
            file_issues.append("Missing KPI print statement")

        # 3) read_csv paths
        for raw_arg in _extract_read_csv_args(content):
            resolved = resolve_read_csv_path(raw_arg, fname)
            if resolved and not resolved.exists():
                file_issues.append(f"read_csv file not found: {resolved}")

        # 4) submission save
        if not ("SUBMISSION_DIR" in content and "submission.csv" in content):
            file_issues.append("Missing save to SUBMISSION_DIR/submission.csv")

        if file_issues:
            issues.append(f"--- {fname} ---\n  " + "\n  ".join(f"- {msg}" for msg in file_issues))
        else:
            passes.append(f"✅ {fname} passed all checks")

    # Count summaries
    total_files = len(py_files)
    total_expected = len(expected_files)
    total_passed = len(passes)
    total_failed = len(issues)

    print("Validation Report\n=================")
    print(f"Total files checked: {total_files}")
    print(f"Expected files: {total_expected}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")

    if passes:
        for ok in passes:
            print(ok)
    if issues:
        print("\nIssues:\n-------")
        for msg in issues:
            print(msg)
        print("\n❌ Some files failed validation")
        return 1

    print("\n🎉 All files passed validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
