"""Keep maintained source and documentation within the repository's line budget."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 2_000
SOURCE_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".cjs",
    ".mjs",
    ".css",
    ".html",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".sh",
    ".ps1",
    ".sql",
}


def main() -> int:
    names = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT
        )
        .decode("utf-8")
        .split("\0")
    )
    failures = []
    checked = 0
    for name in sorted(set(names) - {""}):
        source = ROOT / name
        if source.suffix not in SOURCE_SUFFIXES or not source.is_file():
            continue
        if name.startswith(("node_modules/", "konbaung_reader_app/static/build/")):
            continue
        data = source.read_bytes()
        if b"\0" in data:
            continue
        checked += 1
        count = len(data.splitlines())
        if count > LIMIT:
            failures.append(f"{name}: {count} lines (limit {LIMIT})")
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"Checked {checked} source/documentation files; {len(failures)} over the line limit.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
