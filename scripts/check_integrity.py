"""Check the source ZIP payload without importing third-party libraries."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in entries:
        expected, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Unsafe manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed release file: {name}")
    print(f"Verified {len(entries)} release files.")


if __name__ == "__main__":
    main()
