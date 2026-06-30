"""Write paper_reported_runs.csv: exactly the 66 runs cited in the paper."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "experiments/Results/summary.csv"
OUT = Path(__file__).resolve().parent / "paper_reported_runs.csv"

# Excluded from paper totals (documented in supplement/README.txt):
# - 058-060: A8 top7 pre-judge-cap reruns (superseded by 073-075)
# - 061-063: A8 top10 pre-fix judge overflow runs
# - 070-072: A9 chunk-512 pre-fix judge overflow runs
# - 076:     A8 top10 intermediate rescore (superseded by 077-079)
EXCLUDED_RUN_PREFIXES = (
    "058__",
    "059__",
    "060__",
    "061__",
    "062__",
    "063__",
    "070__",
    "071__",
    "072__",
    "076__",
)


def is_paper_reported(row: dict) -> bool:
    if not row["round"].startswith("ablation_"):
        return False
    return not row["run_folder"].startswith(EXCLUDED_RUN_PREFIXES)


def main() -> None:
    with SUMMARY.open(newline="") as f:
        rows = list(csv.DictReader(f))

    paper_rows = [r for r in rows if is_paper_reported(r)]
    if len(paper_rows) != 66:
        raise SystemExit(f"Expected 66 paper runs, got {len(paper_rows)}")

    fieldnames = ["paper_reported"] + list(rows[0].keys())
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in paper_rows:
            out = {"paper_reported": "true", **row}
            writer.writerow(out)

    print(f"Wrote {OUT} ({len(paper_rows)} rows)")


if __name__ == "__main__":
    main()
