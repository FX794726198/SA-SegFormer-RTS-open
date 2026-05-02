import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Collect region holdout summary.csv files")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--run-names", required=True, help="Comma-separated run names")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    run_names = [item.strip() for item in args.run_names.split(",") if item.strip()]

    rows = []
    missing = []
    for run_name in run_names:
        summary_path = output_root / run_name / "metrics" / "summary.csv"
        if not summary_path.exists():
            missing.append({"run_name": run_name, "status": "missing"})
            continue
        df = pd.read_csv(summary_path)
        if df.empty:
            missing.append({"run_name": run_name, "status": "empty"})
            continue
        row = df.iloc[0].to_dict()
        row["status"] = "ok"
        rows.append(row)

    combined = pd.DataFrame(rows)
    combined.to_csv(output_csv, index=False)

    if missing:
        missing_csv = output_csv.with_name(output_csv.stem + "_missing.csv")
        pd.DataFrame(missing).to_csv(missing_csv, index=False)
        print(f"Missing summaries written to {missing_csv}")

    print(f"Combined summaries written to {output_csv}")


if __name__ == "__main__":
    main()
