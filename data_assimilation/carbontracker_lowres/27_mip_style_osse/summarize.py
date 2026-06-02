"""Aggregate P3 MIP-style OSSE results into a comparison table.

Reads every ``results/<method>_<tag>/scores/`` and prints a markdown table of
RMSE / CRPS / spread-error ratio, plus the EnKF-vs-free gain at selected leads.
"""

import json
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
RES = EXP_DIR / "results"

LEADS = [4, 20, 40, 80, 119]


def load(run_dir: Path):
    s = pd.read_csv(run_dir / "scores" / "metrics_summary.csv", index_col=0)["value"]
    pl = pd.read_csv(run_dir / "scores" / "metrics_per_lead.csv")
    info = json.loads((run_dir / "method_info.json").read_text())
    return s, pl, info


def main():
    runs = sorted([d for d in RES.iterdir() if d.is_dir() and (d / "scores").exists()])
    if not runs:
        print("No results yet.")
        return

    rows = []
    per_lead = {}
    for d in runs:
        s, pl, info = load(d)
        rows.append({
            "run": d.name,
            "method": info.get("method"),
            "ak": info.get("ak_mode"),
            "thin": info.get("thin_fraction"),
            "obs_noise": info.get("obs_noise"),
            "RMSE": round(float(s["rmse_mean"]), 3),
            "CRPS": round(float(s["crps"]), 3),
            "spread/err": round(float(s["spread_error_ratio"]), 2),
            "n_inits": info.get("n_inits"),
            "n_samples": info.get("n_samples"),
            "n_steps": info.get("n_steps"),
        })
        per_lead[d.name] = pl.set_index("lead")["rmse_mean"]

    df = pd.DataFrame(rows).sort_values("run")
    print("\n## P3 MIP-style OSSE — summary\n")
    print(df.to_markdown(index=False))

    # EnKF-vs-free gain per lead (use the canonical full runs if present).
    free = next((k for k in per_lead if k.startswith("none_full")), None)
    enkf = next((k for k in per_lead if k == "enkf_full"), None)
    if free and enkf:
        print(f"\n## EnKF gain over free ({enkf} vs {free})\n")
        gl = []
        for lead in LEADS:
            if lead in per_lead[free].index and lead in per_lead[enkf].index:
                fr = per_lead[free].loc[lead]
                en = per_lead[enkf].loc[lead]
                gl.append({"lead": lead, "free": round(fr, 3), "enkf": round(en, 3),
                           "gain_%": round(100 * (fr - en) / fr, 1)})
        print(pd.DataFrame(gl).to_markdown(index=False))


if __name__ == "__main__":
    main()
