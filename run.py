"""Build everything with one command:  python run.py

Base case -> stream table and balances -> sensitivity study -> charts -> HTML report.
"""
from __future__ import annotations

import argparse
import json

from src.config import load_config, resolve
from src.model import plot_profiles, plot_sensitivity, run_case, sensitivity
from src.report import build_context, render
from src.streams import draw_pfd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    out = resolve(cfg["outputs"]["dir"])
    figs_dir = out / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    case = run_case(cfg)
    a, p = case["absorber"], case["products"]
    print(f"[absorber] D = {a['diameter']:.2f} m, packed height = {a['height']:.1f} m "
          f"(Ha min {a['ha_min']:.1f})")
    print(f"[loop] CO2 {p['co2_captured_t_h']:.3f} t/h, NaOH {p['naoh_per_t_co2']:.3f} t/t CO2, "
          f"crystals {p['decahydrate_t_h']:.2f} t/h")

    sens = sensitivity(cfg)
    figs = {"pfd": figs_dir / "pfd.png", "profiles": figs_dir / "profiles.png"}
    draw_pfd(figs["pfd"])
    plot_profiles(case, figs["profiles"])
    figs.update({f"sens_{k}": v for k, v in plot_sensitivity(sens, cfg, figs_dir).items()})

    case["streams"].to_csv(out / "stream_table.csv", index=False)
    for name, df in sens.items():
        df.to_csv(out / f"sensitivity_{name}.csv", index=False)
    summary = {"absorber": {k: v for k, v in a.items() if k != "profile"},
               "products": p, "energy": case["energy"], "elements": case["elements"],
               "mass_balance": case["mass_balance"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    path = render(build_context(cfg, case, sens, figs), out / cfg["report"]["file"])
    print(f"[report] {path}")


if __name__ == "__main__":
    main()
