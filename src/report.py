"""Render the engineering report. Every number comes from the model run; the template
only formats values. Sentences that depend on results are switched by config thresholds."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _img(path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def num(x, d=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    a = abs(x)
    if a != 0 and (a < 10 ** -d or a >= 1e6):
        return f"{x:.{d + 1}g}"
    return f"{x:,.{d}f}"


def pct(x, d=1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def build_context(cfg: dict, case: dict, sens: dict, figs: dict) -> dict:
    a, rep = case["absorber"], cfg["report"]
    regime_ok = a["ha_min"] > rep["min_hatta_for_fast_regime"] and \
        a["einf_over_ha_min"] > rep["min_einf_over_ha"]
    hs = sens["lean_naoh_molality"].dropna(subset=["packed_height_m"])
    best = hs.loc[hs["packed_height_m"].idxmin()]
    def trend(name):
        h = sens[name].dropna(subset=["packed_height_m"])["packed_height_m"].to_numpy()
        d = np.diff(h)
        return "down" if (d < 0).all() else "up" if (d > 0).all() else "mixed"

    return {
        "trend_T": trend("absorber_liquid_temperature_C"),
        "trend_cryst": trend("crystallizer_temperature_C"),
        "cfg": cfg, "title": rep["title"],
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "c": case, "a": a, "lp": case["loop"], "e": case["energy"], "p": case["products"],
        "el": case["elements"], "mb": case["mass_balance"], "st": case["streams"],
        "regime_ok": regime_ok, "sens": sens, "best_molality": best,
        "img": {k: _img(v) for k, v in figs.items()},
        "num": num, "pct": pct, "sci": lambda x: f"{x:.3g}",
    }


def render(context: dict, out_path) -> Path:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), undefined=StrictUndefined, autoescape=True)
    out_path = Path(out_path)
    out_path.write_text(env.get_template("report.html.j2").render(**context), encoding="utf-8")
    return out_path
