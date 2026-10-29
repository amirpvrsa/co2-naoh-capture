"""Run one design case and the sensitivity study; draw the result charts."""
from __future__ import annotations

import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.absorber import size_absorber  # noqa: E402
from src.loop import element_balance, energy_balance, products, solve_loop  # noqa: E402
from src.streams import overall_mass_balance, stream_table  # noqa: E402

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

# sensitivity parameter -> where it lives in config.yaml, and a readable label
PARAMS = {
    "capture_fraction": (("design_basis", "capture_fraction"), "CO$_2$ capture fraction"),
    "lean_naoh_molality": (("solvent", "lean_naoh_molality"), "Lean NaOH molality (mol/kg water)"),
    "absorber_liquid_temperature_C": (("solvent", "absorber_liquid_temperature_C"), "Absorber liquid temperature (degC)"),
    "per_pass_naoh_conversion": (("solvent", "per_pass_naoh_conversion"), "NaOH conversion per pass"),
    "crystallizer_temperature_C": (("crystallizer", "temperature_C"), "Crystallizer temperature (degC)"),
}


def run_case(cfg: dict) -> dict:
    loop = solve_loop(cfg)
    energy = energy_balance(loop, cfg)
    col = size_absorber(loop, cfg)
    st = stream_table(loop, energy, cfg)
    return {"loop": loop, "energy": energy, "absorber": col, "products": products(loop),
            "elements": element_balance(loop), "streams": st, "mass_balance": overall_mass_balance(st)}


def _summary(case: dict) -> dict:
    a, e, p = case["absorber"], case["energy"], case["products"]
    return {"packed_height_m": a["height"], "diameter_m": a["diameter"],
            "circulation_m3_h": a["q_liq_m3h"], "naoh_per_t_co2": p["naoh_per_t_co2"],
            "chiller_MW": e["q_chiller"] / 1e6, "absorber_cooler_MW": e["q_absorber_cooler"] / 1e6,
            "crystals_t_h": p["decahydrate_t_h"], "ha_min": a["ha_min"],
            "einf_over_ha_min": a["einf_over_ha_min"], "wetting_ok": a["wetting_ok"]}


def sensitivity(cfg: dict) -> dict:
    out = {}
    for name, ((sec, key), _label) in PARAMS.items():
        rows = []
        for v in cfg["sensitivity"][name]:
            c = copy.deepcopy(cfg)
            c[sec][key] = v
            try:
                rows.append({"value": v, **_summary(run_case(c)), "note": ""})
            except (ValueError, RuntimeError) as err:
                rows.append({"value": v, "note": str(err)})
        out[name] = pd.DataFrame(rows)
    return out


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_sensitivity(sens: dict, cfg: dict, fig_dir) -> dict:
    paths = {}
    for name, df in sens.items():
        (sec, key), label = PARAMS[name]
        base = cfg[sec][key]
        metric, ylabel = "packed_height_m", "Packed height (m)"
        ok = df[df["note"] == ""] if "note" in df else df
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.plot(ok["value"], ok[metric], color=BLUE, lw=2, marker="o", ms=5)
        b = ok[np.isclose(ok["value"].astype(float), base)]
        if len(b):
            ax.scatter(b["value"], b[metric], s=110, facecolors="none", edgecolors=INK, lw=1.3,
                       zorder=4, label="Base case")
            ax.legend(fontsize=7, frameon=False)
        _style(ax, f"{ylabel.split(' (')[0]} vs. {label.split(' (')[0]}", label, ylabel)
        fig.tight_layout()
        paths[name] = fig_dir / f"sens_{name}.png"
        fig.savefig(paths[name], dpi=130)
        plt.close(fig)
    return paths


def plot_profiles(case: dict, path) -> None:
    p = case["absorber"]["profile"]
    z = p["z_from_top"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    a1.plot(p["y"] * 100, z, color=BLUE, lw=2)
    _style(a1, "CO2 in the gas", "CO2 mole fraction (%)", "Depth below top of packing (m)")
    a2.plot(p["c_oh"], z, color=BLUE, lw=2)
    _style(a2, "Hydroxide in the liquor", "OH$^-$ concentration (mol/L)", "")
    a1.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
