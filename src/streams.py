"""Stream table (kg/h per component) and the process flow diagram.

Stream numbers match the PFD:
  1 flue gas in            2 treated gas out        3 lean liquor to absorber
  4 rich liquor out        5 rich liquor to chiller 6 crystallizer slurry
  7 wet crystal cake       8 mother liquor recycle  9 makeup caustic (50 wt%)
  10 makeup water
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.absorber import MOLAR_MASS  # noqa: E402
from src.loop import M_HYD, liquor_mass  # noqa: E402
from src.properties import M_H2O, M_NA2CO3, M_NAOH  # noqa: E402

COMPONENTS = ["CO2", "N2", "O2", "H2O", "NaOH", "Na2CO3", "Na2CO3.10H2O"]
H = 3600.0   # s/h


def _liquor(s: dict) -> dict:
    return {"H2O": s["W"] * H, "NaOH": s["OH"] * M_NAOH * H, "Na2CO3": s["CO3"] * M_NA2CO3 * H}


def stream_table(loop: dict, energy: dict, cfg: dict) -> pd.DataFrame:
    f = loop["gas"]["flows"]
    gas_in = {k: f[k] * MOLAR_MASS[k] * H for k in f}
    gas_out = dict(gas_in, CO2=(f["CO2"] - loop["C"]) * MOLAR_MASS["CO2"] * H)
    T_g = cfg["design_basis"]["gas_temperature_C"]
    T_abs = cfg["solvent"]["absorber_liquid_temperature_C"]
    T_c = cfg["crystallizer"]["temperature_C"]
    crystals = {"Na2CO3.10H2O": loop["crystals"] * M_HYD * H}
    naoh_kg_h = loop["makeup_naoh"] * M_NAOH * H
    rows = [
        ("1", "Flue gas in", T_g, gas_in),
        ("2", "Treated gas out", T_g, gas_out),
        ("3", "Lean liquor to absorber", energy["t_lean_to_absorber"], _liquor(loop["lean"])),
        ("4", "Rich liquor from absorber", T_abs, _liquor(loop["rich"])),
        ("5", "Rich liquor to chiller", energy["t_rich_after_exchanger"], _liquor(loop["rich"])),
        ("6", "Crystallizer slurry", T_c, {**_liquor(loop["mother_liquor"]), **crystals}),
        ("7", "Wet crystal cake", T_c, {**_liquor(loop["cake_liquor"]), **crystals}),
        ("8", "Mother liquor recycle", T_c, _liquor(loop["recycle"])),
        ("9", "Makeup caustic (50 wt%)", 25.0, {"NaOH": naoh_kg_h,
                                                "H2O": loop["makeup_solution_water"] * H}),
        ("10", "Makeup water", 25.0, {"H2O": loop["makeup_water"] * H}),
    ]
    out = []
    for no, name, T, comp in rows:
        row = {"stream": no, "description": name, "T_C": T}
        row.update({c: comp.get(c, 0.0) for c in COMPONENTS})
        row["total_kg_h"] = sum(row[c] for c in COMPONENTS)
        out.append(row)
    return pd.DataFrame(out)


def overall_mass_balance(st: pd.DataFrame) -> dict:
    """Plant boundary: in = 1 + 9 + 10, out = 2 + 7. Reaction conserves mass, so these match."""
    tot = st.set_index("stream")["total_kg_h"]
    return {"in": float(tot["1"] + tot["9"] + tot["10"]), "out": float(tot["2"] + tot["7"])}


# ---------------------------------------------------------------- PFD drawing

INK, MUTED, GAS, LIQ, BOX = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#f3f2ef"


def _box(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=BOX, ec=INK, lw=1.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color=INK)


def _arrow(ax, pts, color, label=None, label_at=None):
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        last = (x1, y1) == pts[-1]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if last else "-",
                                     mutation_scale=12, color=color, lw=1.6))
    if label:
        lx, ly = label_at
        ax.text(lx, ly, label, fontsize=8, ha="center", va="center", color=INK,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=color, lw=1.2))


def draw_pfd(path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
    _box(ax, 1.4, 0.8, 1.2, 3.6, "Packed\nabsorber\n(NaOH)")
    _box(ax, 4.0, 2.6, 1.5, 0.9, "Cross\nexchanger")
    _box(ax, 6.4, 2.6, 1.3, 0.9, "Chiller")
    _box(ax, 8.5, 2.4, 1.6, 1.3, "Cooling\ncrystallizer")
    _box(ax, 8.5, 0.5, 1.6, 0.9, "Filter")
    _box(ax, 4.0, 0.5, 1.5, 0.9, "Makeup\nmixer")
    # gas
    _arrow(ax, [(0.2, 1.2), (1.4, 1.2)], GAS, "1", (0.6, 1.2))
    _arrow(ax, [(2.0, 4.4), (2.0, 5.0), (0.2, 5.0)], GAS, "2", (0.6, 5.0))
    # rich liquor: absorber bottom -> exchanger -> chiller -> crystallizer
    _arrow(ax, [(2.6, 1.0), (3.4, 1.0), (3.4, 2.85), (4.0, 2.85)], LIQ, "4", (3.4, 2.0))
    _arrow(ax, [(5.5, 2.85), (6.4, 2.85)], LIQ, "5", (5.95, 2.85))
    _arrow(ax, [(7.7, 3.05), (8.5, 3.05)], LIQ)
    _arrow(ax, [(9.3, 2.4), (9.3, 1.4)], LIQ, "6", (9.3, 1.9))
    _arrow(ax, [(10.1, 0.95), (10.9, 0.95)], INK, "7", (10.45, 0.95))
    ax.text(10.9, 0.55, "Na$_2$CO$_3$·10H$_2$O cake", fontsize=8, ha="right", color=INK)
    # mother liquor back to mixer, then lean through exchanger to absorber top
    _arrow(ax, [(8.5, 0.95), (5.5, 0.95)], LIQ, "8", (7.0, 0.95))
    _arrow(ax, [(4.75, 1.4), (4.75, 2.6)], LIQ)
    _arrow(ax, [(4.75, 3.5), (4.75, 4.2), (2.6, 4.2)], LIQ, "3", (3.6, 4.2))
    _arrow(ax, [(4.4, 0.0), (4.4, 0.5)], LIQ, "9", (4.4, 0.2))
    _arrow(ax, [(5.1, 0.0), (5.1, 0.5)], LIQ, "10", (5.1, 0.2))
    ax.text(4.2, 0.12, "NaOH 50%", fontsize=7, ha="right", color=MUTED)
    ax.text(5.3, 0.12, "water", fontsize=7, ha="left", color=MUTED)
    ax.text(1.0, 2.6, "pump-around\ncooler", fontsize=7, ha="right", color=MUTED)
    ax.text(10.9, 5.0, "blue = gas, orange = liquor", fontsize=8, ha="right", color=MUTED)
    ax.set_title("Process flow diagram (numbers = streams in the stream table)", loc="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
