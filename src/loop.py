"""Liquor loop: absorber -> cross exchanger -> cooling crystallizer -> filter -> back.

Chemistry in the absorber:  CO2 + 2 NaOH -> Na2CO3 + H2O
Per mol CO2 captured (C): 2 mol NaOH used, 1 mol Na2CO3 and 1 mol H2O formed.

Crystallizer: Na2CO3 leaves as the decahydrate Na2CO3.10H2O. The mother liquor is
saturated at the crystallizer temperature:
    (n_CO3,rich - x) * M_Na2CO3 = s * (W_rich - 10 * M_H2O * x)
    =>  x = (n_CO3,rich * M_Na2CO3 - s * W_rich) / (M_Na2CO3 - 10 * s * M_H2O)
    where x = crystals formed (mol/s), n_CO3,rich = carbonate entering (mol/s),
    W_rich = water entering (kg/s), s = solubility (kg Na2CO3 per kg water).
Wet cake carries mother liquor away (cake_moisture kg per kg crystals); the rest is recycled.
Makeup NaOH (50 wt%) and water restore the lean liquor to its specified NaOH molality.
At steady state the lean carbonate level is found by fixed-point iteration.
"""
from __future__ import annotations

import numpy as np

from src.properties import M_CO2, M_H2O, M_NA2CO3, M_NAOH

M_HYD = M_NA2CO3 + 10 * M_H2O          # Na2CO3.10H2O, kg/mol


def feed_gas(cfg: dict) -> dict:
    """Flue-gas molar flows (mol/s) from normal m3/h."""
    d = cfg["design_basis"]
    n_total = d["flue_gas_nm3_per_h"] / 0.0224140 / 3600.0    # 22.414 L/mol at 0 degC, 1 atm
    flows = {k: v * n_total for k, v in d["composition_mol_frac"].items()}
    return {"total": n_total, "flows": flows}


def solubility(T_C: float, cfg_c: dict) -> float:
    """kg Na2CO3 per kg water, interpolated from the table in config."""
    t, g = cfg_c["solubility_table_C"], cfg_c["solubility_table_g_per_100g"]
    if not t[0] <= T_C <= t[-1]:
        raise ValueError(f"crystallizer temperature {T_C} degC is outside the solubility table")
    return float(np.interp(T_C, t, g)) / 100.0


def crystals_formed(n_co3: float, water: float, s: float) -> float:
    """mol/s of Na2CO3.10H2O that must crystallize to leave a saturated liquor."""
    return (n_co3 * M_NA2CO3 - s * water) / (M_NA2CO3 - 10 * s * M_H2O)


def solve_loop(cfg: dict, tol: float = 1e-12, max_iter: int = 5000) -> dict:
    d, sv, cr = cfg["design_basis"], cfg["solvent"], cfg["crystallizer"]
    gas = feed_gas(cfg)
    C = d["capture_fraction"] * gas["flows"]["CO2"]          # mol/s CO2 captured
    n_oh_lean = 2 * C / sv["per_pass_naoh_conversion"]       # mol/s NaOH into absorber
    w_lean = n_oh_lean / sv["lean_naoh_molality"]            # kg/s water in lean liquor
    s = solubility(cr["temperature_C"], cr)

    n_co3_lean = 0.0
    for it in range(max_iter):
        n_oh_rich, n_co3_rich, w_rich = n_oh_lean - 2 * C, n_co3_lean + C, w_lean + M_H2O * C
        # until the recycled carbonate builds up to saturation, nothing crystallizes
        x = max(0.0, crystals_formed(n_co3_rich, w_rich, s))
        ml = {"OH": n_oh_rich, "CO3": n_co3_rich - x, "W": w_rich - 10 * M_H2O * x}
        ml_mass = ml["W"] + ml["OH"] * M_NAOH + ml["CO3"] * M_NA2CO3
        cake_liquor = cr["cake_moisture"] * x * M_HYD
        phi = cake_liquor / ml_mass                          # share of mother liquor lost with cake
        new = (1 - phi) * ml["CO3"]
        if abs(new - n_co3_lean) < tol:
            n_co3_lean = new
            break
        n_co3_lean = new
    else:
        raise RuntimeError("loop did not converge")
    if x <= 0:
        raise ValueError("liquor never reaches saturation: no crystals form at steady state")

    recycle = {k: (1 - phi) * v for k, v in ml.items()}
    mk_naoh = n_oh_lean - recycle["OH"]                      # mol/s NaOH makeup
    mk_soln_water = mk_naoh * M_NAOH * (1 - sv["makeup_naoh_wt_frac"]) / sv["makeup_naoh_wt_frac"]
    mk_water = w_lean - recycle["W"] - mk_soln_water         # kg/s extra water (negative = excess)
    cake = {k: phi * v for k, v in ml.items()}

    return {
        "iterations": it + 1, "C": C, "gas": gas, "solubility": s, "phi": phi,
        "lean": {"OH": n_oh_lean, "CO3": n_co3_lean, "W": w_lean},
        "rich": {"OH": n_oh_rich, "CO3": n_co3_rich, "W": w_rich},
        "crystals": x, "mother_liquor": ml, "recycle": recycle, "cake_liquor": cake,
        "makeup_naoh": mk_naoh, "makeup_solution_water": mk_soln_water, "makeup_water": mk_water,
    }


def liquor_mass(stream: dict) -> float:
    return stream["W"] + stream["OH"] * M_NAOH + stream["CO3"] * M_NA2CO3


def energy_balance(loop: dict, cfg: dict) -> dict:
    """Heat duties in W. Absorber liquid is assumed held at its design temperature."""
    sv, cr, th = cfg["solvent"], cfg["crystallizer"], cfg["thermo"]
    cp = cr["liquor_cp"]
    T_abs, T_c, dT_app = sv["absorber_liquid_temperature_C"], cr["temperature_C"], cr["exchanger_approach_K"]
    m_lean, m_rich = liquor_mass(loop["lean"]), liquor_mass(loop["rich"])

    q_rxn = -th["heat_of_reaction"] * loop["C"]                 # heat released in the absorber
    # Counter-current cross exchanger: cold lean (from crystallizer) heated to T_abs - approach
    t_lean_hot = T_abs - dT_app
    q_hx = m_lean * cp * (t_lean_hot - T_c)
    t_rich_out = T_abs - q_hx / (m_rich * cp)
    if t_rich_out - T_c < dT_app:
        raise ValueError("cross exchanger violates the approach temperature at its cold end")
    # Chiller: rich liquor from t_rich_out down to T_c, plus heat released by crystallization
    q_chill = m_rich * cp * (t_rich_out - T_c) + cr["heat_of_crystallization"] * loop["crystals"]
    # Absorber: lean enters at t_lean_hot; reaction heat would raise it by:
    rise = q_rxn / (m_lean * cp)
    q_abs_cooler = max(0.0, q_rxn - m_lean * cp * (T_abs - t_lean_hot))   # to keep outlet at T_abs
    return {"q_reaction": q_rxn, "q_cross_exchanger": q_hx, "t_lean_to_absorber": t_lean_hot,
            "t_rich_after_exchanger": t_rich_out, "q_chiller": q_chill,
            "adiabatic_rise": rise, "q_absorber_cooler": q_abs_cooler,
            "rise_flag": rise > th["max_adiabatic_rise_K"]}


def element_balance(loop: dict) -> dict:
    """Na and C in vs out (mol/s). Na in = makeup NaOH. C in = CO2 captured.
    Out = crystals (2 Na, 1 C each) + carbonate and NaOH lost in the cake liquor."""
    cake = loop["cake_liquor"]
    na_out = 2 * loop["crystals"] + 2 * cake["CO3"] + cake["OH"]
    c_out = loop["crystals"] + cake["CO3"]
    return {"na_in": loop["makeup_naoh"], "na_out": na_out, "c_in": loop["C"], "c_out": c_out}


def products(loop: dict) -> dict:
    """Headline material numbers in t/h and per tonne CO2."""
    h = 3.6                                                  # kg/s -> t/h
    co2 = loop["C"] * M_CO2 * h
    naoh = loop["makeup_naoh"] * M_NAOH * h
    hyd = loop["crystals"] * M_HYD * h
    return {"co2_captured_t_h": co2, "naoh_t_h": naoh, "naoh_per_t_co2": naoh / co2,
            "naoh_min_per_t_co2": 2 * M_NAOH / M_CO2,                # stoichiometric minimum
            "decahydrate_t_h": hyd, "soda_ash_equiv_t_h": loop["crystals"] * M_NA2CO3 * h,
            "wet_cake_t_h": hyd + (liquor_mass(loop["cake_liquor"])) * h,
            "makeup_water_t_h": loop["makeup_water"] * h}
