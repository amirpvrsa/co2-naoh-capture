"""Rate-based design of the packed absorber.

Diameter from a design gas F-factor:
    u_G = F / sqrt(rho_G),   A = Q_G / u_G,   D = sqrt(4 A / pi)
    where F = gas capacity factor (Pa^0.5), rho_G = gas density, Q_G = gas volumetric flow.

Height from a CO2 balance on a slice dz of packing (gas flows up, liquid down):
    -dN = N_A * a_w * A * dz            =>   Z = integral from N_out to N_in of dN / (N_A a_w A)
    N_A = K_G * p_CO2                   (CO2 flux; equilibrium back-pressure ~0 while OH- is in excess)
    1 / K_G = 1 / k_G + H / (E k_L)     (gas-film resistance + liquid-film resistance)
    E = Ha / tanh(Ha),  Ha = sqrt(D_CO2 k2 c_OH) / k_L     (pseudo-first-order reaction)

    N   = CO2 molar flow in the gas at height z (mol/s);  p_CO2 = P * y = partial pressure (Pa)
    a_w = wetted packing area (m2/m3);  A = column cross-section (m2)
    k_G = gas-film coefficient (mol/(m2 s Pa));  k_L = liquid-film coefficient (m/s)
    H   = Henry's constant in the liquor (Pa m3/mol);  E = enhancement factor from the reaction
    Ha  = Hatta number;  k2 = rate constant (m3/(mol s));  c_OH = hydroxide concentration (mol/m3)

At each slice the liquid composition follows from a balance with the column top:
    n_OH(z) = n_OH,lean - 2 (N(z) - N_out);  n_CO3(z) = n_CO3,lean + (N(z) - N_out)
so hydroxide is used up toward the bottom, which slows the reaction there.

Validity checks (reported, not assumed): Ha > 3 (fast regime) and E_inf / Ha > 5, with
    E_inf = 1 + D_OH c_OH / (2 D_CO2 c_CO2,i)   (instantaneous-reaction limit)
    c_CO2,i = p_i / H,  p_i = p_CO2 - N_A / k_G   (interface conditions)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from src.loop import liquor_mass
from src.masstransfer import k_gas, k_liquid, wetted_area
from src.properties import (M_CO2, M_H2O, M_N2, M_O2, R, diffusivity_co2, henry_solution, ionic_strength, k2_oh,
                            liquor_concentrations, water_viscosity)

MOLAR_MASS = {"CO2": M_CO2, "H2O": M_H2O, "O2": M_O2, "N2": M_N2}


def size_absorber(loop: dict, cfg: dict) -> dict:
    d, sv, pk = cfg["design_basis"], cfg["solvent"], cfg["packing"]
    T_g = d["gas_temperature_C"] + 273.15
    T_l = sv["absorber_liquid_temperature_C"] + 273.15
    P = d["absorber_pressure_kPa"] * 1000.0

    flows = loop["gas"]["flows"]
    n_gas = loop["gas"]["total"]
    m_gas = sum(flows[k] * MOLAR_MASS[k] for k in flows)            # kg/s
    rho_g = P * (m_gas / n_gas) / (R * T_g)
    q_gas = n_gas * R * T_g / P                                     # m3/s
    u_g = pk["design_f_factor"] / np.sqrt(rho_g)
    area = q_gas / u_g
    diameter = np.sqrt(4 * area / np.pi)

    # liquid loading and wetting
    m_liq = liquor_mass(loop["lean"])                               # kg/s
    L = m_liq / area
    G = m_gas / area
    q_liq_m3h = m_liq / sv["liquid_density"] * 3600
    wetting = q_liq_m3h / area                                       # m3/(m2 h)
    wetting_min = pk["min_wetting_rate"] * pk["specific_area"]

    mu_l = water_viscosity(T_l) * sv["viscosity_ratio_to_water"]
    a_w = wetted_area(L, sv["liquid_density"], mu_l, sv["surface_tension"],
                      pk["specific_area"], pk["critical_surface_tension"])
    kg = k_gas(G, rho_g, cfg["gas_properties"]["viscosity"], cfg["gas_properties"]["co2_diffusivity"],
               pk["specific_area"], pk["nominal_size"], T_g, pk["onda_kg_constant"])

    # --- integrate along the column, from the top (N_out) to the bottom (N_in)
    n_in = flows["CO2"]
    n_out = n_in - loop["C"]
    n_inert = n_gas - n_in
    # log-spaced grid: points cluster near the top, where the CO2 is dilute and the
    # integrand 1/rate changes fastest (keeps trapezoid error far below 0.01%)
    N = np.geomspace(n_out, n_in, pk["integration_points"])
    absorbed = N - n_out                                             # mol/s absorbed above this slice
    oh = loop["lean"]["OH"] - 2 * absorbed
    co3 = loop["lean"]["CO3"] + absorbed
    water = loop["lean"]["W"] + M_H2O * absorbed
    y = N / (n_inert + N)
    p = P * y

    prof = {k: np.empty_like(N) for k in ("rate", "Ha", "E", "Einf", "H", "kL", "gas_frac", "c_oh", "I")}
    sol, kin = cfg["solubility"], cfg["kinetics"]
    for i in range(len(N)):
        c = liquor_concentrations(oh[i] / water[i], co3[i] / water[i], sv["liquid_density"])
        I = ionic_strength(c)
        H = henry_solution(T_l, c, sol)
        D = diffusivity_co2(T_l, sol, sv["viscosity_ratio_to_water"])
        c_oh = c["OH-"] * 1000.0                                     # mol/m3
        k2 = k2_oh(T_l, I, kin)
        kl = k_liquid(L, sv["liquid_density"], mu_l, D, a_w, pk["specific_area"], pk["nominal_size"])
        ha = np.sqrt(D * k2 * c_oh) / kl
        E = ha / np.tanh(ha)
        KG = 1.0 / (1.0 / kg + H / (E * kl))
        flux = KG * p[i]
        p_i = p[i] - flux / kg
        einf = 1 + kin["oh_to_co2_diffusivity_ratio"] * c_oh / (2 * p_i / H)
        prof["rate"][i] = flux * a_w * area                          # mol/s per metre of height
        prof["Ha"][i], prof["E"][i], prof["Einf"][i] = ha, E, einf
        prof["H"][i], prof["kL"][i], prof["c_oh"][i], prof["I"][i] = H, kl, c["OH-"], I
        prof["gas_frac"][i] = (1 / kg) / (1 / KG)                    # share of resistance in gas film

    z = cumulative_trapezoid(1.0 / prof["rate"], N, initial=0.0)    # depth below the top of the packing, m
    height = float(z[-1])
    return {
        "T_gas": T_g, "T_liq": T_l, "P": P, "rho_g": rho_g, "q_gas": q_gas, "u_g": u_g,
        "area": area, "diameter": diameter, "m_gas": m_gas, "m_liq": m_liq,
        "L_flux": L, "G_flux": G, "lg_mass_ratio": m_liq / m_gas, "q_liq_m3h": q_liq_m3h,
        "wetting": wetting, "wetting_min": wetting_min, "wetting_ok": wetting >= wetting_min,
        "a_w": a_w, "a_w_frac": a_w / pk["specific_area"], "kG": kg, "mu_l": mu_l,
        "height": height, "design_height": height * (1 + pk["design_margin"]),
        "packing_volume": area * height * (1 + pk["design_margin"]),
        "y_in": y[-1], "y_out": y[0],
        "profile": {"N": N, "z_from_top": z, "y": y, **prof},
        "ha_min": float(prof["Ha"].min()), "einf_over_ha_min": float((prof["Einf"] / prof["Ha"]).min()),
        "gas_resistance_max": float(prof["gas_frac"].max()),
        "oh_out_mol_L": float(prof["c_oh"][-1]),
    }
