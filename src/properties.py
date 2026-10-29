"""Physical properties of CO2 in aqueous NaOH / Na2CO3 liquor.

All functions take temperature T in kelvin. Concentrations are in mol/L (= kmol/m3).

Henry's constant in water (Versteeg & van Swaaij, 1988):
    H_w = A * exp(-B / T)                             [Pa m3/mol]
Salting out in electrolyte (Schumpe model, Weisenberger & Schumpe, 1996):
    log10(H_sol / H_w) = sum_i (h_i + h_G) * c_i
    h_G = h_G0 + h_T * (T - 298.15)
    where c_i = concentration of ion i, h_i = ion parameter, h_G = gas parameter
Diffusivity of CO2 (Versteeg & van Swaaij, 1988), corrected for viscosity:
    D_w   = A * exp(-B / T)                           [m2/s]
    D_sol = D_w * (mu_w / mu_sol)^0.8
Second-order rate constant for CO2 + OH- (Pinsent et al., 1956; ionic-strength
correction of Pohorecki & Moniuk, 1988):
    log10 k_inf = A - B / T                           [L/(mol s)]
    log10(k / k_inf) = c1 * I - c2 * I^2
    I = 0.5 * sum_i c_i * z_i^2   (ionic strength; z_i = ion charge)
"""
from __future__ import annotations

import numpy as np

# Molar masses built from standard atomic weights (kg/mol) so every reaction conserves mass exactly
_NA, _C, _O, _H, _N = 22.98977e-3, 12.011e-3, 15.999e-3, 1.008e-3, 14.007e-3
M_NAOH = _NA + _O + _H
M_NA2CO3 = 2 * _NA + _C + 3 * _O
M_H2O = 2 * _H + _O
M_CO2 = _C + 2 * _O
M_O2, M_N2 = 2 * _O, 2 * _N
R = 8.314462618        # J/(mol K)
G_ACC = 9.80665        # m/s2


def liquor_concentrations(b_oh: float, b_co3: float, density: float) -> dict:
    """Molalities (mol per kg water) -> molar concentrations (mol/L).

        water mass fraction w = 1 / (1 + b_OH * M_NaOH + b_CO3 * M_Na2CO3)
        c_i = b_i * w * rho / 1000
    """
    w = 1.0 / (1.0 + b_oh * M_NAOH + b_co3 * M_NA2CO3)
    f = w * density / 1000.0
    c_oh, c_co3 = b_oh * f, b_co3 * f
    return {"OH-": c_oh, "CO3--": c_co3, "Na+": c_oh + 2 * c_co3}


def ionic_strength(c: dict) -> float:
    charges = {"Na+": 1, "OH-": -1, "CO3--": -2}
    return 0.5 * sum(c[i] * charges[i] ** 2 for i in c)


def henry_water(T: float, A: float, B: float) -> float:
    return A * np.exp(-B / T)


def henry_solution(T: float, c: dict, sol: dict) -> float:
    h_g = sol["gas_h0"] + sol["gas_hT"] * (T - 298.15)
    log_ratio = sum((sol["ion_h"][ion] + h_g) * conc for ion, conc in c.items())
    return henry_water(T, sol["henry_A"], sol["henry_B"]) * 10 ** log_ratio


def diffusivity_co2(T: float, sol: dict, viscosity_ratio: float) -> float:
    d_w = sol["diff_A"] * np.exp(-sol["diff_B"] / T)
    return d_w * (1.0 / viscosity_ratio) ** sol["diffusivity_viscosity_exponent"]


def k2_oh(T: float, I: float, kin: dict) -> float:
    """Second-order rate constant in m3/(mol s) (the L/(mol s) value divided by 1000)."""
    log_k_inf = kin["pinsent_A"] - kin["pinsent_B"] / T
    log_corr = kin["ionic_c1"] * I - kin["ionic_c2"] * I ** 2
    return 10 ** (log_k_inf + log_corr) / 1000.0


def water_viscosity(T: float) -> float:
    """Viscosity of water, Pa s (Vogel equation fit, good to ~1% from 0 to 100 degC):
        mu = 2.414e-5 * 10^(247.8 / (T - 140))
    """
    return 2.414e-5 * 10 ** (247.8 / (T - 140.0))
