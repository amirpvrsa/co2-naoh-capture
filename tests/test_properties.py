"""Property and mass-transfer correlations checked against hand calculations
and well-known literature values."""
import numpy as np
import pytest

from src.config import load_config
from src.masstransfer import k_gas, k_liquid, wetted_area
from src.properties import (diffusivity_co2, henry_solution, henry_water, ionic_strength,
                            k2_oh, liquor_concentrations, water_viscosity)

CFG = load_config()
SOL, KIN = CFG["solubility"], CFG["kinetics"]


def test_henry_water_25C():
    # 2.82e6 * exp(-2044 / 298.15) = 2.82e6 * 1.0531e-3 = 2969.8 Pa m3/mol
    # (about 29.3 atm L/mol, the textbook value for CO2 in water at 25 degC)
    assert henry_water(298.15, SOL["henry_A"], SOL["henry_B"]) == pytest.approx(2969.8, rel=1e-3)


def test_diffusivity_water_25C():
    # 2.35e-6 * exp(-2119 / 298.15) = 1.92e-9 m2/s, the accepted value at 25 degC
    assert diffusivity_co2(298.15, SOL, 1.0) == pytest.approx(1.92e-9, rel=0.01)


def test_diffusivity_viscosity_correction():
    # twice as viscous -> D times 0.5^0.8 = 0.5743
    assert diffusivity_co2(298.15, SOL, 2.0) / diffusivity_co2(298.15, SOL, 1.0) == pytest.approx(0.5743, rel=1e-3)


def test_pinsent_25C_infinite_dilution():
    # log10 k = 13.635 - 2895/298.15 = 3.9252 -> k = 8,417 L/(mol s) = 8.417 m3/(mol s)
    assert k2_oh(298.15, 0.0, KIN) == pytest.approx(8.417, rel=1e-3)


def test_ionic_strength_correction():
    # I = 2: log10 ratio = 0.221*2 - 0.016*4 = 0.378 -> factor 10^0.378 = 2.388
    assert k2_oh(298.15, 2.0, KIN) / k2_oh(298.15, 0.0, KIN) == pytest.approx(2.388, rel=1e-3)


def test_ionic_strength_hand():
    # 1 mol/L NaOH + 0.5 mol/L Na2CO3: Na+ = 2, OH- = 1, CO3-- = 0.5
    # I = 0.5 * (2*1 + 1*1 + 0.5*4) = 2.5
    assert ionic_strength({"Na+": 2.0, "OH-": 1.0, "CO3--": 0.5}) == pytest.approx(2.5)


def test_liquor_concentrations_hand():
    # 2 mol/kg NaOH, no carbonate, density 1000: water fraction = 1/(1 + 2*0.03999) = 0.92594
    # c_OH = 2 * 0.92594 * 1000/1000 = 1.85188 mol/L
    c = liquor_concentrations(2.0, 0.0, 1000.0)
    assert c["OH-"] == pytest.approx(1.85188, rel=1e-4)
    assert c["Na+"] == pytest.approx(c["OH-"])


def test_salting_out_hand():
    # 1 mol/L NaOH at 298.15 K: h_G = -0.0172
    # log10 ratio = (0.1143 - 0.0172)*1 + (0.0839 - 0.0172)*1 = 0.1638 -> ratio 1.4581
    c = {"Na+": 1.0, "OH-": 1.0, "CO3--": 0.0}
    ratio = henry_solution(298.15, c, SOL) / henry_water(298.15, SOL["henry_A"], SOL["henry_B"])
    assert ratio == pytest.approx(10 ** 0.1638, rel=1e-4)


def test_water_viscosity():
    assert water_viscosity(298.15) == pytest.approx(0.890e-3, rel=0.02)
    assert water_viscosity(313.15) == pytest.approx(0.653e-3, rel=0.02)


def test_onda_wetted_area_bounds_and_trend():
    args = dict(rho_l=1100, mu_l=1e-3, sigma=0.076, a_p=112.6, sigma_c=0.075)
    a1 = wetted_area(L=5, **args)
    a2 = wetted_area(L=15, **args)
    assert 0 < a1 < a2 < 112.6            # more liquid wets more packing, never more than a_p


def test_onda_kg_hand():
    # G=1, rho_g=1.2, mu_g=1.8e-5, D_G=1.65e-5, a_p=112.6, d_p=0.05, T=313.15
    # Re = 1/(112.6*1.8e-5) = 493.38; Sc = 1.8e-5/(1.2*1.65e-5) = 0.90909
    # (a_p d_p)^-2 = 5.63^-2 = 0.031548
    # kG = 5.23 * 493.38^0.7 * 0.90909^(1/3) * 0.031548 * 112.6 * 1.65e-5 / (8.31446 * 313.15)
    re, sc = 1 / (112.6 * 1.8e-5), 1.8e-5 / (1.2 * 1.65e-5)
    expected = 5.23 * re ** 0.7 * sc ** (1 / 3) * (112.6 * 0.05) ** -2 * 112.6 * 1.65e-5 / (8.314462618 * 313.15)
    assert k_gas(1.0, 1.2, 1.8e-5, 1.65e-5, 112.6, 0.05, 313.15) == pytest.approx(expected, rel=1e-12)
    assert 1e-6 < expected < 1e-4             # typical range for packed columns, mol/(m2 s Pa)


def test_onda_kl_typical_range():
    kl = k_liquid(L=10, rho_l=1100, mu_l=1e-3, d_l=2e-9, a_w=80, a_p=112.6, d_p=0.05)
    assert 5e-5 < kl < 5e-4                    # typical k_L for random packing, m/s
