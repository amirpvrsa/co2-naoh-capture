"""Loop, energy balance and absorber checked against hand calculations,
element balances and an analytic special case."""
import copy

import numpy as np
import pytest

import src.absorber as absorber
from src.config import load_config
from src.loop import (M_HYD, crystals_formed, element_balance, energy_balance, feed_gas,
                      liquor_mass, solubility, solve_loop)
from src.properties import M_H2O, M_NA2CO3

CFG = load_config()


def test_feed_gas_hand():
    # 10,000 Nm3/h / 0.022414 m3/mol / 3600 s/h = 123.93 mol/s; CO2 = 8% = 9.914 mol/s
    g = feed_gas(CFG)
    assert g["total"] == pytest.approx(123.93, rel=1e-4)
    assert g["flows"]["CO2"] == pytest.approx(0.08 * 123.93, rel=1e-4)


def test_solubility_interpolation():
    # 5 degC sits halfway between 7.0 (0 degC) and 12.5 (10 degC) -> 9.75 g/100 g -> 0.0975
    assert solubility(5.0, CFG["crystallizer"]) == pytest.approx(0.0975)
    with pytest.raises(ValueError):
        solubility(45.0, CFG["crystallizer"])


def test_crystals_formed_hand():
    # 2 mol Na2CO3 in 1 kg water, s = 0.10:
    # x = (2*0.105988 - 0.1*1) / (0.105988 - 10*0.1*0.018015) = 0.111976 / 0.087973 = 1.27285
    x = crystals_formed(2.0, 1.0, 0.10)
    assert x == pytest.approx(0.111976 / 0.087973, rel=1e-4)
    # mother liquor left behind is exactly saturated
    assert (2 - x) * M_NA2CO3 / (1 - 10 * M_H2O * x) == pytest.approx(0.10, rel=1e-9)


def test_element_balances_close():
    e = element_balance(solve_loop(CFG))
    assert e["na_in"] == pytest.approx(e["na_out"], rel=1e-9)
    assert e["c_in"] == pytest.approx(e["c_out"], rel=1e-9)


def test_no_cake_loss_means_stoichiometric_naoh():
    cfg = copy.deepcopy(CFG)
    cfg["crystallizer"]["cake_moisture"] = 0.0
    loop = solve_loop(cfg)
    assert loop["crystals"] == pytest.approx(loop["C"], rel=1e-9)       # every CO2 leaves as one crystal
    assert loop["makeup_naoh"] == pytest.approx(2 * loop["C"], rel=1e-9)  # 2 NaOH per CO2


def test_mother_liquor_is_saturated():
    loop = solve_loop(CFG)
    ml = loop["mother_liquor"]
    assert ml["CO3"] * M_NA2CO3 / ml["W"] == pytest.approx(loop["solubility"], rel=1e-9)


def test_lean_liquor_meets_spec():
    loop = solve_loop(CFG)
    sv = CFG["solvent"]
    assert loop["lean"]["OH"] / loop["lean"]["W"] == pytest.approx(sv["lean_naoh_molality"])
    assert 1 - loop["rich"]["OH"] / loop["lean"]["OH"] == pytest.approx(sv["per_pass_naoh_conversion"])


def test_energy_balance_hand():
    loop = solve_loop(CFG)
    e = energy_balance(loop, CFG)
    cp = CFG["crystallizer"]["liquor_cp"]
    # lean heated from 5 to 35 degC
    assert e["q_cross_exchanger"] == pytest.approx(liquor_mass(loop["lean"]) * cp * 30)
    # reaction heat = 109.5 kJ per mol CO2
    assert e["q_reaction"] == pytest.approx(109.5e3 * loop["C"])


def test_absorber_matches_analytic_gas_film_case(monkeypatch):
    # Make the reaction effectively infinitely fast: liquid resistance vanishes and K_G = k_G,
    # a constant. Then the height has a closed form for a solute in an inert carrier
    # (Y = N / n_inert, the CO2-to-inert mole ratio):
    #   Z = n_inert / (k_G a_w A P) * [ ln(Y_in / Y_out) + (Y_in - Y_out) ]
    monkeypatch.setattr(absorber, "k2_oh", lambda T, I, kin: 1e20)
    loop = solve_loop(CFG)
    r = absorber.size_absorber(loop, CFG)
    n_in = loop["gas"]["flows"]["CO2"]
    n_inert = loop["gas"]["total"] - n_in
    y_in, y_out = n_in / n_inert, (n_in - loop["C"]) / n_inert
    z = n_inert / (r["kG"] * r["a_w"] * r["area"] * r["P"]) * (np.log(y_in / y_out) + y_in - y_out)
    assert r["height"] == pytest.approx(z, rel=1e-5)


def test_absorber_diameter_hand():
    r = absorber.size_absorber(solve_loop(CFG), CFG)
    # u = F / sqrt(rho_G); A = Q / u
    assert r["u_g"] == pytest.approx(CFG["packing"]["design_f_factor"] / np.sqrt(r["rho_g"]))
    assert r["area"] == pytest.approx(r["q_gas"] / r["u_g"])
    assert r["diameter"] == pytest.approx(np.sqrt(4 * r["area"] / np.pi))


def test_absorber_regime_checks_hold_in_base_case():
    r = absorber.size_absorber(solve_loop(CFG), CFG)
    assert r["ha_min"] > CFG["report"]["min_hatta_for_fast_regime"]
    assert r["einf_over_ha_min"] > CFG["report"]["min_einf_over_ha"]
    assert r["wetting_ok"]


def test_more_capture_needs_more_packing():
    heights = []
    for cap in (0.8, 0.9, 0.95):
        cfg = copy.deepcopy(CFG)
        cfg["design_basis"]["capture_fraction"] = cap
        heights.append(absorber.size_absorber(solve_loop(cfg), cfg)["height"])
    assert heights[0] < heights[1] < heights[2]


def test_reaction_conserves_mass():
    from src.properties import M_CO2, M_NA2CO3, M_NAOH
    assert M_CO2 + 2 * M_NAOH == pytest.approx(M_NA2CO3 + M_H2O, abs=1e-15)


def test_plant_mass_balance_closes():
    from src.streams import overall_mass_balance, stream_table
    loop = solve_loop(CFG)
    mb = overall_mass_balance(stream_table(loop, energy_balance(loop, CFG), CFG))
    assert mb["in"] == pytest.approx(mb["out"], rel=1e-10)
