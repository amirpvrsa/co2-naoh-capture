"""Onda, Takeuchi & Okumoto (1968) correlations for random packing.

Wetted area:
    a_w / a_p = 1 - exp(-1.45 (sigma_c/sigma)^0.75 Re_L^0.1 Fr_L^-0.05 We_L^0.2)
    Re_L = L / (a_p mu_L),  Fr_L = L^2 a_p / (rho_L^2 g),  We_L = L^2 / (rho_L sigma a_p)
Liquid-side coefficient (physical absorption):
    k_L (rho_L / (mu_L g))^(1/3) = 0.0051 (L / (a_w mu_L))^(2/3) Sc_L^(-1/2) (a_p d_p)^0.4
Gas-side coefficient:
    k_G R T / (a_p D_G) = C (G / (a_p mu_G))^0.7 Sc_G^(1/3) (a_p d_p)^-2

    L, G     = liquid and gas superficial mass fluxes, kg/(m2 s)
    a_p      = packing specific area, m2/m3;  d_p = nominal packing size, m
    sigma_c  = critical surface tension of the packing material, sigma = liquid surface tension, N/m
    Sc       = mu / (rho D), the Schmidt number;  C = 5.23 for packing larger than 15 mm
    k_L in m/s;  k_G in mol/(m2 s Pa)
"""
from __future__ import annotations

import numpy as np

from src.properties import G_ACC, R


def wetted_area(L, rho_l, mu_l, sigma, a_p, sigma_c) -> float:
    re = L / (a_p * mu_l)
    fr = L ** 2 * a_p / (rho_l ** 2 * G_ACC)
    we = L ** 2 / (rho_l * sigma * a_p)
    return a_p * (1 - np.exp(-1.45 * (sigma_c / sigma) ** 0.75 * re ** 0.1 * fr ** -0.05 * we ** 0.2))


def k_liquid(L, rho_l, mu_l, d_l, a_w, a_p, d_p) -> float:
    sc = mu_l / (rho_l * d_l)
    return 0.0051 * (L / (a_w * mu_l)) ** (2 / 3) * sc ** -0.5 * (a_p * d_p) ** 0.4 \
        * (rho_l / (mu_l * G_ACC)) ** (-1 / 3)


def k_gas(G, rho_g, mu_g, d_g, a_p, d_p, T, C=5.23) -> float:
    sc = mu_g / (rho_g * d_g)
    return C * (G / (a_p * mu_g)) ** 0.7 * sc ** (1 / 3) * (a_p * d_p) ** -2 * a_p * d_g / (R * T)
