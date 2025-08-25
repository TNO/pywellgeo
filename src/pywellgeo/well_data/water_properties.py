"""
WaterProperties
===============

This module contains functions to calculate various properties of water, such as viscosity, heat capacity, and density, based on temperature, pressure, and salinity.

Functions
---------
viscosity(T, S)
    Calculate water viscosity based on temperature and salinity.

viscosityKestin(P, T, S)
    Calculate water viscosity based on pressure, temperature, and salinity using Kestin's correlation.

heatcapacity(T, S)
    Calculate water heat capacity based on temperature and salinity.

density(P, T, S)
    Calculate water density based on pressure, temperature, and salinity.

getWellPres(depth, T, S)
    Calculate the bottom hole well pressure based on depth, temperature, and salinity.
"""

import numpy as np
from pandas.core.series import Series


def viscosity(T, S):
    """
    Calculate water viscosity.

    Parameters
    ----------
    T : float
        Reservoir temperature in degrees Celsius.
    S : float
        Salinity in ppm * 1e-6.

    Returns
    -------
    float
        Viscosity in Pa s.
    """
    v = (
        0.1
        + 0.333 * S
        + (1.65 + 91.9 * S**3) * np.exp(-(0.42 * (S**0.8 - 0.17) ** 2 + 0.045) * T**0.8)
    )
    v *= 1e-3
    return v


def viscosityKestin(P, T, S):
    """
    Calculate water viscosity using Kestin's correlation.

    Parameters
    ----------
    P : float
        Pressure in Pascal.
    T : float
        Reservoir temperature in degrees Celsius.
    S : float
        Salinity in ppm * 1e-6.

    Returns
    -------
    float
        Viscosity in Pa s.
    """
    T = max(T, 0)
    P = max(P, 1)
    S = max(S, 0)

    # Kestin '81 correlation (pressure incl pressure dependency)
    # convert to units used in correlation
    P *= 1e-6  # convert from Pa to MPa
    S_m = S / 0.05844  # mol/kg

    B = -3.960e-2 * S_m + 1.020e-2 * S_m**2 - 7.020e-4 * S_m**3
    A = 3.324e-2 * S_m + 3.624e-3 * S_m**2 - 1.879e-4 * S_m**3

    log_mu0 = (
        1.23780 * (20.0 - T)
        - 1.303e-3 * (20.0 - T) ** 2
        + 3.06e-6 * (20.0 - T) ** 3
        + 2.550e-8 * (20.0 - T) ** 4
    ) / (96.0 + T)

    mu_w_p0_20c = 1.002e-3  # (Pa.s)
    mu_w_p0 = 10 ** (log_mu0 + np.log10(mu_w_p0_20c))
    mu_p0 = 10 ** (A + B * log_mu0) * mu_w_p0

    beta_w = -1.297 + 5.74e-2 * T - 6.97e-4 * T**2 + 4.47e-6 * T**3 - 1.05e-8 * T**4
    beta_E = 0.545 + 2.8e-3 * T - beta_w
    beta_star = 2.5 * S - 2 * S**2 + 0.5 * S**3
    beta = beta_E * beta_star + beta_w

    viscosity = mu_p0 * (1 + 0.001 * beta * P)

    return viscosity


def heatcapacity(T: Series, S: float) -> Series:
    """
    Calculate water heat capacity.

    Parameters
    ----------
    T : float
        Reservoir temperature in degrees Celsius.
    S : float
        Salinity in ppm * 1e-6.

    Returns
    -------
    float
        Heat capacity in J kg-1 K-1.
    """
    Tp = T + 273.15
    Sp = S * 1e3

    heatCapacity = (
        (5.328 + -0.0976 * Sp + 0.000404 * Sp * Sp)
        + (-6.913 * 0.001 + 7.351 * 0.0001 * Sp - 3.15 * 0.000001 * Sp * Sp) * Tp
        + (9.6 * 0.000001 - 1.927 * 0.000001 * Sp + 0.00000000823 * Sp * Sp) * Tp * Tp
        + (
            2.5 * 0.000000001
            + 1.666 * 0.000000001 * Sp
            - 7.125 * 0.000000000001 * Sp * Sp
        )
        * Tp
        * Tp
        * Tp
    )
    heatCapacity = heatCapacity * 1000  #
    return heatCapacity


def density(P: Series, T: Series, S: float) -> Series:
    """
    Calculate water density.

    Parameters
    ----------
    P : float
        Reservoir pressure in Pascal.
    T : float
        Reservoir temperature in degrees Celsius.
    S : float
        Salinity in ppm * 1e-6.

    Returns
    -------
    float
        Density in kg m-3.
    """
    P *= 1e-6
    densityFresh = 1 + 1e-6 * (
        -80.0 * T
        - 3.3 * T * T
        + 0.00175 * T * T * T
        + 489.0 * P
        - 2.0 * T * P
        + 0.016 * T * T * P
        - 1.3e-5 * T * T * T * P
        - 0.333 * P * P
        - 0.002 * T * P * P
    )
    density = densityFresh + S * (
        0.668
        + 0.44 * S
        + 1e-6
        * (
            300.0 * P
            - 2400.0 * P * S
            + T * (80.0 + 3.0 * T - 3300.0 * S - 13.0 * P + 47.0 * P * S)
        )
    )
    density *= 1000.0
    return density


def getWellPres(depth, T, S):
    """
    Calculate the bottom hole well pressure.

    Parameters
    ----------
    depth : float
        TVD of well in meters.
    T : float
        Well temperature in degrees Celsius.
    S : float
        Brine salinity in ppm * 1e-6.

    Returns
    -------
    float
        Bottom hole well pressure in bars.
    """
    nseg = 10
    pwell = 0
    for i in range(nseg):
        P = (i + 0.5) * 9.81 * depth * 1e4 / nseg
        dens = density(P, T, S)
        pwell += 9.81 * dens * depth / nseg
    return pwell * 1e-5
