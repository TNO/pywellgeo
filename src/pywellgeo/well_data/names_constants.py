from dataclasses import dataclass


@dataclass
class Constants:
    """
    This is a dataclass that contains simulation and well specific constants.
    It is used to store constants that are used throughout the simulation and well calculations.
    """

    MAINWELLBORE = "main_wellbore"
    WELLTREE = "welltree"

    GJ2kWh = 1e9 / 3.6e6

    INCH_SI = 0.0254
    SI_INCH = 1 / 0.0254

    SI_BAR = 1e-5
    BAR_SI = 1e5

    DARCY = 9.869233e-13
