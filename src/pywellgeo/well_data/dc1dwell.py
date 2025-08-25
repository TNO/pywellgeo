from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
from numpy import float64

import pywellgeo.well_data.water_properties as waterprop
from pywellgeo.well_data.names_constants import Constants


def find_kod_binary_search(
    ahd: float64,
    L: int,
    tvd: float64,
    tol: Optional[float] = 1e-6,
    max_iter: Optional[int] = 1000,
) -> float64:
    """
    Find the solution for kod in the equation:
    ahd = kod + sqrt(0.25*L^2 + (tvd-kod)^2) using binary search.

    Parameters
    ----------
    ahd : float
        Along hole depth.
    L : float
        Distance between the wells.
    tvd : float
        True vertical depth.
    tol : float, optional
        Tolerance for the solution (default is 1e-6).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).

    Returns
    -------
    kod : float
        Kick off depth.
    """

    def equation(kod):
        return ahd - (kod + np.sqrt(0.25 * L**2 + (tvd - kod) ** 2))

    low, high = 0, tvd
    for _ in range(max_iter):
        mid = (low + high) / 2
        if abs(equation(mid)) < tol:
            return mid
        elif equation(mid) > 0:
            low = mid
        else:
            high = mid
    raise ValueError("Solution did not converge")


def read_input(config: str) -> Dict[str, Any]:
    """Read .yml file with settings"""

    with open(config, "r") as f:
        settings = yaml.safe_load(f)

    return settings


class Dc1dwell:
    """
    Class to calculate the productivity of a well in a confined, infinite, homogeneous reservoir

    """

    def __init__(
        self,
        k: int,
        H: int,
        L: int,
        tvd: List[int],
        temp: List[Union[float, int]],
        salinity: List[int],
        skin: List[float],
        ahd: List[int],
        rw: List[float],
        roughness: float,
        tgrad: Optional[float] = 0.031,
        tsurface: Optional[float] = 10,
        use_tgrad: Optional[bool] = False,
        useheatloss: Optional[bool] = False,
    ) -> None:
        """
        instantiate a dc1dwell object,  tvd, temp, salinity, skin, ahd, rw are arrays of length 2, first index is production well, second index is injection well

        :param k: permeability of the aquifer [mDarcy
        :param H: thickness of the aquifer [m]
        :param L: distance between the wells [m]
        :param tvd: true vertical depth of top of the reservoir in the wells [m] ndarray len = 2
        :param temp: temperature of the wells ( tinj and tres) [C] ndarray len = 2
        :param salinity: salinity [ppm] ndarray len = 2
        :param skin: skin factor [-]  ndarray len = 2
        :param ahd: along hole depth of the wells [m] up till the bottom of the reservoir, ndarray len = 2
        :param rw: well radius  ndarray len = 2
        :param roughness: roughness of the wells [ milli-inch ]

        """
        self.H = H
        self.k = k
        self.L = L

        self.tvd = tvd
        self.temp = temp
        self.salinity = salinity
        self.skin = skin
        self.ahd = ahd
        self.rw = rw
        self.roughness = roughness
        self.use_tgrad = use_tgrad
        self.tsurface = tsurface
        self.tgrad = tgrad
        if use_tgrad:
            self.temp[1] = tsurface + (self.tvd[1] + 0.5 * self.H) * self.tgrad
        self.useheatloss = useheatloss

    def update_restemp(self) -> None:
        if self.use_tgrad:
            self.temp[1] = self.tsurface + (self.tvd[1] + 0.5 * self.H) * self.tgrad

    def update_params(self, **kwargs) -> None:
        """
        update the parameters of the dc1dwell object
        :param kwargs:  parameter dictonary to update

        :return:
        """
        add_new = True
        for key, value in kwargs.items():
            if add_new or hasattr(self, key):
                setattr(self, key, value)
            else:
                print(
                    "parameter ",
                    key,
                    " not found in dc1dwell object in  Dc1dwell.update_params()",
                )
        if self.use_tgrad:
            self.update_restemp()

    def get_params(self) -> Dict[str, Any]:
        """
        get the parameters of the dc1dwell object
        :return: dictionary with the parameters of the dc1dwell object
        """
        return self.__dict__.copy()

    @classmethod
    def from_configfile(cls, configfile: str) -> "Dc1dwell":
        """
        create a dc1dwell object from a configfile

        :param configfile: yaml file with the parameters of the dc1dwell object
        :return: dc1dwell object
        """

        inputs = read_input(configfile)
        k = inputs["k"]
        H = inputs["H"]
        L = inputs["L"]
        tvd = inputs["tvd"]
        temp = inputs["temp"]
        salinity = inputs["salinity"]
        skin = inputs["skin"]
        ahd = inputs["ahd"]
        rw = inputs["rw"]
        roughness = inputs["roughness"]
        use_tgrad = temp[1] == -1
        tgrad = inputs["tgrad"]
        tsurface = inputs["tsurface"]
        useheatloss = inputs["useheatloss"]

        return cls(
            k,
            H,
            L,
            tvd,
            temp,
            salinity,
            skin,
            ahd,
            rw,
            roughness,
            tgrad,
            tsurface,
            use_tgrad,
            useheatloss=useheatloss,
        )

    def getPseudoKop(self) -> float64:
        """
        calculate the kick off depth for a linearized well trajectory, such that the Ahd for the top  corresponds to
        ahd = kod + sqrt( 0.25*L^2 + (tvd-kod)^2)
        :return:
        """

        tvda = np.average(self.tvd)
        ahda = np.average(self.ahd)
        if tvda + 0.5 * self.L < ahda:
            print(
                " fatal error in ahd: too large to be linearized with pseudo Kickoff depth of tvd"
            )
        else:
            kod = find_kod_binary_search(ahda, self.L, tvda)
        return kod

    def calculateDP(self, qvol: Union[float, float64]) -> None:
        self.qvol = qvol
        self.productivity()
        self.dp_syphon()
        self.dp_friction(qvol)
        self.dpres = self.PI_II * qvol
        self.dp = self.dpres + self.dpfriction + self.dpsyphon
        if self.useheatloss:
            self.tprod = self.tproduction()
        else:
            self.tprod = self.temp[1]

    def calculateQvol(
        self,
        target_dp: int,
        initial_guess: Optional[float] = 0.05,
        tol: Optional[float] = 1e-4,
        max_iter: Optional[int] = 100,
    ) -> float64:
        """
        Use Newton-Raphson method to find the flow rate that results in the target pressure drop.

        :param target_dp: Target pressure drop
        :param initial_guess: Initial guess for the flow rate [m3/s]
        :param tol: Tolerance for the solution (default is 1e-4)
        :param max_iter: Maximum number of iterations
        :return: Flow rate that results in the target pressure drop
        """

        def f(qvol):
            self.calculateDP(qvol)
            return self.dp - target_dp

        def f_prime(qvol):
            h = 1e-5
            return (f(qvol + h) - f(qvol)) / h

        qvol = initial_guess
        for _ in range(max_iter):
            f_val = f(qvol)
            if abs(f_val) < tol:
                return qvol
            qvol -= f_val / f_prime(qvol)

        raise ValueError("Solution did not converge")

    def calculateDP_qvol(self) -> None:
        """
        calculate the pressure drop and production temperature of the wells
        for a given flow rate (if self.qvol is set to positive number)
        or it caluclates achievable flow rate and production temperature for the given flowrate
        :return: sets self.qvol or self.dp, and sets self.tprod.
        """
        if self.qvol > 0:
            self.calculateDP(self.qvol)
        else:
            self.calculateQvol(self.dp)

    def productivity(self) -> None:
        """
        calculate the productivity index of the wells (i.e. pressure at reservoir level required for a unit flow rate [m3/s]

        :return: productivity index
        """

        scalefac = 2 * np.pi * self.k * Constants.DARCY * 1e-3 * self.H
        self.PI_II = 0
        wellfac = 0
        for i in range(2):
            mu = waterprop.viscosity(self.temp[i], self.salinity[i] * 1e-6)
            wf = mu * (np.log(self.L / self.rw[i]) + self.skin[i])
            wellfac += wf
        self.PI_II = Constants.SI_BAR * wellfac / scalefac

    def dp_friction(self, qvol: Union[float, float64]) -> None:
        """
        calculate the frictional pressure drop in the wells

        """
        self.dpfriction = 0
        for i in range(2):
            dL = self.ahd[i] + self.H
            density = waterprop.density(
                self.tvd[i] * 0.5, self.temp[i], self.salinity[i] * 1e-6
            )
            viscosity = waterprop.viscosity(self.temp[i], self.salinity[i] * 1e-6)
            tubingdiameter = self.rw[i] * 2
            roughness = self.roughness * Constants.INCH_SI * 1e-3

            v = qvol / (np.pi * 0.25 * tubingdiameter**2)
            Re = density * v * tubingdiameter / viscosity

            tempVariable = 1.14 - 2 * np.log10(
                roughness / tubingdiameter + 21.25 / (Re**0.9)
            )
            f = 1 / (tempVariable * tempVariable)
            dP = f * density * dL * v * v / (2 * tubingdiameter)
            self.dpfriction += dP * Constants.SI_BAR

    def dp_syphon(self) -> None:
        """
        calculate the pressure in the well

        :return: pressure
        """

        Pwprod = waterprop.getWellPres(
            self.tvd[1] + 0.5 * self.H, self.temp[1], self.salinity[1] * 1e-6
        )
        Pwinj = waterprop.getWellPres(
            self.tvd[0] + 0.5 * self.H, self.temp[0], self.salinity[0] * 1e-6
        )
        self.dpsyphon = Pwprod - Pwinj

    def tproduction(self) -> float64:
        """
        calculate the heat loss of the wells W/m, asusming one year of operation

        :param temp:  fluid temperature in the well
        :param tempenv: environmental temperature
        :param rw: well radius
        :return:
        """
        time = 365 * 24 * 3600
        kt = 3  # rock thermal conductivity W/mK
        at = 1.2e-6  # rock thermal diffusivity m2/s
        density = waterprop.density(
            self.tvd[1] * 0.5, self.temp[1], self.salinity[1] * 1e-6
        )
        massflow = self.qvol * density  # kg/s
        cpfluid = waterprop.heatcapacity(self.temp[1], self.salinity[1] * 1e-6)

        # qloss = 4 * np.pi * kt * DT /   np.log(4 * at * time / rw**2))
        L = self.ahd[1]
        rw = self.rw[1]
        self.tgradahd = 0.031 * self.tvd[1] / self.ahd[1]
        tgrad = self.tgradahd

        overkz = (np.log((4 * at * time) / rw**2) - 0.5772) / (4 * np.pi * kt * L)
        wdot = massflow * cpfluid
        kzstar = (1 / overkz) / wdot
        F = self.temp[1]
        Estar = -tgrad * L
        cc2 = Estar / kzstar + 0.0
        theta = -Estar / kzstar + cc2 * np.exp(-kzstar)
        tend = F + Estar + theta

        return tend
