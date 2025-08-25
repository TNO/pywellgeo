import os
from typing import Dict, List, Optional, Union

import numpy as np
from numpy import float64
from pandas.core.series import Series
from pythermonomics.config.geothermal_economics_config import GeothermalEconomicsConfig
from pythermonomics.data.base_well_results import BaseWellResults

import pywellgeo.well_data.water_properties as waterprop
from pywellgeo.well_data.dc1dwell import Dc1dwell
from pywellgeo.well_data.names_constants import Constants
from pywellgeo.well_tree.well_tree_tno import WellTreeTNO


class TrajectoryBase:
    """
    TrajectoryBase creates the well trajectory from the control file WXYZ and platfrom data
    it does not support detailed well trajectory input files or trajectoryinstance varying with sample number
    this should be implemented in the read method of the derived class, currently only supported for TrajectoryDc1d.py
    """

    FORMAT = "BASE"

    def __init__(
        self, options: GeothermalEconomicsConfig, simresults: BaseWellResults
    ) -> None:
        self.options = options
        self.simresults = simresults

    def read(
        self, trajectoryfile: str, trajectoryinstance: Optional[Dc1dwell] = None
    ) -> Dict[str, Dict[str, Union[WellTreeTNO, Dict[str, float64]]]]:
        """
        Base method for reading. Overwritten by all derived classes. Method signature differs between
        derived classes, but to use polymorphism later on the code full signature is as it is defined here.
        :param trajectoryfile: str, path to well trajectory file
        :param trajectoryinstance: Dc1dwell/Dict, instance of well trajectory object
        :return: Dict, dictionary containing all trajectories for all wells
        """
        self.construct_trajectory()
        return self.tw

    @property
    def traj(self):
        return self._traj

    def construct_trajectory(self) -> None:
        settings = self.options
        wells_and_states = self.simresults.wells_and_states
        WXYZ = self.simresults.WXYZ
        t = {}

        for w in wells_and_states.keys():
            t[w] = {}
            # assume casing shoe TVD at start of reservoir section in the simulation grid
            top_reservoir_depth_TVD = WXYZ[w][2][
                2
            ]  # first entry is platform, then kick-off, then first target which is top reservoir
            casing_shoe_TVD = WXYZ[w][2][2]
            xyz = np.asarray(WXYZ[w])
            xyz = xyz.transpose()
            well = WellTreeTNO.from_xyz(xyz[0], xyz[1], -xyz[2], radius=0.1)
            well.init_ahd()
            base_reservoir_depth_TVD = WXYZ[w][-1][2]

            if settings.energy_loss_parameters.useheatloss:
                well.init_temperaturesoil(
                    settings.energy_loss_parameters.tsurface,
                    settings.energy_loss_parameters.tgrad,
                )

            t[w][Constants.WELLTREE] = well
            t[w][Constants.MAINWELLBORE] = {}
            t[w][Constants.MAINWELLBORE][
                "top_reservoir_depth_TVD"
            ] = top_reservoir_depth_TVD
            t[w][Constants.MAINWELLBORE][
                "base_reservoir_depth_TVD"
            ] = base_reservoir_depth_TVD
            t[w][Constants.MAINWELLBORE]["top_reservoir_depth_MD"] = (
                top_reservoir_depth_TVD * self.options.techno_eco_param.well_curvfac
            )
            t[w][Constants.MAINWELLBORE]["casing_shoe_TVD"] = casing_shoe_TVD
            t[w][Constants.MAINWELLBORE]["casing_shoe_MD"] = (
                casing_shoe_TVD * self.options.techno_eco_param.well_curvfac
            )
            t[w][Constants.MAINWELLBORE]["total_depth"] = well.cumulative_ahd()

        self.tw = t

    def getdefaultwellstates(self):
        """
        Get the default well states for the wells in the well trajectory

        :return: dictionary with the well names and the state of the well (prod or inj)
        """
        wkeys = list(self.tw.keys())
        wells_and_states = {wkeys[0]: "inj", wkeys[1]: "prod"}
        return wells_and_states

    def temploss_all(
        self,
        qvol: Series,
        templist: List[Series],
        salinity: Union[float, int],
        wells_and_states: Optional[Dict[str, str]] = None,
        reftime: int = 1,
        k: int = 3,
        at: Optional[float] = 1.2e-6,
        wellradius: Optional[float] = None,
    ) -> List[Series]:
        """
        Calculate the temeperature losses for the production wells in the well trajectory, for a given flowrate

        :param qvol: volumetric flow rate m3/s
        :param templist: temperature of the wells   in C
        :param salinity: fixed salinity in ppm
        :param wells_and_states: dictionary with the well names and the state of the well (prod or inj), same length as templist
        :param wellradius: radius of each of the wells, if None it will take the radius of the main wellbore
        :param reftime: reference time for the temperature losses in years
        :param k: rock thermal conductivity W/mK
        :param at: rock thermal diffusivity m2/s
        :return: temperature losses, list with same length as templist
        """
        if wells_and_states is None:
            wells_and_states = self.getdefaultwellstates()

        time = reftime * 365 * 24 * 3600

        templosses = []
        for i, w in enumerate(wells_and_states.keys()):
            if wells_and_states[w] == "prod":
                temp = templist[i]
                temploss = self.temploss(
                    w, qvol, temp, salinity, time, k, at, wellradius
                )
                templosses.append(temploss)
            else:
                templosses.append(templist[i] * 0)
        return templosses

    def temploss(
        self,
        well: str,
        qvols: Series,
        temp: Series,
        salinity: Union[float, int],
        time: int,
        k: int,
        at: float,
        wellradius: float,
    ) -> Series:
        """
        Calculate the production heat loss of the wells W/m, assuming one year of operation and taking the main branch as representative

        :param well: considered production well
        :param qvols: volumetric flow rate m3/s
        :param temp: bottom hole temperature of the well in C
        :param salinity: fixed salinity in ppm
        :param time: reference time for the temperature losses in seconds
        :param k: rock thermal conductivity W/mK
        :param at: rock thermal diffusivity m2/s
        :param wellradius: radius of the well in m
        :return: temperature losses in C
        :return:
        """
        tvdmax = self.res_mainTVD(well)
        bhp = tvdmax * 1e4
        density = waterprop.density(bhp, temp, salinity * 1e-6)
        massflow = qvols * density  # kg/s
        cpfluid = waterprop.heatcapacity(temp, salinity * 1e-6)
        temploss = 0
        twwell = self.tw[well]
        welltree = twwell[Constants.WELLTREE]
        welltree.init_ahd()  # ensure AHD is initialized
        wlist = welltree.getbranchlist(name="main", perforated=None, includemain=True)
        rwlist = wlist[::-1]
        tempactual = temp * 1.0
        for r in rwlist:
            dtemp = r.temploss(massflow * cpfluid, tempactual, time, k, at, wellradius)
            tempactual = tempactual * 1.0 - dtemp
        temploss = temp - tempactual
        return temploss

    def friction_all(
        self,
        qvol: Series,
        templist: List[Series],
        salinity: float,
        tubingdiameter_inch: float,
        roughness_minch: float,
        wells_and_states: Optional[Dict[str, str]] = None,
    ) -> Series:
        """
        Calculate the friction losses for the wells in the well trajectory, for a given flowrate,
        for branching wells the flowrate is assumed to be equally distributed over the branches

        :param qvol: volumetric flow rate m3/s
        :param templist: temperature of the wells   in C
        :param salinity: fixed salinity in ppm
        :param tubingdiameter_inch: tubing diameter in inch of the wells
        :param roughness_minch: roughness in milli-inch of the wells
        :param wells_and_states: dictionary with the well names and the state of the well (prod or inj)
        :return: total pressure losses for flow rate in bar, but will store as members of the well trajectory
                 dpsum, dp_frictioninj,  dp_frictionprod, dpsyphon, dpsum = dp_frictioninj + dp_frictionprod + dpsyphon
        """

        if wells_and_states is None:
            wells_and_states = self.getdefaultwellstates()

        tw = self.tw
        icount = 0
        tvdmaxave = 0
        for w in wells_and_states.keys():
            tvdmaxave += self.res_mainTVD(w)
            icount += 1
        tvdmaxave /= icount

        self.dpsum = 0
        self.dp_frictioninj = 0
        self.dp_frictionprod = 0
        for i, w in enumerate(wells_and_states.keys()):
            tvdmax = self.res_mainTVD(w)
            bhp = (
                tvdmax * 1e4
            )  # estimate of the BHP , based on the hydrostatic pressure
            temp = templist[i]
            rhow = waterprop.density(bhp, temp, salinity * 1e-6)
            visc = waterprop.viscosity(temp, salinity * 1e-6)

            if wells_and_states[w] == "prod":
                Pwprod = waterprop.getWellPres(tvdmaxave, templist[i], salinity * 1e-6)
            elif wells_and_states[w] == "inj":
                Pwinj = waterprop.getWellPres(tvdmaxave, templist[i], salinity * 1e-6)

            Qmass = qvol * rhow
            dP = self.friction(
                Qmass, rhow, visc, tubingdiameter_inch, roughness_minch, w
            )
            if wells_and_states[w] == "prod":
                self.dp_frictionprod += dP
            else:
                self.dp_frictioninj += dP
            tw[w][Constants.MAINWELLBORE]["dp_friction"] = dP
            self.dpsum += dP
        self.dpSyphon = Pwprod - Pwinj
        self.dpsum += self.dpSyphon
        return self.dpsum

    def friction(
        self,
        Qmass: Series,
        density: Series,
        viscosity: Series,
        tubingdiameter_inch: float,
        roughness_minch: float,
        well: str,
    ) -> Series:
        """
        The friction losses are calculated according to turbulent flow assuming a
        that all Qmass is flowing through the well. The algorithm takes into accoun that for the deeper sections
        flow can be divided over multiple branches. To this end, two section length with different flowrates are assumed.
        The first (main) section sustains all flow, the part wherte multiple branches occur the flow is assumed
        equally distributed over the number of existing branches in the well description

        :param Qmass: mass flow rate kg/s
        :param density:  density kg /m3
        :param viscosity:  viscosity  Pas
        :param tubingdiameter_inch: tubing diameter in inch
        :param roughness_minch: roughness in milli-inch
        :param well: wellname
        :return: Pressure losses for flow rate bar
        """
        twwell = self.tw[well]
        welltree = twwell[Constants.WELLTREE]
        wlist = welltree.getbranchlist(name="main", perforated=None, includemain=True)
        dL = wlist[-1].ahd

        lbranches = []
        for i, wbranch in enumerate(twwell):
            if (wbranch != Constants.WELLTREE) and (wbranch != Constants.MAINWELLBORE):
                wlist = welltree.getbranchlist(
                    name=wbranch, perforated=None, includemain=False
                )
                lbranch = wlist[-1].ahd - wlist[0].xroot.ahd
                lbranches.append(lbranch)

        if len(lbranches) > 0:
            avelbranch = sum(lbranches) / len(lbranches)
            dL -= avelbranch

        tubingdiameter = tubingdiameter_inch * Constants.INCH_SI
        roughness = roughness_minch * Constants.INCH_SI * 1e-3
        dPsign = np.sign(Qmass)

        Qvol = Qmass / density
        v = Qvol / (np.pi * 0.25 * tubingdiameter**2)
        Re = density * v * tubingdiameter / viscosity

        tempVariable = 1.14 - 2 * np.log10(
            roughness / tubingdiameter + 21.25 / (Re**0.9)
        )
        f = 1 / (tempVariable * tempVariable)
        dP = dPsign * f * density * dL * v * v / (2 * tubingdiameter)

        if len(lbranches) > 0:
            Qvol /= len(lbranches) + 1
            v = Qvol / (np.pi * 0.25 * tubingdiameter**2)
            Re = density * v * tubingdiameter / viscosity

            tempVariable = 1.14 - 2 * np.log10(
                roughness / tubingdiameter + 21.25 / (Re**0.9)
            )
            f = 1 / (tempVariable * tempVariable)
            dP += dPsign * f * density * avelbranch * v * v / (2 * tubingdiameter)

        return dP * Constants.SI_BAR

    def res_mainTVD(self, well: str) -> float64:
        """

        :param well: wellname
        :return: tvd of end of the main branch for the well
        """
        wtree = self.tw[well][Constants.WELLTREE]
        wlist = wtree.getbranchlist(name="main", perforated=None, includemain=True)

        return -wlist[-1].x[2]

    def plot(self, fname: str):
        """
        Plot the well trajectory (the wells in the well trajectory, to the basename of the file (excluding extention)

        :param fname: filename
        """

        outputnamebase = os.path.join(
            os.getcwd() + "\\", "output", str(fname).split(".")[0]
        )
        tw = self.tw
        for i, w in enumerate(tw):
            welltree = tw[w][Constants.WELLTREE]
            if i == 0:
                fig, ax = welltree.plotTree3D(
                    doplot=(i == len(tw) - 1),
                    tofile=outputnamebase + "_welltree_3D.png",
                )
            else:
                welltree.plotTree3D(
                    fig=fig, ax=ax, doplot=(i == len(tw) - 1), tofile=outputnamebase
                )
