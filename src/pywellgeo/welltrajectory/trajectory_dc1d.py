from typing import Dict, Optional, Union

import numpy as np
import yaml
from numpy import float64
from pythermonomics.config.geothermal_economics_config import GeothermalEconomicsConfig

from pywellgeo.well_data.dc1dwell import Dc1dwell
from pywellgeo.well_data.names_constants import Constants
from pywellgeo.well_tree.well_tree_tno import WellTreeTNO
from pywellgeo.welltrajectory.trajectory_base import TrajectoryBase


def read_input(config):
    """Read .yml file with settings"""

    with open(config, "r") as f:
        settings = yaml.safe_load(f)

    return settings


class TrajectoryDc1d(TrajectoryBase):
    """
    TrajectoryBase creates the well trajectory from a detaile dwell trajectory input file, as
    developed in RESULT, t includes
    """

    def __init__(self, options: GeothermalEconomicsConfig):
        super().__init__(options, None)

    def read(
        self,
        trajectoryfile: str,
        trajectoryinstance: Optional[Dc1dwell] = None,
    ) -> Dict[str, Dict[str, Union[WellTreeTNO, Dict[str, float64]]]]:
        """
        DC1DWell derived class implementation of read method. Read trajectory file, appends parameters,
        constructs WXYZ for each well and returns a dictionary with all well trajectories.
        :param trajectoryfile: str, path to well trajectory file
        :param trajectoryinstance: Dc1dwell/Dict, instance of well trajectory object
        :return: Dict, dictionary containing all trajectories for all wells
        """
        self.trajectoryfile = trajectoryfile

        self.dc1d = Dc1dwell.from_configfile(trajectoryfile)
        if trajectoryinstance is not None:
            self.dc1d.update_params(**trajectoryinstance.get_params())
        # create a tw with amain producor and injector

        tw = {"INJ1": {}, "PRD1": {}}

        kop = self.dc1d.getPseudoKop()
        stepout = self.dc1d.L * 0.5

        self.options.energy_loss_parameters.useheatloss = self.dc1d.useheatloss
        self.options.energy_loss_parameters.tsurface = self.dc1d.tsurface
        self.options.energy_loss_parameters.tgrad = self.dc1d.tgrad

        for i, w in enumerate(tw):
            tvdtopres = self.dc1d.tvd[i]
            sign = 1
            if i == 1:
                sign = -1
            WXYZ = [
                [0, 0, 0],
                [0, 0, kop],
                [stepout * sign, 0, tvdtopres],
                [stepout * sign, 0, tvdtopres + self.dc1d.H],
            ]
            tw[w] = {}
            try:
                xyz = np.asarray(WXYZ)
            except Exception:
                print("problem with WXYZ")
            xyz = xyz.transpose()
            welltree = WellTreeTNO.from_xyz(
                xyz[0], xyz[1], -xyz[2], radius=self.dc1d.rw[i]
            )
            welltree.init_ahd()
            welltree.init_temperaturesoil(self.dc1d.tsurface, self.dc1d.tgrad)
            tw[w][Constants.WELLTREE] = welltree
            tw[w][Constants.MAINWELLBORE] = {}
            self.complement_tw(tw[w], welltree, tvdtopres)

        self.tw = tw
        return self.tw

    def complement_tw(
        self,
        twwell: Dict[str, WellTreeTNO],
        welltree: WellTreeTNO,
        tvdtopreservoir: int,
    ) -> None:
        for i, wbranch in enumerate(twwell):
            if wbranch == Constants.MAINWELLBORE:
                twwell[wbranch]["total_depth"] = welltree.get_ahd(TVD=10000)
                twwell[wbranch]["casing_shoe_MD"] = welltree.get_ahd(
                    TVD=tvdtopreservoir
                )
                twwell[wbranch]["top_reservoir_depth_MD"] = welltree.get_ahd(
                    TVD=tvdtopreservoir
                )
                twwell[wbranch]["top_reservoir_depth_TVD"] = tvdtopreservoir
                twwell[wbranch]["bottom_reservoir"] = tvdtopreservoir + self.dc1d.H
            elif wbranch != Constants.WELLTREE:
                wlist = welltree.getbranchlist(
                    name=wbranch, perforated=None, includemain=False
                )
                twwell[wbranch]["total_depth"] = wlist[-1].ahd
                twwell[wbranch]["branch_start_MD"] = wlist[0].xroot.ahd
                twwell[wbranch]["branch_start_TVD"] = -wlist[0].xroot.x[2]
                twwell[wbranch]["branch_end_MD"] = wlist[-1].ahd
                twwell[wbranch]["branch_end_TVD"] = -wlist[-1].x[2]
