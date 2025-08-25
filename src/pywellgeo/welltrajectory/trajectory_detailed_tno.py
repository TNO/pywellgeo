from typing import Any, Dict, Optional, Union

import yaml
from pythermonomics.config.geothermal_economics_config import GeothermalEconomicsConfig
from pythermonomics.data.base_well_results import BaseWellResults

from pywellgeo.well_data.dc1dwell import Dc1dwell
from pywellgeo.well_data.names_constants import Constants
from pywellgeo.well_tree.well_tree_azim_dip import WellTreeAzimDip
from pywellgeo.well_tree.well_tree_tno import WellTreeTNO
from pywellgeo.welltrajectory.trajectory_base import TrajectoryBase


def read_input(config):
    """Read .yml file with settings"""

    with open(config, "r") as f:
        settings = yaml.safe_load(f)

    return settings


class TrajectoryDetailedTNO(TrajectoryBase):
    """
    TrajectoryBase creates the well trajectory from a detaile dwell trajectory input file, as
    developed in RESULT, t includes
    """

    def __init__(self, options: GeothermalEconomicsConfig, simresults: BaseWellResults):
        super().__init__(options, simresults)

    def read(
        self, trajectoryfile: str, trajectoryinstance: Optional[Dc1dwell] = None
    ) -> Dict[
        str,
        Dict[
            str,
            Union[
                Dict[str, Any],
                WellTreeTNO,
            ],
        ],
    ]:
        """
        DetailedTno derived class implementation of read method. Reads the trajectory input
        and returns a dictionary with all well trajectories.
        :param trajectoryfile: str, path to well trajectory file
        :param trajectoryinstance: Dc1dwell/Dict, instance of well trajectory object
        :return: Dict, dictionary containing all trajectories for all wells
        """
        self.trajectoryfile = trajectoryfile

        trajectory_yml = WellTreeAzimDip.read_input(trajectoryfile)
        tw = trajectory_yml["well_trajectories"]
        tvdtopreservoir = trajectory_yml["reservoir"]["basic"][
            "top_reservoir_depth_TVD"
        ]
        tvdbottomreservoir = trajectory_yml["reservoir"]["basic"][
            "bottom_reservoir_depth_TVD"
        ]
        trj = None
        for w in tw:
            nroot = WellTreeAzimDip.process_inputs(trajectoryfile, w)
            trj = nroot.compute_trajectories(
                w, trajectories=trj, savecsv=True, step_depth=30
            )
            welltree = WellTreeTNO.from_trajectories(
                trj[w], tw[w]["main_wellbore"]["radius"]
            )
            welltree.init_ahd()
            if self.options.energy_loss_parameters.useheatloss:
                welltree.init_temperaturesoil(
                    self.options.energy_loss_parameters.tsurface,
                    self.options.energy_loss_parameters.tgrad,
                )
            tw[w][Constants.WELLTREE] = welltree
            self.complement_tw(tw[w], welltree, tvdtopreservoir, tvdbottomreservoir)
        self.tw = tw
        return self.tw

    def complement_tw(
        self,
        twwell: Dict[
            str,
            Union[
                Dict[str, Any],
                WellTreeTNO,
            ],
        ],
        welltree: WellTreeTNO,
        tvdtopreservoir: float,
        tvdbottomreservoir: float,
    ) -> None:
        for i, wbranch in enumerate(twwell):
            if wbranch == Constants.MAINWELLBORE:
                twwell[wbranch]["total_depth"] = welltree.get_ahd(TVD=10000)
                twwell[wbranch]["casing_shoe_MD"] = welltree.get_ahd(
                    TVD=tvdtopreservoir
                )
                twwell[wbranch]["top_reservoir_depth_TVD"] = tvdtopreservoir
                twwell[wbranch]["bottom_reservoir"] = tvdbottomreservoir
                twwell[wbranch]["top_reservoir_depth_MD"] = welltree.get_ahd(
                    TVD=tvdtopreservoir
                )
            elif wbranch != Constants.WELLTREE:
                wlist = welltree.getbranchlist(
                    name=wbranch, perforated=None, includemain=False
                )
                twwell[wbranch]["total_depth"] = wlist[-1].ahd
                twwell[wbranch]["branch_start_MD"] = wlist[0].xroot.ahd
                twwell[wbranch]["branch_start_TVD"] = -wlist[0].xroot.x[2]
                twwell[wbranch]["branch_end_MD"] = wlist[-1].ahd
                twwell[wbranch]["branch_end_TVD"] = -wlist[-1].x[2]
