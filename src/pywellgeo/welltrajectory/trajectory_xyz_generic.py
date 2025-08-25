from typing import Any, Dict, Optional, Union

import numpy as np
import yaml
from pythermonomics.config.geothermal_economics_config import GeothermalEconomicsConfig
from pythermonomics.data.base_well_results import BaseWellResults

from pywellgeo.well_data.dc1dwell import Dc1dwell
from pywellgeo.well_data.names_constants import Constants
from pywellgeo.well_tree.well_tree_tno import WellTreeTNO
from pywellgeo.welltrajectory.trajectory_base import TrajectoryBase


def densify_by_max_distance(coords, mindist):
    coords = np.asarray(coords)
    densified = [coords[0]]
    for i in range(1, len(coords)):
        start = densified[-1]
        end = coords[i]
        dist = np.linalg.norm(end - start)
        if dist > mindist:
            num_new = int(np.ceil(dist / mindist)) - 1
            for j in range(1, num_new + 1):
                new_point = start + (end - start) * (j / (num_new + 1))
                densified.append(new_point)
        densified.append(end)
    return np.array(densified)


def read_input(config):
    """Read .yml file with settings"""

    with open(config, "r") as f:
        settings = yaml.safe_load(f)

    return settings


class TrajectoryXyzGeneric(TrajectoryBase):
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

        with open(trajectoryfile, "r") as f:
            trajectory_yml = yaml.safe_load(f)

        tw = trajectory_yml["well_trajectories"]
        tvdtopreservoir = trajectory_yml["reservoir"]["basic"][
            "top_reservoir_depth_TVD"
        ]
        tvdbottomreservoir = trajectory_yml["reservoir"]["basic"][
            "bottom_reservoir_depth_TVD"
        ]
        for w in tw:
            self.xyz = np.asarray(tw[w]["main_wellbore"]["xyz"], dtype=np.float64)
            radius = tw[w]["main_wellbore"]["radius"]
            mindist = tw[w]["main_wellbore"]["mindist"]
            self.xyz = densify_by_max_distance(self.xyz, mindist)
            xyz = self.xyz.transpose()
            welltree = WellTreeTNO.from_xyz(xyz[0], xyz[1], -xyz[2], radius=radius)
            branches = list(tw[w].keys())
            branches.remove("main_wellbore")
            for sub in branches:
                xyz = np.asarray(tw[w][sub]["xyz"], dtype=np.float64)
                radius = tw[w][sub]["radius"]
                mindist = tw[w][sub]["mindist"]
                xyz = densify_by_max_distance(xyz, mindist)
                xyz = xyz.transpose()
                welltree.add_xyz(xyz[0], xyz[1], -xyz[2], radius=radius, sbranch=sub)

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
