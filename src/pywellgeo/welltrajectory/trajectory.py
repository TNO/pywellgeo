from typing import Any, Dict, Optional

import yaml
from pythermonomics.config.geothermal_economics_config import GeothermalEconomicsConfig
from pythermonomics.data.base_well_results import BaseWellResults

from pywellgeo.welltrajectory.trajectory_base import TrajectoryBase
from pywellgeo.welltrajectory.trajectory_factory import TrajectoryFactory


def read_input(config: str) -> Dict[str, Any]:
    """Read .yml file with settings"""

    with open(config, "r") as f:
        settings = yaml.safe_load(f)

    return settings


class Trajectory:
    """
    Trajectory creates a trajectory database for all the wells, which can be based
    on a OPM input deck WXYZ or trajectory input file and detailed constructed well paths
    """

    def __init__(
        self,
        trajectoryfile: str,
        options: GeothermalEconomicsConfig,
        simresults: Optional[BaseWellResults] = None,
        trajectoryinstance: Optional[Dict] = None,
    ) -> None:
        """
        :param geothermal_economics control file (instance of geothermal_economics.py)
        :param trajectoryfile: trajectory file (base settings for the well to inititatie the well path)
        :param trajectoryinstance: object instance whihc can be used to udpate geometrical aspects
              of the well for raster asessment with varying depths, thickness etc

        """
        self.options = options
        self.simresults = simresults

        if trajectoryfile is None:
            format = "BASE"
            # set the format to the base trjectory not needing input file, still file is set to control file
            # only for outputfile identfication purposes
            self.trajectoryfile = None
        else:
            # read the format from the trajectoryfile
            inputs = read_input(trajectoryfile)
            format = inputs["format"]
            self.trajectoryfile = trajectoryfile

        self._trajectoryinput = TrajectoryFactory.getTrajectoryClass(
            options, simresults, format
        )
        self._trajectoryinstance = trajectoryinstance
        self.read()

    @property
    def trajectoryinput(self) -> TrajectoryBase:
        """
          contains the trajectory input object, and functionality to read the trajectory, as well as supplementary information
          can
        :return:
        """
        return self._trajectoryinput

    @property
    def trajectoryinstance(self) -> None:
        """
          contains the trajectory instance object where specific well geometrical aspects can be updated
        :return:
        """
        return self._trajectoryinstance

    def read(self) -> None:
        """
        reads a grid from fname and gridIO object
        """
        trajectoryinput = self.trajectoryinput
        self.tw = trajectoryinput.read(
            trajectoryfile=self.trajectoryfile,
            trajectoryinstance=self.trajectoryinstance,
        )
