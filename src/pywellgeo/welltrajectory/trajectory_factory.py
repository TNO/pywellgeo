from typing import Optional

from pythermonomics.config.geothermal_economics_config import GeothermalEconomicsConfig
from pythermonomics.data.base_well_results import BaseWellResults

from pywellgeo.welltrajectory.trajectory_base import TrajectoryBase
from pywellgeo.welltrajectory.trajectory_dc1d import TrajectoryDc1d
from pywellgeo.welltrajectory.trajectory_detailed_tno import TrajectoryDetailedTNO
from pywellgeo.welltrajectory.trajectory_xyz_generic import TrajectoryXyzGeneric


class TrajectoryFactory:
    """
    This  class is a factory for IO objects

    static Methods:

         getTrajectoryClass (format: str) : gets an instance of the apppropriate Trajectory object
    """

    @staticmethod
    def getTrajectoryClass(
        options: GeothermalEconomicsConfig,
        simresults: BaseWellResults,
        format: Optional[str] = "BASE",
    ) -> TrajectoryBase:
        """Factory Method"""
        builder = {
            "BASE": TrajectoryBase,
            "DETAILEDTNO": TrajectoryDetailedTNO,
            "DC1DWELL": TrajectoryDc1d,
            "XYZGENERIC": TrajectoryXyzGeneric,
        }

        cls = builder.get(format)
        if cls is None:
            raise ValueError(f"Unknown trajectory format: {format}")

        if cls == TrajectoryDc1d:
            # Doesn't use simulation results, it builds its own WXYZ inside the method
            return cls(options)
        else:
            return cls(options, simresults)
