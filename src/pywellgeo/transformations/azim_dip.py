import copy
from typing import Union

import numpy as np
from numpy import float64, ndarray
from scipy.spatial.transform import Rotation as R

TINY = 1e-10


def checkTiny(x: float64) -> Union[float, float64]:
    """clip x to +/- TINY

    :param  x parameter to check
    :return cliped x to +/- TINY
    """
    if abs(x) < TINY:
        if x < 0:
            return -TINY
        else:
            return TINY
    return x


class AzimDip:
    """
    AzimDip class, spherical coordinates and its conversion to vector notation
    using the following convention

    - azmimuth = 0 corresponds to positive y-axis
    - azimuth = 90 corresponds to positive x-axis
    - dip >0  corresponds to negative z axis
    """

    def __init__(self, azim: Union[float64, int], dip: Union[float64, int]) -> None:
        """
        instantiate an azim, dip  object
        """

        self.azim = azim
        self.dip = dip

        self.r = R.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    @classmethod
    def copy(self):
        """
         create a deep copy of the object

        :return: deep copy of the object
        """
        x = copy.deepcopy(self)
        return x

    @property
    def azim(self):
        """
        degrees of the azimuth (from 0 to 360, 0 is north)
        """
        return self._azim

    @azim.setter
    def azim(self, azim):
        self._azim = azim

    @property
    def dip(self):
        """
        degrees of the dip (from -90 to 90, 0 is horizontal)
        """
        return self._dip

    @dip.setter
    def dip(self, dip):
        """
        dip

        :param dip: degrees of the dip (from -90 to 90, 0 is horizontal)
        """
        self._dip = dip

    def __str__(self) -> str:
        return str(round(self.azim, 2)) + "/" + str(round(self.dip, 2))

    @classmethod
    def from_vector(cls, dipvec: ndarray) -> "AzimDip":
        """
        instantiate an azim dip vector object from a dip vector

        :param  dipvec 3D vector (np.array)
        """
        n = AzimDip.normalize_vector(dipvec)
        n[2] = checkTiny(n[2])
        if np.abs(1 - n[2] ** 2) < TINY:
            if n[2] > 0:
                n[2] -= TINY
            else:
                n[2] += TINY
        dip = np.degrees(np.arctan(n[2] / np.sqrt(1 - n[2] ** 2))) * -1
        azim = np.degrees(np.arctan2(n[0], n[1]))
        if azim < 0:
            azim += 360
        ad = cls(azim, dip)
        return ad

    def azimdip2Vector(self, n: ndarray = None) -> ndarray:
        """
        convert azimuth and dip spherical coordinates
        to a cartesianvector in the dip direction

        :return  cartesian coordinates of the azimdip directon(x, y, z)
        """
        if n is None:
            n = np.array([0.0, 0.0, 0.0])
        dr = np.radians(self.dip)
        ar = np.radians(self.azim)
        ch = np.cos(dr)

        n[0] = ch * np.sin(ar)
        n[1] = ch * np.cos(ar)
        n[2] = -np.sin(dr)
        return n

    def azimdip2normal(self, n: ndarray = None) -> ndarray:
        """
        convert azimuth and dip spherical coordinates
        to a cartesianvector in the normal direction of the plane dipping in azim and dip

        :return  cartesian coordinates of the normal directon(x, y, z)
        """
        if n is None:
            n = np.array([0.0, 0.0, 0.0])
        p = AzimDip(self.azim, self.dip)
        p.swapnormal()
        dr = np.radians(p.dip - 90)
        ar = np.radians(p.azim)
        ch = np.cos(dr)

        n[0] = ch * np.sin(ar)
        n[1] = ch * np.cos(ar)
        n[2] = -np.sin(dr)
        return n

    def swapnormal(self) -> None:
        """
        swap to the normal of azim,dip plane (azim,dip) -> (azim+180, 90-dip), preserving sign of dip
        """
        self.azim += 180
        if self.azim >= 360:
            self.azim -= 360
        if self.dip >= 0:
            self.dip = 90 + (90 - self.dip)
        else:
            self.dip = -90 + (-90 - self.dip)

    def swapsign(self):
        """
        swap the sign of the dip. (azim,dip) -> (azim+180, -dip)
        """
        self.azim += 180
        if self.azim >= 360:
            self.azim -= 360
        self.dip *= -1

    def plane2normal_ref(self):
        """
        convert azimuth and dip spherical coordinates to its normal or vice versa
        """
        self.dip = 90 - self.dip
        self.azim = self.azim + 180
        if self.dip < 0:
            self.azim += 180
            self.dip = -self.dip
        if self.azim > 360:
            self.azim -= 360

    def plane2normal(self) -> None:
        """
        convert azimuth and dip spherical coordinates to its normal or vice versa, forcing dip>0
        """
        if self.dip < 0:
            self.dip += 90
        elif self.dip > 90:
            self.dip -= 90
        elif self.dip <= 90:
            self.dip = 90 - self.dip
            self.azim += 180
        if self.azim > 360:
            self.azim -= 360

    @classmethod
    def normalize_vector(cls, vec: ndarray) -> ndarray:
        """
        normalize a 3D vector

        :param vec: np array with 3 elements
        :return: normalized vector
        """
        norm = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5
        if norm > TINY:
            vec = vec / norm
        return vec

    @classmethod
    def test(cls):
        """
        test the AzimDip class
        """
        ad = AzimDip(60, 120)
        n = ad.azimdip2Vector()
        print(ad)
        print(n)
        ad = AzimDip.from_vector(n)
        print(ad)
        print(n)
        rm = ad.r.as_matrix()
        print(rm)
        rn = rm.dot(n)
        print(rn)
