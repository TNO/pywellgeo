from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from numpy import float64, int64, ndarray
from pandas.core.frame import DataFrame

from pywellgeo.transformations.azim_dip import AzimDip
from pywellgeo.transformations.coordinate_transformation import CoordinateTransformation
from pywellgeo.well_data.names_constants import Constants
from pywellgeo.well_tree.well_tree_tno import WellTreeTNO


def replace_item(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj


def getKOP_L(bur: float, phi: float64, xoff: float64, zoff: float64) -> Union[
    Tuple[int, int, int, float64, int],
    Tuple[float64, float64, float64, float64, float64],
]:
    """
    Compute KOP and  L specification for well in plane entering vertical at 0,0 and exiting at inclination phi at xoff, zoff

    :param  bur:  build up rate for the well, degrees per 30 m
    :param  phi: target inclination (degrees) of the well at xoff, zoff in range (0,180)
    :param  xoff:   off axis point of continuation of well trajectory (should be positive)
    :param  yoff:   in axis point of continuation of well trajectory (should be positive)
    :param  bur:  buildup rate of the well typically 3 degrees (per 30 m)
    :return:   tuple  A, xb, zb, KOP,  L

    notes
    -----
    - where A is arclength for the buildup,
    - xb, zb are the horizontal and vertical spacing for the Buildup,
    - KOP is Kickoff point and  L Length to reach xoff, zoff (such that xb+L*sin(phi)=xoff and zb+KOP +L*cos(phi)=zoff)
    - total MD is equal to KOP + A + L
    """
    A = (phi / bur) * 30
    R = ((180 / bur) * 30) / np.pi
    phirad = (phi / 180.0) * np.pi
    xb = R * (1 - np.cos(phirad))  # xb is same as d1
    d2 = max(xoff - xb, 0)

    bsin = np.sin(phirad)
    zb = R * bsin  # zb is same as v2
    ok = False
    if abs(bsin) > 1e-10:
        v3 = d2 * np.cos(phirad) / bsin
        KOP = zoff - zb - v3
        L = (v3**2 + d2**2) ** 0.5
        ok = (xb <= xoff + 1e-10) and (KOP > -1e-10)
    else:
        # phi=0 or 180
        KOP = zoff
        L = 0
        ok = abs(xoff) < 1
    if ok:
        return A, xb, zb, KOP, L
    else:
        return -1, -1, -1, KOP, -1


def find_phi_L(dx, dz, bur=3.0):
    """
    Compute KOP, phi and L for  well in plane entering and exiting at same orientation
    (vertical in z direction) to reach  coordinates (0.5*dx,0.5*dz) starting from (0,0)
    phi is chosen as low possible value, based on dx (KOP close to 0)
    it also resurns xb and zb which are the horizontal and vertical spacing for the Buildup,
    and returns the arclength R for the build up

    :param  dx:  off axis shift of continuation of well trajectory (should be positive)
    :param  dz:   in axis shift of continuation of well trajectory (should be positive)
    :param  bur:  buildup rate of the well typically 3 degrees (per 30 m)
    :return tuple: phi, R, d1, v2, KOP, L to reach  coordinates (0.5dx,0.5dz) starting from 0,0
    """
    xoff = dx * 0.5
    zoff = dz * 0.5
    if np.abs(xoff) < 1e-3:
        # in line so phi=0.0
        phi = 0.0
        V, xb, zb, KOP, L = getKOP_L(bur, phi, xoff, zoff)
        return phi, V, xb, zb, KOP, L

    else:
        for iphi in np.arange(0, 180):
            phi = iphi * 1.0
            V, xb, zb, KOP, L = getKOP_L(bur, phi, xoff, zoff)
            if V >= 0:
                return phi, V, xb, zb, KOP, L

    # print('not enough space in plane to connect subparalel entry and exit')
    return -1, -1, -1, -1, -1, -1


def find_phi(
    dx: float64, dz: float64, bur: Optional[float] = 3.0
) -> Tuple[float64, float64, float64, float64, float64, float64]:
    """
    Compute  phi (pitch) such that with KOP is zero
    it also resurns xb and zb which are the horizontal and vertical spacing for the Buildup,
    and returns the arclength R for the build up

    :param  dx:   off axis shift of continuation of well trajectory (should be positive)
    :param  dz:   in axis shift of continuation of well trajectory (should be positive)
    :param  bur:  buildup rate of the well typically 3 degrees (per 30 m)
    :return:  tuple phi, R, d1, v2, KOP, L to reach  coordinates (0.5dx,0.5dz) starting from 0,0
    """
    xoff = dx
    zoff = dz
    if np.abs(xoff) < 1e-3:
        # in line so phi=0.0
        phi = 0.0
        V = 0.0
        xb = 0.0
        zb = 0.0
        L = (dx**2 + dz**2) ** 0.5
        return V, xb, zb, phi, L
    else:
        # first make sure that a phi is possible
        for iphi in np.arange(0, 180):
            phi = iphi * 1.0
            V, xb, zb, KOP, L = getKOP_L(bur, phi, xoff, zoff)
            if V >= 0:
                # in the range phi to phi-1 should be the solution
                philow = phi - 1
                phihigh = phi
                trykop = KOP
                while (abs(trykop) >= 1e-3) and (trykop < 0):
                    phi = 0.5 * (philow + phihigh)
                    V, xb, zb, trykop, L = getKOP_L(bur, phi, xoff, zoff)
                    if abs(trykop) >= 1e-3:
                        if trykop > 0:
                            phihigh = phi
                        else:
                            philow = phi

                return phi, V, xb, zb, trykop, L

    # print('not enough space in plane to connect subparalel entry and exit')
    return -1, -1, -1, -1, -1, -1


class WellTreeAzimDip:
    """
    Well  Tree for well representation with multiple branches with azim dip notation
    the branches are the subsequent well bore subsegments, the representation is in x,y,z
    with curved segments which will be constructed from input yml. They are self explanatory in the sense that
    that consecutive (x,y,z) and (azim,dip) shooting directions are coplanar
    """

    branchcolors = ["black", "orange", "red", "grey"]

    perforationcolor = "blue"

    def __init__(
        self,
        x: Union[float, float64, int, int64],
        y: Union[float, float64, int, int64],
        z: Union[float, float64, int, int64],
        azimdip: AzimDip,
        radius: float,
        BUR: float,
        xroot: Optional[WellTreeAzimDip] = None,
        perforated: Optional[bool] = True,
        color: Optional[str] = "black",
        name: Optional[str] = "main",
        namesub: Optional[str] = "wellhead",
    ) -> None:
        self.xroot = xroot
        if xroot is not None:
            self.xroot.branches.append(self)
        self.x = np.asarray([x, y, z])
        self.azimdip = azimdip
        self.radius = radius
        self.BUR = BUR
        self.perforated = perforated
        self.branches = []
        self.color = color
        self.name = name
        self.namesub = namesub
        self.inclination = 0
        self.xb = np.nan
        self.zb = np.nan
        self.KOP = np.nan
        self.L = np.nan
        self.ahd = np.nan

    def __str__(self):
        str1 = str("x:" + str(self.x) + ", ad:" + str(self.azimdip))
        str2 = str(
            "KOP,inclination, L, ahd, BUR, xb,zb:"
            + str(self.KOP)
            + ","
            + str(self.inclination)
            + ","
            + str(self.L)
            + ","
            + str(self.ahd)
            + ","
            + str(self.BUR)
            + ","
            + str(self.xb)
            + ","
            + str(self.zb)
        )
        return str(str1 + "\n" + str2)

    def printtree(self, name="main"):
        if self.name == name:
            print(self)
        for node in self.branches:
            node.printtree(name=name)

    def find(self, namesubtofind="wellhead"):
        if self.namesub == namesubtofind:
            return self
        else:
            for node in self.branches:
                return node.find(namesubtofind)

    def nameslist(self, names: Optional[List[str]] = None) -> List[str]:
        if names is None:
            names = []
        if self.name not in names:
            names.append(self.name)
        for node in self.branches:
            node.nameslist(names)
        return names

    def getbranchlist(
        self, branchname: str, list: Optional[List[WellTreeAzimDip]] = None
    ) -> List[WellTreeAzimDip]:
        if list is None:
            list = []
        for b in self.branches:
            if b.name == branchname:
                list.append(b)
            list = b.getbranchlist(branchname, list)
        return list

    @classmethod
    def copy(self):
        x = copy.deepcopy(self)
        return x

    @classmethod
    def addBranches(
        cls,
        p: Dict[
            str,
            Dict[
                str,
                Dict[str, Union[str, Dict[str, int], float, Dict[str, Dict[str, int]]]],
            ],
        ],
        well: str,
        sname: Optional[str] = "main",
        nroot: "WellTreeAzimDip" = None,
    ) -> "WellTreeAzimDip":
        main = p[well]["main_wellbore"]
        if sname == "main":
            bmain = main
        else:
            bmain = p[well][sname]

        x = main["wellhead"]["x"] * 1.0
        y = main["wellhead"]["y"] * 1.0
        z = 0.0
        BUR = main["BUR"]
        radius = main["radius"]
        color = WellTreeAzimDip.branchcolors[0]

        if sname == "main":
            # new tree for the well
            azimdip = AzimDip(0, 90)
            nroot = cls(
                x, y, z, azimdip, radius, BUR, perforated=False, color=color, name=sname
            )
            nlast = nroot
        else:
            # find the node in the tree to attach the branch to
            nlast = nroot.find(bmain["startsub"])
            x = nlast.x[0]
            y = nlast.x[1]
            z = nlast.x[2]

        # add subbranches

        subbranches = bmain["subs"]
        for sub in subbranches:
            namesub = sub
            branch = subbranches[sub]
            existL, L = WellTreeAzimDip.check_inKeys(branch, "L")
            if existL:
                # create a new branch shooting from existing x,y,z  and azimdip to new x,y,z
                nlast = WellTreeAzimDip.fromL(
                    nlast, L, radius, BUR, sname=sname, namesub=namesub
                )
            else:
                existDX, dx = WellTreeAzimDip.check_inKeys(branch, "dx", None)
                existDY, dy = WellTreeAzimDip.check_inKeys(branch, "dy", None)
                existDZ, dz = WellTreeAzimDip.check_inKeys(branch, "dz", None)
                if existDX:
                    xnew = nlast.x[0] + dx
                else:
                    existX, xnew = WellTreeAzimDip.check_inKeys(branch, "x", x)
                if existDY:
                    ynew = nlast.x[1] + dy
                else:
                    existY, ynew = WellTreeAzimDip.check_inKeys(branch, "y", y)
                if existDZ:
                    znew = nlast.x[2] + dz
                else:
                    existZ, znew = WellTreeAzimDip.check_inKeys(branch, "z", z)

                existAzim, azimnew = WellTreeAzimDip.check_inKeys(branch, "azim")
                existDip, dipnew = WellTreeAzimDip.check_inKeys(branch, "dip")
                existSamedir, samedir = WellTreeAzimDip.check_inKeys(branch, "samedir")
                if (existAzim) and (existDip):
                    # azim and dip specified, slide 4 solution  construct a plane in azim/dip and additional node
                    # first check  if
                    nlast = WellTreeAzimDip.fromAzimDipFromtargetXYZ(
                        nlast,
                        xnew,
                        ynew,
                        znew,
                        azimnew,
                        dipnew,
                        radius,
                        BUR,
                        sname=sname,
                        namesub=namesub,
                    )
                elif (existDip) or (existAzim):
                    # if azim or dip  is not specified it will use the new x,y,z and the old azim,dip to construct the plane to reach x,y,z
                    # this is the slide 1 or 2 solution, it will find the azim from x,y,z if dip is given, else
                    nlast = WellTreeAzimDip.fromAzimDipFromtargetXYZ(
                        nlast,
                        xnew,
                        ynew,
                        znew,
                        azimnew,
                        dipnew,
                        radius,
                        BUR,
                        sname=sname,
                        namesub=namesub,
                    )
                else:
                    if existSamedir and samedir is True:
                        namesuba = namesub + "a"
                        # get midpoint
                        xmid = (
                            0.5 * (np.asarray([xnew, ynew, znew]) - nlast.x) + nlast.x
                        )
                        # and trajectory with KOP=0
                        nlast = WellTreeAzimDip.fromtargetXYZ(
                            nlast,
                            xmid[0],
                            xmid[1],
                            xmid[2],
                            radius,
                            BUR,
                            sname=sname,
                            namesub=namesuba,
                        )
                        # now get complementary branch
                        namesubb = namesub + "b"
                        # convert the (90-nlast.inclination) to global
                        t, plane = nlast.getCoordinateTransformation()
                        adlocal = AzimDip(90 - nlast.inclination, 0)
                        adglobal = t.transform2global_orientation(adlocal)
                        adglobal.swapnormal()
                        nnew = cls(
                            xnew,
                            ynew,
                            znew,
                            adglobal,
                            radius,
                            BUR,
                            xroot=nlast,
                            name=sname,
                            namesub=namesubb,
                        )
                        nnew.KOP = nlast.L
                        nnew.L = 0
                        nnew.inclination = nlast.inclination
                        nnew.xb = nlast.xb
                        nnew.zb = nlast.zb
                        adlocal = AzimDip(90 - nnew.inclination, 0)
                        t, plane = nnew.getCoordinateTransformation()
                        nnew.azimdip = t.transform2global_orientation(adlocal)
                        nlast = nnew
                    else:
                        nlast = WellTreeAzimDip.fromtargetXYZ(
                            nlast,
                            xnew,
                            ynew,
                            znew,
                            radius,
                            BUR,
                            sname=sname,
                            namesub=namesub,
                        )
                    # none specified, try to find the shortest path (with minimum BU, inclination)

            x = nlast.x[0]
            y = nlast.x[1]
            z = nlast.x[2]
        return nroot

    @classmethod
    def from_input_trajectory(
        cls,
        input_trajectory: Dict[
            str,
            Dict[
                str,
                Dict[str, Union[str, Dict[str, int], float, Dict[str, Dict[str, int]]]],
            ],
        ],
        wellname: str,
    ) -> "WellTreeAzimDip":
        """compute tree branches from input yaml
        :param input_trajectory dictionary (yaml section of well_trajectories)
        :param wellname  wellname to consider
        """

        p = input_trajectory
        # find well
        for well in p.keys():
            if well == wellname:
                # construct main branch of the well
                nroot = WellTreeAzimDip.addBranches(p, well, sname="main")
                branches = list(p[well].keys())
                branches.remove("main_wellbore")
                for sub in branches:
                    nroot = WellTreeAzimDip.addBranches(p, well, sname=sub, nroot=nroot)

        return nroot

    @classmethod
    def fromL(
        cls,
        nlast: "WellTreeAzimDip",
        L: int,
        radius: float,
        BUR: float,
        sname: Optional[str] = "main",
        namesub: Optional[str] = "sub",
    ) -> "WellTreeAzimDip":
        vL = nlast.azimdip.azimdip2Vector() * L
        nlast = cls(
            nlast.x[0] + vL[0],
            nlast.x[1] + vL[1],
            nlast.x[2] + vL[2],
            nlast.azimdip,
            radius,
            BUR,
            xroot=nlast,
            name=sname,
            namesub=namesub,
        )
        nlast.L = L
        nlast.KOP = 0
        nlast.inclination = 0
        return nlast

    @classmethod
    def fromtargetXYZ(
        cls,
        nlast: "WellTreeAzimDip",
        xnew: Union[int, int64],
        ynew: Union[int, int64],
        znew: Union[int, int64],
        radius: float,
        BUR: float,
        sname: Optional[str] = "main",
        namesub: Optional[str] = "sub",
    ) -> "WellTreeAzimDip":
        """
        get the plane normal from the self, and KOP to reach xnew, ynew,znew,
        from the end point to reach
        :param nlast  parent node
        :param xnew x-coordinate of point to reach
        :param ynew y-coordinate of point to reach
        :param znew z-coordinate of point to reach
        :param radius radius of the well segment
        :param BUR   build up rate of the segment
        :param sname name of the branch
        :param namesub name of the segment in the branch
        :returns WellTreeAzimDip node reprsenting the path
        """
        xn = np.asarray([xnew, ynew, znew])
        # initiate with azimdip of latest well path. Get this from nlast
        # t, plane = nlast.getCoordinateTransformation()
        # adlocal = AzimDip (-90+nlast.inclination, 0 )
        # adglobal =  t.transform2global_orientation(adlocal)
        adglobal = AzimDip(0, 0)
        nnew = cls(
            xnew,
            ynew,
            znew,
            adglobal,
            radius,
            BUR,
            xroot=nlast,
            name=sname,
            namesub=namesub,
        )
        t, plane = nnew.getCoordinateTransformation()

        xlocal = t.transform2local(xn)
        nnew.inclination, nnew.A, nnew.xb, nnew.zb, nnew.KOP, nnew.L = find_phi(
            xlocal[1], xlocal[0], BUR
        )

        adlocal = AzimDip(90 - nnew.inclination, 0)
        nnew.azimdip = t.transform2global_orientation(adlocal)
        print(
            "finding KOP for plane, adinclination, adlocal, inclincation, xoff, zoff ",
            plane,
            nnew.azimdip,
            adlocal,
            nnew.inclination,
            xlocal[1],
            xlocal[0],
        )

        return nnew

    @classmethod
    def gettangent(
        cls, t2: CoordinateTransformation, xint: ndarray, radius: float
    ) -> Tuple[ndarray, AzimDip, float64]:
        """
        this methods
        - assumes a local coordinate system with x-axis in the direction of ad (dip direction)
        and y along the strike, stored in t2
        - xint is located in that plane
        - it finds the line tangent to the circle either centered at (0,-radius) or (0,radius) in the
        local coordinate system of the plane dipping in ad, from xint touching the circle.
        It is oriented such that shooting from xint,
        the subsequent arc segment results in the  orientation of ad
        - the tangent location and the abslute azimip of the tangent  (adtan) at that location is given,
        - and the inclination to be used of the plane direction relative to adtan

        :param radius: radius of the circle
        :return: tuple  xtan , adtan, inclination

        """

        xsp = t2.transform2local(xint)
        s = np.sign(xsp[1])
        xr = np.asarray([0, s * radius, 0])
        xdif = xsp - xr
        L = np.dot(xdif, xdif) ** 0.5
        argsin = radius / L
        if argsin > 1:
            # try other sign
            s = -s
            xr = np.asarray([0, s * radius, 0])
            xdif = xsp - xr
            L = np.dot(xdif, xdif) ** 0.5
            argsin = radius / L
        if argsin > 1:
            print(
                "getttangent, cannot find tangent line, because intersection is inside BU circles"
            )
            return -1, None, -1
        else:
            phi = np.degrees(np.arcsin(radius / L))
            adtanp = AzimDip.from_vector(xdif)
            adtanp.azim += -s * (90 - phi)
            xtanp = adtanp.azimdip2Vector() * radius
            xtan = t2.transform2global(xtanp + xr)
            xdif2 = xtan - xint
            adtan = AzimDip.from_vector(xdif2)

            # now work on the inclication
            # add -s*90 for the inclination as we are after the tangent
            adtanp.azim += -s * 90

            if adtanp.azim > 360:
                adtanp.azim -= 360
            if adtanp.azim < 0:
                adtanp.azim += 360
            inclination = -s * (90 - adtanp.azim)
            # now get global orientation of azim and dip of the tangent

            return xtan, adtan, inclination

    @classmethod
    def fromAzimDipFromtargetXYZ(
        cls,
        nlast: "WellTreeAzimDip",
        xnew: int,
        ynew: int,
        znew: int,
        azimnew: Optional[int],
        dipnew: int,
        radius: float,
        BUR: float,
        sname: Optional[str] = "main",
        namesub: Optional[str] = "sub",
    ) -> "WellTreeAzimDip":
        """
        get the plane normal, and KOP to reach xnew, ynew,znew,
        and either azimnew or dipnew
        :param nlast  parent node
        :param xnew x-coordinate of point to reach
        :param ynew y-coordinate of point to reach
        :param znew z-coordinate of point to reach
        :param azimnew azimuth (strike) for the wellbore to reach
        :param dipnew dip for the well bore to reach at x,y,a location
        :returns WellTreeAzimDip node reprsenting the path
        """
        xn = np.asarray([xnew, ynew, znew])
        # initiate
        adglobal = AzimDip(0, 0)
        nnew = cls(
            xnew,
            ynew,
            znew,
            adglobal,
            radius,
            BUR,
            xroot=nlast,
            name=sname,
            namesub=namesub,
        )
        # get plane and coordinate transformation from the new coordinates and last azimdip (xroot)
        t, plane = nnew.getCoordinateTransformation()

        adlocal = t.transform2local_orientation(nlast.azimdip)

        # get the phi in the rotated plane based on the global azim and or dip
        # azim = plane.findAzim(dip)
        if azimnew is not None:
            if dipnew is None:
                dipnew = WellTreeAzimDip.findDip(plane, azimnew)
        else:
            if azimnew is None:
                azimnew = WellTreeAzimDip.findAzim(plane, dipnew)

        # is the point to reach in the x,y plane, y is
        xlocal = t.transform2local(xn)
        adglobal = AzimDip(azimnew, dipnew)
        adlocal = t.transform2local_orientation(adglobal)

        if abs(adlocal.dip) > 1e-1:
            # the target orientation is in a different plane than the plane from the x,y,z target and the azimdip of xroot
            # therefore get the intersection of the plane shooting from the xroot plane at xroot.x  and the AzimDip(azimnew,dipnew) at xnew.x
            print("not supported")
            # the way to this is first calculate the intersection point of the x.root, and azimdip line with the new plane (AzimDip(azimnew,dipnew) at xnew.x)
            # nnew.CoordinateTransformation
            plane = AzimDip(azimnew, dipnew)
            t2 = CoordinateTransformation(plane, origin=xn)
            xintshoot = t2.line_plane_intersect(nlast.azimdip, nlast.x)
            radius = ((180.0 / BUR) * 30.0) / np.pi
            xint, ad_tan, inclination = WellTreeAzimDip.gettangent(
                t2, xintshoot, radius
            )

            nnew.namesub = namesub + "a"
            nnew.x = xint
            xlocal = t.transform2local(xint)
            adglobal = ad_tan
            adlocal = t.transform2local_orientation(adglobal)
            nnew.azimdip = adglobal
            nnew.inclination = 90 - adlocal.azim
            if nnew.inclination < 0:
                nnew.inclination += 360
            plane2 = AzimDip(ad_tan.azim - 90, 90)
            print(
                "finding KOP for plane, adinclination, adlocal, inclincation, xoff, zoff ",
                plane2,
                nnew.azimdip,
                adlocal,
                nnew.inclination,
                xlocal[1],
                xlocal[0],
            )
            nnew.A, nnew.xb, nnew.zb, nnew.KOP, nnew.L = getKOP_L(
                BUR, nnew.inclination, xlocal[1], xlocal[0]
            )
            if nnew.A < 0:
                print("ERROR failed to find KOP/L for node ", nnew.x)
            nlast = nnew
            nnew = WellTreeAzimDip.fromAzimDipFromtargetXYZ(
                nlast, xnew, ynew, znew, azimnew, dipnew, radius, BUR, sname, namesub
            )

            """
            t2 = CoordinateTransformation(plane, origin=xint)
            xlocal = t2.transform2local(xn)
            namesubb = namesub + 'b'
            nnew = cls(xnew, ynew, znew, plane, radius, BUR, xroot=nnew, name=sname, namesub=namesubb)
            nnew.inclination = inclination
            if nnew.inclination < 0:
                nnew.inclination += 360
            print('finding KOP for plane, adinclination, adlocal, inclincation, xoff, zoff ', plane, nnew.azimdip,
                  adlocal, nnew.inclination, xlocal[1], xlocal[0])
            nnew.A, nnew.xb, nnew.zb, nnew.KOP, nnew.L = getKOP_L(BUR, nnew.inclination, xlocal[1], xlocal[0])
            """
        else:
            nnew.azimdip = adglobal
            nnew.inclination = 90 - adlocal.azim
            if nnew.inclination < 0:
                nnew.inclination += 360
            print(
                "finding KOP for plane, adinclination, adlocal, inclincation, xoff, zoff ",
                plane,
                nnew.azimdip,
                adlocal,
                nnew.inclination,
                xlocal[1],
                xlocal[0],
            )
            nnew.A, nnew.xb, nnew.zb, nnew.KOP, nnew.L = getKOP_L(
                BUR, nnew.inclination, xlocal[1], xlocal[0]
            )
        return nnew

    @classmethod
    def findAzim(self, plane: AzimDip, dip: int) -> float64:
        """
        find azimuth with the correct dip. It only considers a positive y value in the local coordinate system

        :param plane: plane containing the entry and exit of the path
        :param dip: target dip
        :return: azim azimuth of the exit trajectory
        """

        # check if the target dip can be reached on the plane
        if abs(plane.dip) < abs(dip):
            print("ERROR: findAzim in WellTreeAzimDip ", plane, " < target dip ", dip)
            return plane.azim
        else:
            rdip = np.radians(dip)
            sindip = np.sin(rdip)
            frac = sindip / np.sin(np.radians(plane.dip))
            phi = np.degrees(np.arccos(frac))
            t = CoordinateTransformation(plane, pitch=0)
            ad1 = AzimDip((90 - phi), 0)
            adg1 = t.transform2global_orientation(ad1)
            return adg1.azim

    @classmethod
    def findDip(self, tplane: AzimDip, azim):
        """
        find the dip at the specified direction of the azimuth of the well bore at the exit point in the plane
        starting from the entry point and azimdip direction

        :param tplane: plane containing the entry and exit of the path
        :param azim: azimuth direction to find dip for the exit
        :return: dip of vector on the tplane  which intersects with vertical plane at azim and of which innerproduct
          with the azim strike is positive
        """

        plane = AzimDip(tplane.azim, tplane.dip)
        plane.plane2normal()
        n1 = plane.azimdip2Vector()

        p = AzimDip(azim + 90, 90)
        p.plane2normal()
        n2 = p.azimdip2Vector()

        # find intersection
        vint = np.cross(n1, n2)

        # check for a positive dot/product to
        ad0 = AzimDip(azim, 0)
        vwell = ad0.azimdip2Vector()

        if vwell.dot(vint) < 0:
            vint = -vint
        adint = AzimDip.from_vector(vint)
        return adint.dip

    @classmethod
    def findInterAzimDipPoint(self, tplane: AzimDip, azim):
        return

    @classmethod
    def check_inKeys(
        cls,
        dict: Dict[str, int],
        key: str,
        oldval: Optional[Union[float, int64]] = None,
    ) -> Union[Tuple[bool, None], Tuple[bool, int]]:
        if key in dict.keys():
            return True, dict[key]
        else:
            return False, oldval

    @classmethod
    def read_input(cls, config: str) -> Dict[
        str,
        Union[
            str,
            Dict[str, Dict[str, Union[float, int]]],
            Dict[
                str,
                Dict[
                    str,
                    Dict[
                        str,
                        Union[str, Dict[str, int], float, Dict[str, Dict[str, int]]],
                    ],
                ],
            ],
        ],
    ]:
        """Read .yml file with settings"""
        with open(config, "r") as f:
            settings = yaml.safe_load(f)
        return settings

    @classmethod
    def substitute_input(cls, inputs, subdict):
        """substitute items in subdict in inputs, not functioning at present"""
        # return replace_item(inputs, subdict)

    @classmethod
    def process_inputs(cls, inputname: str, wellname: str) -> "WellTreeAzimDip":
        """Process all input files"""
        inputs = WellTreeAzimDip.read_input(inputname)
        nroot = WellTreeAzimDip.from_input_trajectory(
            inputs["well_trajectories"], wellname
        )
        nroot.init_ahd()
        return nroot

    def getCoordinateTransformation(
        self, doprint: Optional[bool] = False
    ) -> Tuple[CoordinateTransformation, AzimDip]:
        """ "
        defines coordinate transformation in such a way that
        - the local x axis is positive in the direction of the
        previous azim/dip orientation,
        - the local x,y plane is spanned by  the vector   (self.x-xroot.x) and the azim/dip vector
        - the positive y aligns with the vector (self.x-xroot.x)
        :returns CoordinateTransformation object and the x,y plane in azim dip notation
        """
        orx = self.xroot.x
        vx = self.xroot.azimdip.azimdip2Vector()
        v2 = self.x - orx
        plane, pitch = CoordinateTransformation.plane_pitch_from_vectors(vx, v2)
        t = CoordinateTransformation(plane, origin=orx, pitch=-(90 - pitch))
        # check if v2 is projected with postive y otherwise, "invert" the plane, 180 and -dip
        #
        adlocal90 = t.transform2local_orientation(self.xroot.azimdip)
        if doprint:
            print("adlocal", adlocal90)
        v2local = t.transform2local(self.x)
        if v2local[1] < 0:
            plane.swapnormal()
            t = CoordinateTransformation(plane, origin=orx, pitch=90 - pitch)
            v2local = t.transform2local(self.x)
            adlocal90 = t.transform2local_orientation(self.xroot.azimdip)
        if doprint:
            print("adlocal", adlocal90)
        return t, plane

    def init_ahd(self) -> None:
        if self.xroot is None:
            self.ahd = 0
        else:
            self.ahd = self.xroot.ahd + self.getAHDincrement()
        for b in self.branches:
            b.init_ahd()

    def getLocalCoordinates(
        self,
        md: float64,
        blist: List[WellTreeAzimDip],
        tlist: List[CoordinateTransformation],
    ) -> Tuple[int, ndarray, AzimDip, ndarray, AzimDip]:
        """ "
        based on measured depth md (along hole depth)
        """
        ib = 0
        for b in blist:
            if (md >= b.xroot.ahd) and (md <= b.ahd):
                A = 30.0 * b.inclination / b.BUR
                R = ((180 / b.BUR) * 30) / np.pi
                mdlocal = md - b.xroot.ahd
                t = tlist[ib]
                if mdlocal < b.KOP:
                    # pre-buildup
                    vlocal = np.asarray([mdlocal, 0, 0])
                    adlocal = AzimDip(90, 0)
                elif (mdlocal - b.KOP) < A:
                    # build up part
                    inclination = (mdlocal - b.KOP) * b.BUR / 30.0
                    adlocal = AzimDip(90 - inclination, 0)
                    dx = np.sin(np.radians(inclination)) * R
                    dy = (1 - np.cos(np.radians(inclination))) * R
                    vlocal = np.asarray([b.KOP + dx, dy, 0])
                else:
                    # post build-up
                    adlocal = AzimDip(90 - b.inclination, 0)
                    inclinationrad = np.radians(b.inclination)
                    dx = np.sin(inclinationrad) * R
                    dy = (1 - np.cos(inclinationrad)) * R
                    _len = mdlocal - b.KOP - A
                    dxl = np.cos(inclinationrad) * _len
                    dyl = np.sin(inclinationrad) * _len
                    vlocal = np.asarray([b.KOP + dx + dxl, dy + dyl, 0])
                vglobal = t.transform2global(vlocal)
                adglobal = t.transform2global_orientation(adlocal)
                return ib, vlocal, adlocal, vglobal, adglobal
            else:
                ib += 1

        return -1, None, None, None, None

    def compute_trajectories(
        self,
        wellname: str,
        trajectories: Optional[Dict[str, Dict[str, DataFrame]]] = None,
        step_depth: int = 10,
        savecsv: Optional[bool] = False,
    ) -> Dict[str, Dict[str, DataFrame]]:
        # main wellbore
        if trajectories is None:
            trajectories = {}

        # along hole measured depth array for main branch at step_depth intervals
        w = wellname
        trajectories[w] = {}
        # bnames should iterate over the available branches
        branches = self.nameslist()
        for bname in branches:
            blist = self.getbranchlist(bname)
            tlist = []
            for b in blist:
                t, plane = b.getCoordinateTransformation()
                tlist.append(t)

            ahdmax = blist[-1].ahd
            ahdmin = blist[0].xroot.ahd
            # for other branches ahdmin is different>0
            drange = np.array(range(int(ahdmin), int(ahdmax + step_depth), step_depth), dtype=float)
            wt = pd.DataFrame()
            wt["depth"] = drange
            orx = blist[0].xroot.x

            # shift the well points to fit the ahd of the segments
            ib = 0
            wt.loc[0, "depth"] = ahdmin
            for i in wt.index:
                ahd = wt.loc[i, "depth"]
                if ahd < ahdmin:
                    wt.loc[i, "depth"] = ahdmin
                ahdmax = blist[ib].ahd
                if ahd > ahdmax:
                    wt.loc[i, "depth"] = ahdmax
                    if ib < np.size(blist):
                        ib += 1

            for i in wt.index:
                ahd = wt.loc[i, "depth"]
                ib, vlocal, adlocal, vglobal, adglobal = self.getLocalCoordinates(
                    ahd, blist, tlist
                )
                try:
                    wt.loc[i, "inclination"] = 90 - adglobal.dip
                except AttributeError:
                    print ("AttributeError in compute_trajectories, adglobal", adglobal)
                    exit()
                wt.loc[i, "azimuth"] = adglobal.azim
                wt.loc[i, "dx"] = vglobal[0] - orx[0]
                wt.loc[i, "dy"] = vglobal[1] - orx[1]
                if bname == "main":
                    wt.loc[i, "x"] = vglobal[0]
                    wt.loc[i, "y"] = vglobal[1]
                    wt.loc[i, "TVD"] = vglobal[2]
                else:
                    wt.loc[i, bname + "_" + "x"] = vglobal[0]
                    wt.loc[i, bname + "_" + "y"] = vglobal[1]
                    wt.loc[i, bname + "_" + "TVD"] = vglobal[2]
            trajectories[w][bname] = wt

        if savecsv:
            dfs = []
            dfsnames = self.nameslist()
            for branch in dfsnames:
                df = pd.DataFrame(trajectories[w][branch])
                dfs.append(df)

            dfc = pd.concat(dfs, keys=dfsnames)

            # Make sure directory exists before writing
            if not os.path.exists("output_welltrajectories"):
                os.makedirs("output_welltrajectories")
            with open("output_welltrajectories/trajectory_%s.csv" % w, "w") as f:
                dfc.to_csv(f, lineterminator="\n", index=False)

        return trajectories

    def getAHDincrement(self) -> Union[float, float64]:
        return self.KOP + (self.inclination / self.BUR) * 30 + self.L

    @classmethod
    def test1(cls):
        x = 100
        y = 100
        azimdip = AzimDip(180, 90)
        radius = 0.08
        BUR = 3.0
        nroot = cls(x, y, 0, azimdip, radius, BUR)
        nlast = nroot
        xnew = x - 800
        ynew = y - 200
        znew = -2000
        azimnew = None
        dipnew = 30
        BUR = 3.0
        nlast = WellTreeAzimDip.fromAzimDipFromtargetXYZ(
            nlast, xnew, ynew, znew, azimnew, dipnew, radius, BUR
        )
        azimnew = 130
        azimnew = None
        dipnew = -20
        ynew = ynew - 1500
        xnew = xnew - 1600
        znew = znew + 200

        nlast = WellTreeAzimDip.fromAzimDipFromtargetXYZ(
            nlast, xnew, ynew, znew, azimnew, dipnew, radius, BUR
        )

        xnew = xnew - 1000
        ynew = ynew + 300
        znew = -800
        azimnew = 0
        nlast = WellTreeAzimDip.fromAzimDipFromtargetXYZ(
            nlast, xnew, ynew, znew, azimnew, dipnew, radius, BUR
        )
        xnew = xnew + 1800
        ynew = ynew + 600
        znew = -800
        azimnew = 110
        nlast = WellTreeAzimDip.fromAzimDipFromtargetXYZ(
            nlast, xnew, ynew, znew, azimnew, dipnew, radius, BUR
        )

        xnew = -300
        ynew = -300
        znew = -800
        azimnew = 45
        nlast = WellTreeAzimDip.fromAzimDipFromtargetXYZ(
            nlast, xnew, ynew, znew, azimnew, dipnew, radius, BUR
        )

        xnew = 100
        ynew = 100
        znew = -0
        azimnew = None
        dipnew = -90
        nlast = WellTreeAzimDip.fromAzimDipFromtargetXYZ(
            nlast, xnew, ynew, znew, azimnew, dipnew, radius, BUR
        )
        nroot.init_ahd()

        branchname = "main"
        nroot.printtree(name=branchname)
        trj = nroot.compute_trajectories("well1")

        print(trj)

        well1 = WellTreeTNO.from_trajectories(trj["well1"], 0.1)
        fig, ax = well1.plotTree(doplot=True)

    @classmethod
    def get_survey_EL(cls, trj, wname1, wname2, bname="main"):
        """
        save the main branchs of 'well1' and 'well2' trajectory with name sbranch to survey points file
        :param trj : dictionary with well (branch) surveys
        :param wname1  first well to start survey data
        :param wname2 second well to extend survey data in reverse orde (the endpoint of wname1 and wname2 are expected correspond)
        :param bname branch name of the connected laterals
        :return: survey as dataframe
        """
        df1 = pd.DataFrame(trj[wname1][bname])
        df2 = pd.DataFrame(trj[wname2][bname])
        dfa = df2.sort_values("depth", ascending=False)
        depthcor = dfa.iloc[0]["depth"]
        dfa["depth"] = depthcor - dfa["depth"]
        dfa["depth"] = dfa["depth"] + df1.iloc[-1]["depth"]

        df = pd.concat([df1, dfa])
        return df

    @classmethod
    def test_inputfile(cls, filename, saveEL=False):
        filename_noext = str(os.path.basename(filename)).split(".")[0]

        # get a dict on the yml to get the wells
        trajectory_yml = WellTreeAzimDip.read_input(filename)
        tw = trajectory_yml["well_trajectories"]
        trj = None
        for i, w in enumerate(tw):
            nroot = WellTreeAzimDip.process_inputs(filename, w)
            trj = nroot.compute_trajectories(
                w, trajectories=trj, savecsv=True, step_depth=30
            )
            welltree = WellTreeTNO.from_trajectories(
                trj[w], tw[w]["main_wellbore"]["radius"]
            )
            tw[w][Constants.WELLTREE] = welltree

        wellnames = []
        for w in tw.keys():
            wellnames.append(w)

        wellname1 = wellnames[0]
        wellname2 = wellnames[1]

        well1 = WellTreeTNO.from_trajectories(trj[wellname1], 0.1)
        well2 = WellTreeTNO.from_trajectories(trj[wellname2], 0.1)

        minx, maxx = well1.minmax()
        print("min,max coordinates", minx, maxx)
        if saveEL:
            dfsurvey = WellTreeAzimDip.get_survey_EL(
                trj, wname1=wellname2, wname2=wellname1, bname="main"
            )
            with open("output/" + filename_noext + "_EL_connected.csv", "w") as f:
                dfsurvey.to_csv(f, lineterminator="\n", index=True)

        well1.coarsen(segmentlength=30, perforated=None)
        well2.coarsen(segmentlength=30, perforated=None)
        df = well1.getBranchSurvey("branch1", includemain=True, perforated=None)
        with open("output/" + filename_noext + "_survey.csv", "w") as f:
            df.to_csv(f, lineterminator="\n", index=True)

        fig, ax = well1.plotTree3D(doplot=False)
        well2.plotTree3D(
            fig, ax, tofile="output/" + filename_noext + "_welltree_3D.png"
        )

        axes = [0, 1, 2]
        axes_s = ["xy", "xz", "yz"]
        for axi in axes:
            fig, ax = well1.plotTree(axis=axi)
            well2.plotTree(fig=fig, ax=ax, axis=axi)
            plt.savefig("output/" + filename_noext + "_welltree_%s.png" % axes_s[axi])
