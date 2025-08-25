"""
WellTreeTNO
===========

This module contains the `WellTreeTNO` class, which is used to represent a well with multiple branches.

"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import bool, float64, ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series


class WellTreeTNO:
    branchcolors = ["black", "orange", "red", "grey"]

    perforationcolor = "blue"

    def __init__(
        self,
        x: float64,
        y: float64,
        z: float64,
        radius: float,
        xroot: Optional["WellTreeTNO"] = None,
        perforated: Optional[bool] = True,
        color: Optional[str] = "black",
        name: Optional[str] = "main",
    ) -> None:
        """
        create a wellTree object, as a multiple linked list (allowing for multiple branches)

        :param x: np array (1d)  x coordinates
        :param y: nd array (1d)  y coordinates
        :param z: nd array (1d)  z coordinates
        :param radius: radius of the well
        :param xroot: connecting wellTree node (default is None
        :param perforated:  perforation status (default is True)
        :param color:   color of the wellTree node (default is black)
        :param name: name of the wellTree node (default is 'main')
        """
        self.xroot = xroot
        if xroot is not None:
            self.xroot.branches.append(self)
        self.x = np.asarray([x, y, z])
        self.radius = radius
        self.perforated = perforated
        self.branches = []
        self.color = color
        self.name = name
        self.ref = None

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        copied = WellTreeTNO(
            self.x[0],
            self.x[1],
            self.x[2],
            self.radius,
            xroot=None,  # Avoid copying xroot to prevent cycles
            perforated=self.perforated,
            color=self.color,
            name=self.name,
        )
        memo[id(self)] = copied
        copied.branches = [copy.deepcopy(b, memo) for b in self.branches]
        for branch in copied.branches:
            branch.xroot = copied
        return copied

    @classmethod
    def copy(self):
        """
        create a deep copy of the current instance
        :return: copy of the current instance
        """
        x = copy.deepcopy(self)
        return x

    @classmethod
    def from_xyz(
        cls,
        x: ndarray,
        y: ndarray,
        z: ndarray,
        radius: Optional[float] = 0.10795,
        sname: Optional[str] = "main",
        nroot: Optional["WellTreeTNO"] = None,
    ) -> "WellTreeTNO":
        """
        create a wellTree object from x,y,z coordinates

        :param x: np array (1d) with x coordinates
        :param y: np array (1d) with y coordinates
        :param z: np array (1d) with z coordinates
        :param radius: radius of the well
        :return: wellTree object, name is 'main'
        """
        color = "black"
        color = WellTreeTNO.branchcolors[0]
        for i, xval in enumerate(x):
            if nroot is None:
                nroot = cls(
                    x[i], y[i], z[i], radius, perforated=False, color=color, name=sname
                )
                nlast = nroot
            else:
                nlast = cls(
                    x[i],
                    y[i],
                    z[i],
                    radius,
                    xroot=nlast,
                    perforated=False,
                    color=color,
                    name=sname,
                )
        return nroot

    def add_xyz(
        self,
        x: ndarray,
        y: ndarray,
        z: ndarray,
        sbranch: Optional[str] = "branch",
        color: Optional[str] = "black",
        radius: Optional[float] = 0.10795,
    ) -> None:
        """
        add a branch to the well tree from x,y,z coordinates, it will insert the branch at the last point where the branch
        connects to the existing wellTree

        :param x: np array (1d) with x coordinates
        :param y: np array (1d) with y coordinates
        :param z: np array (1d) with z coordinates
        :param sbranch: name of the new branch
        :param color: color of the new branch
        :param radius: radius of the new branch

        """
        n0, xb, yb, zb = self.findlastsame(x, y, z)
        nlast = n0
        if n0 is not None:
            for inode, xval in enumerate(xb):
                if inode > 0:
                    nlast = WellTreeTNO(
                        xb[inode],
                        yb[inode],
                        zb[inode],
                        radius,
                        xroot=nlast,
                        perforated=False,
                        color=color,
                        name=sbranch,
                    )
        else:
            print(
                "connecting trajectory point in main cannot be found for first point in ",
                sbranch,
            )

    @classmethod
    def from_trajectories(
        cls, trajectory: Dict[str, DataFrame], radius: float
    ) -> "WellTreeTNO":
        """
        obtain wellTree object from  'main' and trajectories 'branch1',  'branch2', etc from the well stored in dictionary trajectory
        from trajectory[branchname] and columns for 'x','y', 'TVD' (if branchname is 'main') else
        columns for branchname + '_x', branchname + '_y', branchname + '_TVD'. It will automatically remove
        any duplicate segements from branches above the kickoff point

        trajectory is created by the trajectories.py class from yml input

        :param trajectory:
        :param branchname:
        :param includebranchname:
        :return: wellTree for the well main and branches
        """
        x, y, z = cls.get_xyzasnparay(trajectory, "main")
        nroot = cls.from_xyz(x, y, z, radius)

        nextbranch = True
        sbranchbase = "branch"
        i = 1
        while nextbranch:
            try:
                sbranch = sbranchbase + str(i)
                icolor = i % 4
                color = WellTreeTNO.branchcolors[icolor]
                xb, yb, zb = WellTreeTNO.get_xyzasnparay(
                    trajectory, sbranch, includebranchname=True
                )
                nroot.add_xyz(xb, yb, zb, sbranch=sbranch, color=color, radius=radius)
                i += 1
            except Exception:
                nextbranch = False
        return nroot

    @classmethod
    def from_vertical(cls, x, y, zmax, radius):
        """
        create a vertical wellTree object from x,y,z coordinates

        :param x: x coordinate
        :param y: y coordinate
        :param zmax: maximum z coordinate
        :param radius: radius of the well
        :return: wellTree object, name is 'main'
        """
        n0 = cls(x, y, 0, radius, perforated=False)
        _ = cls(
            x, y, zmax, radius, xroot=n0, perforated=False
        )  # Adding node to xroot, object is stored in n0.branches
        return n0

    def sameLocation(self, x: float64, y: float64, z: float64) -> bool:
        """
        check if the wellTree node is at the location x,y,z

        :param x: float
        :param y: float
        :param z: float
        :return: boolean if the wellTree node is at the location x,y,z
        """
        return (
            (abs(self.x[0] - x) < 1)
            and (abs(self.x[1] - y) < 1)
            and (abs(self.x[2] - z) < 1)
        )

    def exist(
        self, x: float, y: float, z: float
    ) -> Union[Tuple[bool, None], Tuple[bool, WellTreeTNO]]:
        """
        find the wellTree node in the wellTree which corresponds to the point with coordinates x,y,z

        :param x: float
        :param y: float
        :param z: float
        :return: boolean if the point exists, corresponding wellTree node
        """
        exist = False
        nfound = None
        if self.sameLocation(x, y, z):
            exist = True
            nfound = self
        else:
            for i, branch in enumerate(self.branches):
                exist, nfound = branch.exist(x, y, z)
                if exist:
                    return exist, nfound
        return exist, nfound

    def init_ahd(self) -> None:
        """
        initialize the along hole depth (AHD) for the wellTree

        """
        if self.xroot is None:
            self.ahd = 0
        else:
            self.ahd = self.xroot.ahd + self.getAHDincrement()
        for b in self.branches:
            b.init_ahd()

    def init_temperaturesoil(self, tsurface: float, tgrad: float) -> None:
        """
        initialize the Soil temperature surrounding the wellTree based on a surface temperature at Z=0 [C] and a temperature gradient [C/m]

        :param tsurface: surface temperature [C]
        :param tgrad: temperature gradient [C/m]

        """
        if self.xroot is None:
            self.temp = tsurface
        else:
            self.temp = tsurface + (-self.x[2]) * tgrad
        for b in self.branches:
            b.init_temperaturesoil(tsurface, tgrad)

    def getAHDincrement(self) -> float64:
        """
        get the along hole depth increment for the current node

        :return: float ahdincrement
        """
        dx = self.x - self.xroot.x
        ahdincrement = np.dot(dx, dx) ** 0.5
        return ahdincrement

    def cumulative_ahd(self) -> float64:
        """
        find (cumulative) ahd of the branches from this node

        :return: ahd of the branches in the root
        """
        ahd = 0
        if self.xroot is not None:
            dx = self.x - self.xroot.x
            ahd = np.dot(dx, dx) ** 0.5
        for i, branch in enumerate(self.branches):
            ahd += branch.cumulative_ahd()
        return ahd

    def get_ahd(
        self, TVD: Union[float, int], branchname: Optional[str] = "main"
    ) -> float64:
        """
          get the along hole depth (or MD)  of TVD, if it cannot be reached it returns the maximum AHD

        :param TVD: target true vertical depth
        :param branchname:  name of the branch to inspect, if it is not specified it follows the main branch
        :return: value of along hole depth corresponding to TVD

        """

        wlist = self.getbranchlist(name=branchname, perforated=None, includemain=True)

        # check if TVD is in range else rerutn best possible value
        if -wlist[0].x[2] >= TVD:
            return 0
        if -wlist[-1].x[2] < TVD:
            return wlist[-1].ahd
        #  return the ahd
        for b in wlist:
            if -b.x[2] >= TVD:
                # return ahd matching with TVD
                w = (-b.x[2] - TVD) / -(b.x[2] - b.xroot.x[2])
                ahd = b.ahd * (1 - w) + b.xroot.ahd * w
                return ahd

    def minmax(self, minx=None, maxx=None):
        """
        determine the min and max coordinates

        :param minx: min cooridnates tuple
        :param maxx: max coordinates tuple
        :return:
        """
        try:
            if minx is None:
                minx = self.x
                maxx = self.x
        except Exception:
            pass
        # print ('coordinates:', self.x)
        minx = np.minimum(self.x, minx)
        maxx = np.maximum(self.x, maxx)

        for branch in self.branches:
            minx, maxx = branch.minmax(minx, maxx)

        return minx, maxx

    def findlastsame(
        self, x: ndarray, y: ndarray, z: ndarray
    ) -> Tuple[WellTreeTNO, ndarray, ndarray, ndarray]:
        """
          finds the first coordinates in x,y,z starting from the last index which match the existing wellTreeNode
          if found it returns the wellTreeNode where the branch kicks off from existing wellTree
          and it returns the sliced arrays of x,y,z, coordinates containing the branch from the kickoff point

        :param x: np array (1d) with x coordinates of branch trajectory
        :param y: np array (1d) with y coordinates of branch trajectory
        :param z: np array (1d) with z coordinates of branch trajectory
        :return: the kickoff wellTree node for the branch and sliced arrays starting from the kickoff point
        """
        for i, val in enumerate(x):
            ind = np.size(x) - i - 1
            exist, nfound = self.exist(x[ind], y[ind], z[ind])
            if nfound is not None:
                xb = x[ind:]
                yb = y[ind:]
                zb = z[ind:]
                return nfound, xb, yb, zb
        return None, x, y, z

    @staticmethod
    def get_xyzasnparay(
        trajectory: Dict[str, DataFrame],
        branchname: str,
        includebranchname: Optional[bool] = False,
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """
            obtain x,y,z coordinate arrays from specific branches (or main) from the well stored in dictionary trajectory
            from trajectory[branchname] and columns for 'x','y', 'TVD' (if includebranchname is False) else
            columns for branchname + '_x', branchname + '_y', branchname + '_TVD'
            trajectory is created by the trajectories.py class from yml input

        :param trajectory:
        :param branchname:
        :param includebranchname:
        :return: x,y,z numpy 1d arrays of coordinates (assuming ordered towards toe of well)
        """
        sx = "x"
        sy = "y"
        sz = "TVD"
        if includebranchname:
            sx = branchname + "_" + sx
            sy = branchname + "_" + sy
            sz = branchname + "_" + sz
        x = np.asarray(trajectory[branchname][sx])
        y = np.asarray(trajectory[branchname][sy])
        z = np.asarray(trajectory[branchname][sz])
        return x, y, z

    def splitz(self, z):
        """
        split the well tree at depth z, adding z to the well tree if it is not already there

        :param z: depth to split the well tree
        """
        tiny = 1e-10
        parent = self.xroot
        if (abs(self.x[2] - z) > tiny) and (abs(parent.x[2] - z) > tiny):
            dz = self.x - parent.x
            z0 = z - parent.x[2]
            zscale = z0 / dz[2]
            xint = zscale * dz + parent.x
            # create new node at xint
            parent.branches.remove(self)
            nnew = WellTreeTNO(
                xint[0],
                xint[1],
                xint[2],
                self.radius,
                xroot=parent,
                perforated=False,
                color=self.color,
                name=self.name,
            )
            self.xroot = nnew
            self.perforated = False
            nnew.branches.append(self)

    def scale(self, scalevec):
        """
        scale the well tree by a vector scalevec

        :param scalevec: scale vector
        """
        for j in range(3):
            self.x[j] *= scalevec[j]
        for b in self.branches:
            b.scale(scalevec)

    def coarsen(self, segmentlength=100, perforated=True):
        """
        coarsen the topology to the target segmentlength and only the (non)perforated section if perforated!=None

        :param segmentlength: target segment length (m), alonh hole depth
        :param perforated:   consider only (non) perforated section (if !=None)
        :return:
        """
        # check if any of the branches can be merged to the child of the branch (only if it is not a kick off point)

        inext = 0
        while inext < len(self.branches):
            i = inext
            inext += 1
            b = self.branches[i]
            if b.branches is not None:
                if len(b.branches) == 1:
                    b2 = b.branches[0]
                    if (perforated is None) or (
                        b2.perforated == perforated and b.perforated == perforated
                    ):
                        xdif = b.x - b.xroot.x
                        dist1 = np.dot(xdif, xdif) ** 0.5
                        if dist1 < segmentlength:
                            # delete b and connect to b2
                            self.branches[i] = b2
                            b2.xroot = self
                            # revisit the branch
                            inext = inext - 1

        for b in self.branches:
            b.coarsen(segmentlength=segmentlength, perforated=perforated)

    def splitwell(self, ztop, zbase):
        """
        split the well tree to isolate the perforated section from ztop to zbase

        :param ztop: top coordinate (negative value)
        :param zbase: base coordinate (more negative value)
        :return:
        """
        tiny = 1e-10
        for b in self.branches:
            # crossing ztop?
            checkvaltop = (b.x[2] - ztop) * (self.x[2] - ztop)
            checkvalbase = (b.x[2] - zbase) * (self.x[2] - zbase)
            checkvalz = abs(b.x[2] - self.x[2])
            if checkvalz > tiny:
                # assume not horizontal and partially inside reservoir, calculate intersection, the split may not be necessary
                # if the ztop corresponds to self.x[2] or b.x[2]
                if checkvaltop < tiny:
                    b.splitz(ztop)

        for b in self.branches:
            # crossing zbase?
            checkvaltop = (b.x[2] - ztop) * (self.x[2] - ztop)
            checkvalbase = (b.x[2] - zbase) * (self.x[2] - zbase)
            checkvalz = abs(b.x[2] - self.x[2])
            if checkvalz > tiny:
                # assume not horizontal and partially inside reservoir, calculate intersection, the split may not be necessary
                # if the ztop corresponds to self.x[2] or b.x[2]
                if checkvalbase < tiny:
                    b.splitz(zbase)

        for b in self.branches:
            b.splitwell(ztop, zbase)

    def setperforate(self, ztop, zbase):
        """
        set the perforation status of the well tree segments

        :param ztop: float top coordinate (negative value)
        :param zbase: float base coordinate (more negative value)
        :return:
        """
        for b in self.branches:
            b.perforated = False
            if ((b.x[2] <= ztop) and (b.x[2] >= zbase)) and (
                (self.x[2] <= ztop) and (self.x[2] >= zbase)
            ):
                b.perforated = True

        for b in self.branches:
            b.setperforate(ztop, zbase)

    def perforate(self, ztop, zbase):
        """
        insert additional nodes to isolate the perforated section
        from ztop to zbottom, label the corresponding segments as perforated and color them blue

        :param ztop: top coordinate (negative value)
        :param zbase: bottom coordinate (more negative value)
        """

        self.splitwell(ztop, zbase)
        self.setperforate(ztop, zbase)

    @staticmethod
    def condenseBranch(wlistin, segmentlength=10):
        """
        condense the branch to the target segment length

        :param wlistin: list of well tree nodes
        :param segmentlength:  target segment length
        :return:
        """
        wlist = []
        for b in wlistin:
            wlist.append(b)
        # wlist = copy.deepcopy(wlistin)
        checkdist = True
        while checkdist:
            mindist = segmentlength
            for i, b in enumerate(wlist):
                if (i > 0) and (i < len(wlist) - 1):
                    xdif = wlist[i - 1].x - wlist[i].x
                    dist = np.dot(xdif, xdif) ** 0.5
                    if dist < mindist:
                        mindist = dist
                        i2del = i
                    xdif = wlist[i].x - wlist[i + 1].x
                    dist = np.dot(xdif, xdif) ** 0.5
                    if dist < mindist:
                        mindist = dist
                        i2del = i
            checkdist = mindist < segmentlength
            if checkdist:
                del wlist[i2del]
        return wlist

    def getbranch(
        self,
        name: Optional[str] = None,
        perforated: Optional[bool] = True,
        includemain: Optional[bool] = False,
        wlist: Optional[List["WellTreeTNO"]] = None,
    ) -> None:
        """
        collect the (perforated) segments
        in a list of welltree nodes, and begiining and ned points name and radius,
        this routine is used for the well index calculation

        :param name: name of the branch
        :param perforated: section (True or False) or ignored (perforated==None)
        :param includemain: include the main branch
        :param wlist: list of well tree nodes
        :return: dataframe with start and end points of the tree elements corresponding to name and perforated
        """
        wlist = self.getbranchlist(
            name=name, perforated=perforated, includemain=includemain, wlist=wlist
        )
        if self.xroot is None:
            xs = [p.xroot.x for p in wlist]
            xe = [p.x for p in wlist]
            name = [p.name for p in wlist]
            radius = [p.radius for p in wlist]
            data = {
                "wellTreeTNO": wlist,
                "xs": xs,
                "xe": xe,
                "name": name,
                "rw": radius,
            }
            wperf = pd.DataFrame(data)
            return wperf

    def getbranchlist(
        self,
        name: Optional[str] = None,
        perforated: Optional[bool] = True,
        includemain: Optional[bool] = False,
        wlist: Optional[List["WellTreeTNO"]] = None,
    ) -> List["WellTreeTNO"]:
        """
        collect the (perforated) segments
        in a list of welltree nodes, and begiining and ned points name and radius,
        this routine is used for the well index calculation

        :param name: name of the branch
        :param perforated: section (True or False) or ignored (perforated==None)
        :param includemain: include the main branch
        :param wlist: list of well tree nodes
        :return: list of well tree nodes correspondng to the name and perforated
        """
        if wlist is None:
            wlist = []
        for b in self.branches:
            if includemain and len(self.branches) == 1:
                # if only
                wlist.append(b)
            elif ((name is None) or (b.name == name)) and (
                (perforated is None) or (b.perforated == perforated)
            ):
                includemain = False
                wlist.append(b)
            b.getbranch(name, perforated, includemain, wlist)
        return wlist

    def getBranchSurvey(
        self, name=None, perforated=True, includemain=False, wlist=None
    ):
        """
        collect survey points of selected  (perforated) segments

        :param name: name of the branch
        :param perforated: section (True or False) or ignored (perforated==None)
        :param includemain: include the main branch
        :param wlist: list of well tree nodes
        :return: dataframe with start and end points of the tree elements corresponding to name and perforated
        """
        if wlist is None:
            wlist = []
        for b in self.branches:
            if includemain and len(self.branches) == 1:
                # if only
                wlist.append(b)
            elif ((name is None) or (b.name == name)) and (
                (perforated is None) or (b.perforated == perforated)
            ):
                includemain = False
                wlist.append(b)
            b.getbranch(name, perforated, includemain, wlist)
        if self.xroot is None:
            xs = [p.xroot.x[0] for p in wlist]
            xe = [p.x[0] for p in wlist]
            ys = [p.xroot.x[1] for p in wlist]
            ye = [p.x[1] for p in wlist]
            zs = [p.xroot.x[2] for p in wlist]
            ze = [p.x[2] for p in wlist]
            name = [p.name for p in wlist]
            radius = [p.radius for p in wlist]
            data = {
                "xs": xs,
                "xe": xe,
                "ys": ys,
                "ye": ye,
                "zs": zs,
                "ze": ze,
                "name": name,
                "rw": radius,
            }
            wperf = pd.DataFrame(data)
            return wperf

    def temploss(
        self,
        wdot: Series,
        temp: Series,
        time: int,
        kt: int,
        at: float,
        wellradius: Optional[float] = None,
    ) -> Series:
        """
        calculate the temperature loss for the well branch

        :param wdot:  massflow * heat capacity rate (W/m3)
        :param time: reference time (s)
        :param kt: thermal conductivity (W/mK)
        :param at: thermal diffusivity (m2/s)
        :param wellradius: well radius (m)
        :return: temperature loss
        """
        # try:
        L = self.ahd - self.xroot.ahd
        # except Exception:
        #    print(f"Error calculating ahd for {self.name}, x: {self.x}, xroot: {self.xroot.x}")
        rw = self.radius
        if wellradius is not None:
            rw = wellradius
        tgrad = (self.temp - self.xroot.temp) / L

        overkz = (np.log((4 * at * time) / rw**2) - 0.5772) / (4 * np.pi * kt * L)
        kzstar = (1 / overkz) / wdot
        F = self.temp
        Estar = -tgrad * L
        cc2 = Estar / kzstar + (temp - self.temp)
        theta = -Estar / kzstar + cc2 * np.exp(-kzstar)
        tend = F + Estar + theta
        temploss = temp - tend
        # print ("L, temploss", L, temploss)
        return temploss

    def plotTree3D(self, fig=None, ax=None, doplot=True, tofile=None):
        """
        plot the welltree in a 3D window

        :param fig: included only when appending to existing plot
        :param ax: included only when appending to existing plot
        :param doplot: if True the plot is finalized if False the result can be appended using return values in next call
        :param tofile: specify a file to save the resulting plot
        :return: fig, ax objects for next call when appending to existing (if doplot==False)
        """
        if fig is None:
            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_subplot(111, projection="3d")

        for b in self.branches:
            xplot = [self.x[0], b.x[0]]
            yplot = [self.x[1], b.x[1]]
            zplot = [self.x[2], b.x[2]]
            if b.perforated:
                ax.plot(xplot, yplot, zplot, color=self.perforationcolor)
            else:
                ax.plot(xplot, yplot, zplot, color=b.color)
            b.plotTree3D(fig, ax)

        # fix dore axis equal bug in matplotlib
        # Create cubic bounding box to simulate equal aspect ratio

        if (self.xroot is None) and (doplot):
            # ax.legend()
            ax.set_box_aspect([1, 1, 1])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("depth")

            plt.tight_layout()

            if tofile is None:
                plt.show()
            else:
                plt.savefig(tofile)

        if not doplot:
            return fig, ax
        else:
            return None, None

    def plotTree(self, fig=None, ax=None, axis=0):
        """
        plot the welltree in (x,y),  (x,z) or (y,z) plot

        :param fig: included only when appending to existing plot
        :param ax: included only when appending to existing plot
        :param axis: 0 means xy plot, 1
        :return: fig, ax objects for next call when appending to existing (if doplot==False)
        """

        if fig is None:
            fig = plt.figure(figsize=(9, 9))
            ax = plt.subplot(1, 1, 1)
            if axis == 0:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            elif axis == 1:
                ax.set_xlabel("x")
                ax.set_ylabel("depth")
            elif axis == 2:
                ax.set_xlabel("y")
                ax.set_ylabel("depth")
            # ax.legend()
            # plt.tight_layout()

        for b in self.branches:
            xplot = [self.x[0], b.x[0]]
            yplot = [self.x[1], b.x[1]]
            zplot = [self.x[2], b.x[2]]
            # print(' x,y,z plot ', xplot[0], yplot[0], zplot[0])

            if b.perforated:
                col = self.perforationcolor
            else:
                col = b.color
            if axis == 0:
                ax.plot(xplot, yplot, color=col)
            elif axis == 1:
                ax.plot(xplot, zplot, color=col)
            elif axis == 2:
                ax.plot(yplot, zplot, color=col)

            fig, ax = b.plotTree(fig, ax, axis=axis)

        return fig, ax

        # fix dore axis equal bug in matplotlib
        # Create cubic bounding box to simulate equal aspect ratio
