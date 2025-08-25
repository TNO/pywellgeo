from typing import Tuple, Union

import numpy as np
from numpy import float64, ndarray

from pywellgeo.transformations.azim_dip import TINY, AzimDip


class CoordinateTransformation:
    """
    class to support coordinate transformations

    transformation is determined by rotated basis and global origin spec

    the transformed and rotated basis is based on

      x(local) =  R T x(global)

      - x = homogeneous coordinates (x,y,z,1)
      - T = 4x4 translation matrix. This is a diagonal unit matrix with in  the fourth column the translation components  -ox,-oy,-oz , where ox,oy,oz is the orogin
      - R = 4x4  unit matrix, where 3x3 first rows and columns is the rotation matrix.

    It contains in the rows the unit axes (in global coordinate system orientation)

    The initialization is performed in two possible manners:

      - from a definition of the plane for the local coordinates (containing two of the axes), and direction of the first axis.
        The plane and first axis are defined by spherical coordinates azimdip and the pitch of the first axis in the plane
        (the third corresponds to the normal to the plane and the second to the outer product of the thrid and first)

      - from a rotated axis framework where the first two unit vectors are given

    The backrotation is performed by using the inverse of RT

       - x(global) =  (RT)-1 x(local)

    """

    def __init__(
        self,
        plane: AzimDip,
        origin: ndarray = np.asarray([0, 0, 0]),
        pitch: Union[float64, int] = 0,
    ) -> None:
        """
        initialize the coordinate transformation

        :param plane: spherical coordinates of the plane to use,
                        azim/dip is the reference direction for the x-axis for pitch is 0
                        y-axis is in the plane (90 degrees anticlockwise), z-axis is pointing upward

        :param origin: of the local coordinate system

        :param pitch: rotation clockwise of the local x-axis in the plane relative to the azim-dip,
                        for pitch=0, 90/0 in local coordinates corresponds to the plane azim/dip in global coordinates
                        for pitch=90, 0/0 in local coordinates corresponds to the plane azim/dip in global coordinates
        """

        self.plane = plane
        self.origin = origin
        self.plane = plane
        theta = np.radians(plane.azim - 90)
        r1 = self.rotz(theta)
        theta = np.radians(-plane.dip)
        r2 = self.roty(theta)
        if abs(pitch) > TINY:
            theta = np.radians(pitch)
            rpitch = self.rotz(theta)
            r2 = np.matmul(rpitch, r2)
        self.r3 = np.matmul(r2, r1)

        self.r = np.identity(4)
        self.r[0:3, 0:3] = self.r3
        self.t = np.identity(4)

        self.t[0:3, 3] = -origin

        self.rt = np.matmul(self.r, self.t)
        self.rtinv = np.linalg.inv(self.rt)

    def transform2local(self, vglobal: ndarray) -> ndarray:
        """
        transform a vector from global to local coordinates

        :param vglobal: np array with 3 elements representing the vector in global coordinates
        :return: np array with 3 elements representing the vector in local coordinates
        """
        v4 = np.zeros(4)
        v4[0:3] = vglobal
        v4[3] = 1.0
        r = np.matmul(self.rt, v4)
        return r[0:3]

    def transform2global(self, vlocal: ndarray) -> ndarray:
        """
        transform a vector from local to global coordinates

        :param vlocal: np array with 3 elements representing the vector in local coordinates
        :return: np array with 3 elements representing the vector in global coordinates
        """
        v4 = np.zeros(4)
        v4[0:3] = vlocal
        v4[3] = 1
        r = np.matmul(self.rtinv, v4)
        return r[0:3]

    def transform2local_orientation(self, azimdip: AzimDip) -> AzimDip:
        """
        transform an orientation from global to local coordinates

        :param azimdip: AzimDip object representing the orientation in global coordinates
        :return: local AzimDip object representing the orientation in local coordinates
        """
        vglobal = azimdip.azimdip2Vector()
        r = np.matmul(self.r3, vglobal)
        adlocal = AzimDip.from_vector(r)
        return adlocal

    def transform2global_orientation(self, azimdip: AzimDip) -> AzimDip:
        """
        transform an orientation from local to global coordinates

        :param azimdip: AzimDip object representing the orientation in local coordinates
        :return: global AzimDip object representing the orientation in global coordinates
        """
        vlocal = azimdip.azimdip2Vector()
        r = np.matmul(self.r3.T, vlocal)
        adglobal = AzimDip.from_vector(r)
        return adglobal

    def rotx(self, theta):
        """
        rotx gives rotation matrix about X axis

        :param theta: radians angle for rotation matrix
        :return: rotation matrix (3x3) representing a rotation of theta radians about the x-axis
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        mat = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        return mat

    def roty(self, theta: float64) -> ndarray:
        """
        roty gives rotation matrix about X axis

        :param theta: radians angle for rotation matrix
        :return: rotation matrix (3x3) representing a rotation of theta radians about the y-axis
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        mat = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        return mat

    def rotz(self, theta: float64) -> ndarray:
        """
        rotz gives rotation matrix about X axis

        :param theta: radians angle for rotation matrix
        :return: rotation matrix   (3x3) representing a rotation of theta radians about the z-axis
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        mat = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
        return mat

    @classmethod
    def plane_pitch_from_vectors(
        cls, vecx: ndarray, vec2: ndarray
    ) -> Union[Tuple[AzimDip, int], Tuple[AzimDip, float64]]:
        """
        get the plane and pitch from two vectors, sharing the origin as starting point

            :param vecx (np array) the first vector and x-axis orientation
            :param vec2 (np array) the second vector in the plane
            :return: Azimdip object representing the plane and the pitch of the x-axis in the plane
        """
        v1 = AzimDip.normalize_vector(vecx)
        v2 = AzimDip.normalize_vector(vec2)

        if abs(np.dot(v1, v2) - 1) < 1e-10:
            print("warning parallel orientation  take as plane ")
            plane = AzimDip.from_vector(v1)
            pitch = 90
            return plane, pitch

        v3 = np.cross(v1, v2)
        v3 = AzimDip.normalize_vector(v3)

        adnormal = AzimDip.from_vector(v3)

        adnormal.plane2normal()
        plane = adnormal
        adpitch = AzimDip.from_vector(v1)

        cs = CoordinateTransformation(plane)
        adpitch_plane = cs.transform2local_orientation(adpitch)
        pitch = adpitch_plane.azim

        return plane, pitch

    def line_plane_intersect(self, ad_ray: AzimDip, rayPoint: ndarray) -> ndarray:
        """
        calculate the intersection of line (shooting from raypoint in ad_ray direction) and the plane of self

        :param ad_ray: AzimDip object representing the direction of the line
        :param rayPoint: np array representing the starting point of the line

        :return: intersection point in the coordinate-system of self(its x,y,z coordinates)
        """
        epsilon = 1e-6

        # Define plane
        planeNormal = self.plane.azimdip2normal()
        planePoint = self.origin  # Any point on the plane

        # Define ray
        rayDirection = ad_ray.azimdip2Vector()

        ndotu = planeNormal.dot(rayDirection)

        if abs(ndotu) < epsilon:
            print("no intersection or line is within plane")

        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint
        return Psi

    def planes_intersect(self, b_t):
        """
        calculate the intersection of the two basis planes of the transformation.

        :param b_t: CoordinateTransformation object representing the other basis


        return: intersection point and AzimDip of orientation of intersection, if
        """

        a_vec = self.plane.azimdip2normal()
        a3 = a_vec.cross(self.origin)
        b_vec = b_t.plane.azimdip2normal()
        b3 = b_vec.cross(b_t.origin)

        aXb_vec = np.cross(a_vec, b_vec)

        A = np.array([a_vec, b_vec, aXb_vec])
        d = np.array([-a3, -b3, 0.0]).reshape(3, 1)

        # TODO: could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error
        p_inter = np.linalg.solve(A, d).T

        ad_aXb = AzimDip.from_vector(aXb_vec)

        return p_inter[0], ad_aXb

    @classmethod
    def test(cls):
        ad1 = AzimDip(160, 90)
        orx = np.asarray([100, 200, 300])
        pitchplane = 0
        cls = CoordinateTransformation(ad1, origin=orx, pitch=pitchplane)
        adl = cls.transform2local_orientation(ad1)
        print(adl)
        adg = cls.transform2global_orientation(adl)
        print(adg)

        v1 = np.array([0.5, 0.0, 0.5])
        v2 = np.array([0.5, 0.0, -0.5])
        plane, pitch = cls.plane_pitch_from_vectors(v2, v1)
        print("plane ", plane)
        print("pitch", pitch)

        adtest = AzimDip.from_vector(v2)
        cls = CoordinateTransformation(plane, pitch=pitch)
        adl = cls.transform2local_orientation(adtest)
        print(adl)

        return cls
