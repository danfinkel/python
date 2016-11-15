# -*- coding: utf-8 -*-
"""
    Standard geospatial coordinate tranformations

    Created on Wed Apr 22 12:14:27 2015
    Author: Dan Finkel (dan.finkel@gmail.com)
"""
import numpy as np
from math import pow, degrees
from scipy import arctan, sqrt, arctan2


def geod2ecef(lat, lon, alt):
    """
     Convert WGS-84 geodetic coordinates to ECEF using an ellipsodial model.

      lat, lon are vertical vectors in radians.
      alt    is a vertical vector altitude in meters.

      Output is vecef which is ECEF xyz [#pts  3]
    """

    vecef = []
    # Earth flatening factor
    ESQ = 0.00669437999013

    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)

    # Uses earth MAJOR radius (Equator)
    grad = 6378137.0 / np.sqrt(1.0 - (slat**2) * ESQ)

    vecef.append((grad + alt) * clat * clon)
    vecef.append((grad + alt) * clat * slon)
    vecef.append((grad * (1.0 - ESQ) + alt) * slat)

    return np.array(vecef)


def ecef2enu(vecef, gref):
    """

     Converts ECEF vector to a topocentric ENU vector wrt reference point "gref"
     (in WGS-84 coordinates).  Translation from the reference point is performed.

      vecef    Matrix of XYZ-ecef points [#pts by 3].
      gref Input reference point is in Lat, Long, Alt in rads, rads and meters.

      venu Matrix of XYZ points w.r.t. gref point [#pts by 3] in meters.
    """

    # Convert gref to ECEF
    ref_ecef = np.array(geod2ecef(gref[0], gref[1], gref[2]))

    # Get the rotation matrix from ecef to the reference lat/long point.
    slat = np.sin(gref[0])
    clat = np.cos(gref[0])
    slon = np.sin(gref[1])
    clon = np.cos(gref[1])

    # Rotation matrix
    rot = np.array([[-slon, -clon * slat, clon * clat],
                    [clon, -slon * slat, slon * clat],
                    [0.0, clat, slat]])

    tmp = vecef - ref_ecef
    venu = np.dot(tmp, rot)

    return venu


def geod2enu(lla, gref):
    """
     Converts geodetic coordinates to local enu frame centered at gref
    """

    ecef = geod2ecef(lla[0], lla[1], lla[2])
    enu = ecef2enu(ecef, gref)

    return enu


def enu2geod(enu, gref):
    """
     Converts geodetic coordinates to local enu frame centered at gref
    """

    ecef = enu2ecef(gref, enu)
    geod = ecef2geodetic(ecef[0], ecef[1], ecef[2])

    return geod


def enu2rae(enu, Penu):
    """
     Convert enu to rae
    """

    if enu.size == 3:

        # Build rae vector
        rae = np.zeros(3)

        rae[0] = np.sqrt(np.sum(enu * enu))
        rae[1] = np.arctan2(enu[0], enu[1])
        rae[2] = np.arctan2(enu[2], np.sqrt(np.sum(enu[0:2] * enu[0:2])))

        Rg = np.sqrt(np.sum(enu[0:2] * enu[0:2]))

        # Build covariance matrix
        H = np.zeros([3, 3])

        # First row is enu/rae[0]
        H[0, 0] = enu[0] / rae[0]
        H[0, 1] = enu[1] / rae[0]
        H[0, 2] = enu[2] / rae[0]

        # 2nd row
        H[1, 0] = enu[1] / Rg**2
        H[1, 1] = -1 * enu[0] / Rg**2

        # 3rd row
        H[2, 0] = -1 * enu[0] * enu[2] / ((rae[0]**2) * Rg)
        H[2, 1] = -1 * enu[1] * enu[2] / ((rae[0]**2)*Rg)
        H[2, 2] = Rg / rae[0]**2

        Prae = np.dot(np.dot(H, Penu), np.transpose(H))
    else:
        # Build rae vector
        rae = np.zeros(2)

        rae[0] = np.sqrt(np.sum(enu * enu))
        rae[1] = np.arctan2(enu[0], enu[1])

        Rg = np.sqrt(np.sum(enu[0:2] * enu[0:2]))

        # Build covariance matrix
        H = np.zeros([2, 2])

        # First row is enu/rae[0]
        H[0, 0] = enu[0] / rae[0]
        H[0, 1] = enu[1] / rae[0]

        # 2nd row
        H[1, 0] = enu[1] / Rg**2
        H[1, 1] = -1 * enu[0] / Rg**2

        Prae = np.dot(np.dot(H, Penu), np.transpose(H))
    return rae, Prae


def rae2eunu(rae, Prae):
    """
    Convert rae to enu
    """

    # Build enu vector
    enu = np.zeros(3)

    enu[0] = rae[0] * np.sin(rae[1]) * np.cos(rae[2])
    enu[1] = rae[0] * np.cos(rae[1]) * np.cos(rae[2])
    enu[2] = rae[0] * np.sin(rae[2])

    A = np.zeros([3, 3])

    A[0, 0] = np.sin(rae[1]) * np.cos(rae[2])
    A[0, 1] = rae[0] * np.cos(rae[1]) * np.cos(rae[2])
    A[0, 2] = -rae[0] * np.sin(rae[1]) * np.sin(rae[2])

    A[1, 0] = np.cos(rae[1]) * np.cos(rae[2])
    A[1, 1] = -rae[0] * np.sin(rae[1]) * np.cos(rae[2])
    A[1, 2] = -rae[0] * np.cos(rae[1]) * np.sin(rae[2])

    A[2, 0] = np.sin(rae[2])
    A[2, 1] = 0
    A[2, 2] = rae[0] * np.cos(rae[2])

    Penu = np.dot(np.dot(A, Prae), np.transpose(A))

    return enu, Penu


def ecef2geodetic(x, y, z):
    """
    Convert ecef 2 geodetic
        Convert ECEF coordinates to geodetic.
        J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates
        to geodetic coordinates," IEEE Transactions on Aerospace and
        Electronic Systems, vol. 30, pp. 957-961, 1994.
    """

    a = 6378137
    b = 6356752.3142

    esq = 0.00669437999013
    e1sq = 6.73949674228 * 0.001

    r = sqrt(x * x + y * y)
    Esq = a * a - b * b
    F = 54 * b * b * z * z
    G = r * r + (1 - esq) * z * z - esq * Esq
    C = (esq * esq * F * r * r) / (pow(G, 3))
    S = sqrt(1 + C + sqrt(C * C + 2 * C))
    P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
    Q = sqrt(1 + 2 * esq * esq * P)
    r_0 = -(P * esq * r) / (1 + Q) + sqrt(0.5 * a * a * (1 + 1.0 / Q) -
        P * (1 - esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
    U = sqrt(pow((r - esq * r_0), 2) + z * z)
    V = sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z)
    Z_0 = b * b * z / (a * V)
    h = U * (1 - b * b / (a * V))
    lat = arctan((z + e1sq * Z_0) / r)
    lon = arctan2(y, x)

    return degrees(lat), degrees(lon), h


def enu2ecef(gref, enu):
    """
    Convert enu 2 ecef
    """

    # Convert gref to ECEF
    ref_ecef = np.array(geod2ecef(gref[0], gref[1], gref[2]))

    # Get the rotation matrix from ecef to the reference lat/long point.
    slat = np.sin(gref[0])
    clat = np.cos(gref[0])
    slon = np.sin(gref[1])
    clon = np.cos(gref[1])

    # Rotation matrix
    rot = np.array([[-slon, -clon * slat, clon * clat],
                    [clon, -slon * slat, slon * clat],
                    [0.0, clat, slat]])

    invrot = np.linalg.inv(rot)
    ecef = np.dot(enu, invrot) + ref_ecef
    return ecef


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes dis in m between two lat/lon pairs using Haversine formula

     R = Earth Radius
     lat_d = lat2 - lat1
     lon_d = lon2 - lon1
     a = sin^2(lat_d / 2) + cos(lat1) * cos(lat2) * sin^2(lon_d / 2)
    """

    EARTH_RADIUS_M = 6371000.0

    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = np.sin(delta_lat / 2) * np.sin(delta_lat / 2) + \
        np.cos(np.radians(lat1)) * \
        np.cos(np.radians(lat2)) * \
        np.sin(delta_lon / 2) * np.sin(delta_lon / 2)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return c * EARTH_RADIUS_M
