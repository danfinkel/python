import numpy as np
from scipy.spatial import Voronoi
import shapely.geometry as sg
import shapely.geometry
import shapely.ops


def voronoi(towers, bounding_box):
    """
    Create voronoi regions.
    We build extra regions so
    later we can pair down and
    create polygons.
       - towers --> list of tower coordinates
       - bounding_box --> limits of region (left, right, bottom, top)
    """
    # Mirror points
    points_center = towers
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)

    # Compute Voronoi over all points (input + mirror)
    vor = Voronoi(points)
    return vor


def is_in(poly, towers):
    """
    Check to see if any tower
    is in the polygon
        - poly --> shapely polygon
        - towers --> list of tower coordinates
    """
    for s in towers:
        pt = sg.Point(s)
        if poly.contains(pt):
            return True
    return False


def build_vor_polys(vor, exterior_poly, towers):
    """
    Create shapely polygons of the
    relevant voronoi regions
        - vor --> scipy voronoi output
        - exterior_poly --> exterior region polygon
        - towers --> list of tower coordinates
    """
    polys = []

    # Build all lines in the voronoi region set
    lines = [shapely.geometry.LineString(vor.vertices[line])
            for line in vor.ridge_vertices
            if -1 not in line]

    # polygonize the lines and check if
    # they are original or copies
    for poly in shapely.ops.polygonize(lines):
        if is_in(poly, towers):
            poly = poly.intersection(exterior_poly)
            x, y = poly.exterior.xy
            polys.append(poly)
    return polys
