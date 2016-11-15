import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shapefile
from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely.geometry as sg
import shapely.geometry
import shapely.ops

"""
Plots with Voronoi overlays of 1854 Soho Cholera outbreak.
Data sourced from
    http://blog.rtwilson.com/john-snows-famous-cholera-analysis-data-in-modern-gis-formats/
"""


def build_soho_poly(bbox):

    coords = np.array([[bbox[0], bbox[2]],
               [bbox[0], bbox[3]],
               [bbox[1], bbox[3]],
               [bbox[1], bbox[2]],
               [bbox[0], bbox[2]]])
    sohopoly = sg.Polygon(coords)
    return sohopoly


def voronoi(towers, bounding_box):

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
    # Compute Voronoi
    vor = Voronoi(points)
    return vor


def is_in(poly, pumps):
    for s in pumps:
        pt = sg.Point(s)
        if poly.contains(pt):
            return True
    return False


def build_vor_polys(vor, soho_poly, pumps_xy):

    polys = []

    # Build all lines in the voronoi region set
    lines = [shapely.geometry.LineString(vor.vertices[line])
            for line in vor.ridge_vertices
            if -1 not in line]

    # polygonize the lines and check if
    # they are original or copies
    for poly in shapely.ops.polygonize(lines):
        if is_in(poly, pumps_xy):
            poly = poly.intersection(soho_poly)
            x, y = poly.exterior.xy
            polys.append(poly)
    return polys


if __name__ == '__main__':

    # Pull in pump and death locations
    pump_shps = shapefile.Reader("SnowGIS/Pumps.shp")
    death_shps = shapefile.Reader("SnowGIS/Cholera_Deaths.shp")

    # Extract xy locations
    pumps_xy = np.array([s.points[0] for s in pump_shps.shapes()])
    deaths_xy = np.array([d.points[0] for d in death_shps.shapes()])

    # Full color image
    # img = mpimg.imread('OSMap.png')
    # f = open('OSMap.tfw', 'rb')

    # Grayscale image
    img = mpimg.imread('SnowGIS/OSMap_Grayscale.tif')
    f = open('SnowGIS/OSMap_Grayscale.tfw', 'rb')

    # tfw file includes scaling information for the tiff image
    tfw_data = [float(line.rstrip('\n')) for line in f]

    # Scale and display image
    xmin = tfw_data[4]
    xmax = xmin + img.shape[1] * tfw_data[0]
    ymin = tfw_data[5]
    ymax = ymin + img.shape[0] * tfw_data[3]
    imgplot = plt.imshow(img, extent=(xmin, xmax, ymax, ymin))

    # Overlay pumps and death locations
    plt.plot(pumps_xy[:, 0], pumps_xy[:, 1], '.', markersize=24)
    plt.plot(deaths_xy[:, 0], deaths_xy[:, 1], '.r', markersize=8)

    bbox = np.array([xmin, xmax, ymax, ymin])
    vor = voronoi(pumps_xy, bbox)
    soho_poly = build_soho_poly(bbox)
    polys = build_vor_polys(vor, soho_poly, pumps_xy)
    # voronoi_plot_2d(vor)
    for poly in polys:
        x, y = poly.exterior.xy
        plt.plot(x, y, color='#006600', alpha=0.7, linewidth=5, solid_capstyle='round', zorder=2)

    plt.show()