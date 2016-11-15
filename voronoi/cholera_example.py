import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shapefile
import shapely.geometry as sg

from voronoi_library import voronoi, build_vor_polys


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
