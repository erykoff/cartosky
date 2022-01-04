import os

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import cartosky  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


def test_lines_polygons(tmp_path):
    """Test drawing lines and polygons."""
    # This code draws a bunch of geodesics and polygons that are
    # both empty and filled, and cross over the boundary, to ensure
    # that features are working as intended.
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = cartosky.McBrydeSkymap(ax=ax)
    # Draw a simple geodesic
    m.draw_line_lonlat([0., 45.], [0., 45.], color='red', label='One')
    # Draw another geodesic that crosses the wrap boundary
    m.draw_line_lonlat([170., -170.], [0., 45.],
                       color='blue', linestyle='--', label='Two')
    # Draw a simple unfilled polygon
    m.draw_polygon_lonlat([20, 40, 40, 20], [20, 20, 40, 40],
                          edgecolor='magenta', label='Three')
    # Draw a simple filled polygon
    m.draw_polygon_lonlat([20, 40, 40, 20], [-20, -20, -40, -40],
                          edgecolor='black', facecolor='red', linestyle='--', label='Four')
    # Draw a simple unfilled polygon that wraps around
    m.draw_polygon_lonlat([170, 190, 190, 170], [20, 20, 40, 40],
                          edgecolor='green', label='Five')
    # Draw a simple filled polygon that wraps around
    m.draw_polygon_lonlat([170, 190, 190, 170], [-20, -20, -40, -40],
                          edgecolor='yellow', facecolor='blue', label='Six')
    m.legend()
    fname = 'lines_and_polygons.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
    if err:
        raise ImageComparisonFailure(err)
