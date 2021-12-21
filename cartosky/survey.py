from .skymap import Skymap
from .utils import get_datafile  # fix this

__all__ = ['SurveySkymap', 'DESSkymap']


class SurveySkymap(Skymap):
    """Extension of a Skymap to add routines for drawing survey outlines.
    """
    def draw_des(self, **kwargs):
        """Draw the DES footprint."""
        return self.draw_des17(**kwargs)

    def draw_des17(self, edgecolor='red', lw=2, **kwargs):
        """Draw the DES 2017 footprint."""
        filename = get_datafile('des-round17-poly.txt')
        return self.draw_polygon_file(filename, edgecolor=edgecolor, lw=lw, **kwargs)


class DESSkymap(SurveySkymap):
    def __init__(self, ax=None, projection_name='mbtfpq', lon_0=0, gridlines=True,
                 celestial=True, extent=[90, -50, -75, 10], **kwargs):
        super().__init__(ax=ax, projection_name=projection_name, lon_0=lon_0, gridlines=gridlines,
                         celestial=celestial, extent=extent, **kwargs)

    def draw_hpxmap(self, hpxmap, nest=False, zoom=False, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        return super().draw_hpxmap(hpxmap,
                                   nest=nest,
                                   zoom=zoom,
                                   xsize=xsize,
                                   vmin=vmin,
                                   vmax=vmax,
                                   rasterized=rasterized,
                                   lon_range=lon_range,
                                   lat_range=lat_range,
                                   **kwargs)

    def draw_hpxpix(self, nside, pixels, values, nest=False, zoom=False, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        return super().draw_hpxpix(nside,
                                   pixels,
                                   values,
                                   nest=nest,
                                   zoom=zoom,
                                   xsize=xsize,
                                   vmin=vmin,
                                   vmax=vmax,
                                   rasterized=rasterized,
                                   lon_range=lon_range,
                                   lat_range=lat_range,
                                   **kwargs)

    def draw_hspmap(self, hspmap, zoom=False, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        return super().draw_hspmap(hspmap,
                                   zoom=zoom,
                                   xsize=xsize,
                                   vmin=vmin,
                                   vmax=vmax,
                                   rasterized=rasterized,
                                   lon_range=lon_range,
                                   lat_range=lat_range,
                                   **kwargs)

    def draw_hpxbin(self, lon, lat, C=None, nside=256, nest=False, zoom=False, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        return super().draw_hpxmap(lon,
                                   lat,
                                   C=C,
                                   nside=nside,
                                   nest=nest,
                                   zoom=zoom,
                                   xsize=xsize,
                                   vmin=vmin,
                                   vmax=vmax,
                                   rasterized=rasterized,
                                   lon_range=lon_range,
                                   lat_range=lat_range,
                                   **kwargs)
