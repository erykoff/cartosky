import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
# import healsparse as hsp

from cartopy.crs import PlateCarree

import mpl_toolkits.axisartist as axisartist
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .projections import get_projection
from .formatters import WrappedFormatterDMS
from .hpx_utils import healpix_pixels_range, hspmap_to_xy, hpxmap_to_xy, healpix_to_xy

__all__ = ['Skymap', 'McBrydeSkymap', 'OrthoSkymap', 'MollweideSkymap', 'HammerSkymap', 'AitoffSkymap']


class Skymap():
    """Base class for creating Skymap objects.

    Parameters
    ----------
    """
    def __init__(self, ax=None, projection_name='cyl', lon_0=0, gridlines=True, celestial=True,
                 extent=None, **kwargs):
        # self.set_observer(kwargs.pop('observer', None))
        # self.set_date(kwargs.pop('date', None))

        if ax is None:
            ax = plt.gca()

        fig = ax.figure
        subspec = ax.get_subplotspec()
        fig.delaxes(ax)

        kwargs['lon_0'] = lon_0
        self.projection = get_projection(projection_name, **kwargs)
        self._ax = fig.add_subplot(subspec, projection=self.projection)
        self._aa = None

        self._ax.set_global()

        self.do_celestial = celestial
        self.do_gridlines = gridlines

        self._wrap = (lon_0 + 180.) % 360.
        self._lon_0 = self.projection.proj4_params['lon_0']

        if extent is None:
            extent = [-self._wrap, self._wrap, -90.0, 90.0]

        self.set_extent(extent)

    def proj(self, lon_or_x, lat_or_y, inverse=False):
        """Apply projection.

        Convert from lon/lat to x/y (forward) or x/y to lon/lat (inverse=True).

        Parameters
        ----------
        lon_or_x : `float` or `list` or `np.ndarray`
            Longitude(s) (RA) (forward) or x values (inverse)
        lat : `float` or `list` or `np.ndarray`
            Latitude(s) (Dec) (forward) or y values (inverse)
        inverse : `bool`, optional
            Apply inverse transformation from x/y to ra/dec.

        Returns
        -------
        x_or_lon : `np.ndarray`
            Array of x values (forward) or longitudes (inverse)
        y_or_lat : `np.ndarray`
            Array of y values (forward) or latitudes (inverse)
        """
        lon_or_x = np.atleast_1d(lon_or_x)
        lat_or_y = np.atleast_1d(lat_or_y)
        if inverse:
            proj_xyz = PlateCarree().transform_points(self.projection, lon_or_x, lat_or_y)
        else:
            proj_xyz = self.projection.transform_points(PlateCarree(), lon_or_x, lat_or_y)
        return proj_xyz[..., 0], proj_xyz[..., 1]

    def set_extent(self, extent):
        """Set the extent and create an axis artist.

        Note that calling this method will remove all formatting options.

        Parameters
        ----------
        extent : array-like
            Extent as [lon_min, lon_max, lat_min, lat_max].
        """
        # Reset any axis artist if necessary
        if self._aa is not None:
            self._aa.remove()
            self._aa = None

        self._set_axes_limits(extent, invert=False)
        self._create_axes()
        self._set_axes_limits(extent, invert=self.do_celestial)

        self._ax.set_frame_on(False)
        if self.do_gridlines:
            self._aa.grid(True, linestyle=':', color='k', lw=0.5)

    def _set_axes_limits(self, extent, invert=True):
        """Set axis limits from an extent.

        Parameters
        ----------
        extent : array-like
            Extent as [lon_min, lon_max, lat_min, lat_max].
        """
        if len(extent) != 4:
            raise ValueError("Must specify extent as a 4-element array.")

        # Check if the longitude range is the full sphere.
        if np.isclose(np.abs(extent[0] - extent[1]), 360.0):
            lon, lat = np.meshgrid(
                np.linspace(extent[0], extent[1], 20), np.linspace(extent[2], extent[3], 20)
            )
            x, y = self.proj(lon.ravel(), lat.ravel())
            extent_xform = [np.max(x),
                            np.min(x),
                            np.min(y),
                            np.max(y)]
            self._ax.set_extent(extent_xform, crs=self.projection)
        else:
            self._ax.set_extent(extent, crs=PlateCarree())

        if self._aa is not None:
            self._aa.set_xlim(self._ax.get_xlim())
            self._aa.set_ylim(self._ax.get_ylim())

        if invert:
            self._ax.invert_xaxis()
            if self._aa is not None:
                self._aa.invert_xaxis()

        return self._ax.get_xlim(), self._ax.get_ylim()

    def _create_axes(self):
        """Create axes and axis artist.
        """
        def tr(lon, lat):
            return self.proj(lon, lat)

        def inv_tr(x, y):
            return self.proj(x, y, inverse=True)

        extreme_finder = angle_helper.ExtremeFinderCycle(20, 20)
        grid_locator1 = angle_helper.LocatorD(10, include_last=False)
        grid_locator2 = angle_helper.LocatorD(6, include_last=False)

        tick_formatter1 = WrappedFormatterDMS(self._wrap)
        tick_formatter2 = angle_helper.FormatterDMS()

        grid_helper = GridHelperCurveLinear(
            (tr, inv_tr),
            extreme_finder=extreme_finder,
            grid_locator1=grid_locator1,
            grid_locator2=grid_locator2,
            tick_formatter1=tick_formatter1,
            tick_formatter2=tick_formatter2,
        )

        fig = self._ax.figure
        rect = self._ax.get_position()
        self._aa = axisartist.Axes(fig, rect, grid_helper=grid_helper, frameon=False)
        fig.add_axes(self._aa)

        def format_coord(x, y):
            return 'lon=%1.4f, lat=%1.4f' % (inv_tr(x, y))

        self._aa.format_coord = format_coord
        self._aa.axis['left'].major_ticklabels.set_visible(True)
        self._aa.axis['right'].major_ticklabels.set_visible(False)
        self._aa.axis['bottom'].major_ticklabels.set_visible(True)
        self._aa.axis['top'].major_ticklabels.set_visible(True)

        self.set_xlabel('Right Ascension', size=16)
        self.set_ylabel('Declination', size=16)

        fig.sca(self._ax)

        return fig, self._ax

    def set_xlabel(self, text, side='bottom', **kwargs):
        """Set the label on the x axis.

        Parameters
        ----------
        text : `str`
            x label string.
        side : `str`, optional
            Side to set the label.  Can be ``bottom`` or ``top``.
        **kwargs : `dict`
            Additional keyword arguments accepted by ax.set_xlabel().
        """
        return self._aa.axis[side].label.set(text=text, **kwargs)

    def set_ylabel(self, text, side='left', **kwargs):
        """Set the label on the y axis.

        Parameters
        ----------
        text : `str`
            x label string.
        side : `str`, optional
            Side to set the label.  Can be ``left`` or ``right``.
        **kwargs : `dict`
            Additional keyword arguments accepted by ax.set_xlabel().
        """
        return self._aa.axis[side].label.set(text=text, **kwargs)

    @property
    def ax(self):
        return self._ax

    @property
    def aa(self):
        return self._aa

    def compute_extent(self, lon, lat):
        """Compute plotting extent for a set of lon/lat points.

        Uses a simple looping algorithm to find the ideal range so that
        all the points fit within the projected frame, with a small border.

        Parameters
        ----------
        lon : `np.ndarray`
            Array of longitude values.
        lat : `np.ndarray`
            Array of latitude values.

        Returns
        -------
        extent : `list`
            Plotting extent [lon_max, lon_min, lat_min, lat_max]
        """
        lon_wrap = (lon + self._wrap) % 360 - self._wrap

        # Compute lat range with cushion
        lat_min0 = np.min(lat)
        lat_max0 = np.max(lat)
        lat_range = lat_max0 - lat_min0
        lat_min = np.clip(lat_min0 - 0.05*lat_range, -90.0, None)
        lat_max = np.clip(lat_max0 + 0.05*lat_range, None, 90.0)

        # Compute an ideally fitting lon range with cushion
        x, y = self.proj(lon, lat)

        lon_min0 = np.min(lon_wrap)
        lon_max0 = np.max(lon_wrap)
        lon_step = (lon_max0 - lon_min0)/20.
        lon_cent = (lon_min0 + lon_max0)/2.

        # Compute lon_min so that it fits all the data.
        enclosed = False
        lon_min = lon_cent - lon_step
        while not enclosed and lon_min > (self._lon_0 - 180.0):
            e_x, e_y = self.proj([lon_min, lon_min], [lat_min, lat_max])
            n_out = np.sum(x < e_x.min())
            if n_out == 0:
                enclosed = True
            else:
                lon_min = np.clip(lon_min - lon_step, self._lon_0 - 180., None)

        # Compute lon_max so that it fits all the data
        enclosed = False
        lon_max = lon_cent + lon_step
        while not enclosed and lon_max < (self._lon_0 + 180.0):
            e_x, e_y = self.proj([lon_max, lon_max], [lat_min, lat_max])
            n_out = np.sum(x > e_x.max())
            if n_out == 0:
                enclosed = True
            else:
                lon_max = np.clip(lon_max + lon_step, None, self._lon_0 + 180.)

        return [lon_max, lon_min, lat_min, lat_max]

    def plot(self, *args, transform=PlateCarree(), **kwargs):
        """Plot with ax.plot(*args, **kwargs)."""
        return self._ax.plot(*args, transform=transform, **kwargs)

    def scatter(self, *args, transform=PlateCarree(), **kwargs):
        """Plot with ax.scatter(*args, **kwargs)."""
        return self._ax.scatter(*args, transform=transform, **kwargs)

    def pcolormesh(self, *args, transform=PlateCarree(), **kwargs):
        """Plot with ax.pcolormesh(*args, **kwargs)."""
        return self._ax.pcolormesh(*args, transform=transform, **kwargs)

    def hexbin(self, *args, transform=PlateCarree(), **kwargs):
        """Plot with ax.hexbin(*args, **kwargs)."""
        return self._ax.hexbin(*args, transform=transform, **kwargs)

    def draw_hpxmap(self, hpxmap, nest=False, zoom=True, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Use pcolormesh to draw a healpix map.

        Parameters
        ----------
        """
        nside = hp.npix2nside(hpxmap.size)
        pixels, = np.where(hpxmap != hp.UNSEEN)

        if lon_range is None or lat_range is None:
            _lon_range, _lat_range = healpix_pixels_range(nside,
                                                          pixels,
                                                          self._wrap,
                                                          nest=nest)
            if lon_range is None:
                lon_range = _lon_range
            if lat_range is None:
                lat_range = _lat_range

        # FIXME: add aspect ratio
        lon_raster, lat_raster, values_raster = hpxmap_to_xy(hpxmap,
                                                             lon_range,
                                                             lat_range,
                                                             nest=nest,
                                                             xsize=xsize)

        if vmin is None or vmax is None:
            _vmin, _vmax = np.percentile(hpxmap[pixels], (2.5, 97.5))
            if vmin is None:
                vmin = _vmin
            if vmax is None:
                vmax = _vmax

        if zoom:
            extent = self.compute_extent(lon_raster[:-1, :-1][~values_raster.mask],
                                         lat_raster[:-1, :-1][~values_raster.mask])
            # Okay, it's clear we need a new axis artist ...
            # self.set_axes_limits(extent, invert=False)
            # self.create_axes()
            # self.set_axes_limits(extent, invert=self.do_celestial)
            self.set_extent(extent)

            # self.set_axes_limits(extent)

        im = self.pcolormesh(lon_raster, lat_raster, values_raster, **kwargs)
        self._ax._sci(im)
        return im, lon_raster, lat_raster, values_raster

    def draw_hpxpix(self, nside, pixels, values, nest=False, zoom=True, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Use pcolormesh to draw a healpix map made of pixels and values.

        Parameters
        ----------
        """
        if lon_range is None or lat_range is None:
            _lon_range, _lat_range = healpix_pixels_range(nside,
                                                          pixels,
                                                          self._wrap,
                                                          nest=nest)
            if lon_range is None:
                lon_range = _lon_range
            if lat_range is None:
                lat_range = _lat_range

        # FIXME: add aspect ratio
        lon_raster, lat_raster, values_raster = healpix_to_xy(
            nside,
            pixels,
            values,
            nest=nest,
            xsize=xsize,
            lon_range=lon_range,
            lat_range=lat_range
        )

        if vmin is None or vmax is None:
            _vmin, _vmax = np.percentile(values, (2.5, 97.5))
            if vmin is None:
                vmin = _vmin
            if vmax is None:
                vmax = _vmax

        if zoom:
            extent = self.compute_extent(lon_raster[:-1, :-1][~values_raster.mask],
                                         lat_raster[:-1, :-1][~values_raster.mask])
            self.set_extent(extent)

        im = self.pcolormesh(lon_raster, lat_raster, values_raster, **kwargs)
        self._ax._sci(im)
        return im, lon_raster, lat_raster, values_raster

    def draw_hspmap(self, hspmap, zoom=True, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Use pcolormesh to draw a healsparse map.

        Parameters
        ----------
        """
        valid_pixels = hspmap.valid_pixels

        if lon_range is None or lat_range is None:
            _lon_range, _lat_range = healpix_pixels_range(hspmap.nside_sparse,
                                                          valid_pixels,
                                                          self._wrap,
                                                          nest=True)
            if lon_range is None:
                lon_range = _lon_range
            if lat_range is None:
                lat_range = _lat_range

        # FIXME: add aspect ratio
        lon_raster, lat_raster, values_raster = hspmap_to_xy(hspmap,
                                                             lon_range,
                                                             lat_range,
                                                             xsize=xsize)

        if vmin is None or vmax is None:
            _vmin, _vmax = np.percentile(hspmap[valid_pixels], (2.5, 97.5))
            if vmin is None:
                vmin = _vmin
            if vmax is None:
                vmax = _vmax

        if zoom:
            # Watch for masked array here...
            extent = self.compute_extent(lon_raster[:-1, :-1][~values_raster.mask],
                                         lat_raster[:-1, :-1][~values_raster.mask])
            self.set_extent(extent)

        im = self.pcolormesh(lon_raster, lat_raster, values_raster, **kwargs)
        self._ax._sci(im)
        return im, lon_raster, lat_raster, values_raster

    def draw_inset_colorbar(self, ax=None, format=None, label=None, ticks=None, fontsize=11,
                            width="25%", height="5%", loc=7, bbox_to_anchor=(0., -0.04, 1, 1),
                            orientation='horizontal', **kwargs):
        """Draw an inset colorbar.

        Parameters
        ----------
        """
        if ax is None:
            ax = plt.gca()
        im = plt.gci()
        cax = inset_axes(ax,
                         width=width,
                         height=height,
                         loc=loc,
                         bbox_to_anchor=bbox_to_anchor,
                         bbox_transform=ax.transAxes,
                         **kwargs)
        cmin, cmax = im.get_clim()

        if ticks is None and cmin is not None and cmax is not None:
            cmed = (cmax + cmin)/2.
            delta = (cmax - cmin)/10.
            ticks = np.array([cmin + delta, cmed, cmax - delta])

        tmin = np.min(np.abs(ticks[0]))
        tmax = np.max(np.abs(ticks[1]))

        if format is None:
            if (tmin < 1e-2) or (tmax > 1e3):
                format = '$%.1e$'
            elif (tmin > 0.1) and (tmax < 100):
                format = '$%.1f$'
            elif (tmax > 100):
                format = '$%i$'
            else:
                format = '$%.2g$'

        custom_format = False
        if format == 'custom':
            custom_format = True
            ticks = np.array([cmin, 0.85*cmax])
            format = '$%.0e$'

        cbar = plt.colorbar(cax=cax, orientation=orientation, ticks=ticks, format=format, **kwargs)
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(axis='x', labelsize=fontsize)

        if custom_format:
            ticklabels = cax.get_xticklabels()
            for i, lab in enumerate(ticklabels):
                val, exp = ticklabels[i].get_text().split('e')
                ticklabels[i].set_text(r'$%s \times 10^{%i}$'%(val, int(exp)))
            cax.set_xticklabels(ticklabels)

        if label is not None:
            cbar.set_label(label, size=fontsize)
            cax.xaxis.set_label_position('top')

        plt.sca(ax)

        return cbar, cax


class McBrydeSkymap(Skymap):
    def __init__(self, **kwargs):
        super().__init__(projection_name='mbtfpq', **kwargs)


class OrthoSkymap(Skymap):
    def __init__(self, **kwargs):
        super().__init__(projection_name='ortho', **kwargs)


class MollweideSkymap(Skymap):
    def __init__(self, **kwargs):
        super().__init__(projection_name='moll', **kwargs)


class HammerSkymap(Skymap):
    def __init__(self, **kwargs):
        super().__init__(projection_name='hammer', **kwargs)


class AitoffSkymap(Skymap):
    def __init__(self, **kwargs):
        super().__init__(projection_name='aitoff', **kwargs)
