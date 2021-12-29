import matplotlib.pyplot as plt
import warnings
import numpy as np
import healpy as hp

from cartopy.crs import PlateCarree
from pyproj import Geod

import mpl_toolkits.axisartist as axisartist
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .projections import get_projection, RADIUS
from .hpx_utils import healpix_pixels_range, hspmap_to_xy, hpxmap_to_xy, healpix_to_xy, healpix_bin
from .mpl_utils import ExtremeFinderWrapped, WrappedFormatterDMS, GridHelperSkymap

__all__ = ['Skymap', 'McBrydeSkymap', 'OrthoSkymap', 'MollweideSkymap',
           'HammerSkymap', 'EqualEarthSkymap']


class Skymap():
    """Base class for creating Skymap objects.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`, optional
        Axis object to replace with a skymap axes
    projection_name : `str`, optional
        Valid proj4/cartosky projection name.
    lon_0 : `float`, optional
        Central longitude of projection.
    gridlines : `bool`, optional
        Draw gridlines?
    celestial : `bool`, optional
        Do celestial plotting (e.g. invert longitude axis).
    extent : iterable, optional
        Default exent of the map, [lon_min, lon_max, lat_min, lat_max].
        Note that lon_min, lon_max can be specified in any order, the
        orientation of the map is set by ``celestial``.
    **kwargs : `dict`, optional
        Additional arguments to send to cartosky/proj4 projection initialization.
    """
    def __init__(self, ax=None, projection_name='cyl', lon_0=0, gridlines=True, celestial=True,
                 extent=None, **kwargs):
        # self.set_observer(kwargs.pop('observer', None))
        # self.set_date(kwargs.pop('date', None))
        self._redraw_dict = {'hpxmap': None,
                             'hspmap': None,
                             'vmin': None,
                             'vmax': None,
                             'xsize': None,
                             'kwargs_pcolormesh': None,
                             'nside': None,
                             'nest': None}

        if ax is None:
            ax = plt.gca()

        fig = ax.figure
        subspec = ax.get_subplotspec()
        fig.delaxes(ax)

        if lon_0 == 180.0:
            # We must move this by epsilon or the code gets confused with 0 == 360
            lon_0 = 179.9999

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
            extent = [lon_0 - 180.0, lon_0 + 180.0, -90.0, 90.0]

        self.set_extent(extent)

    def proj(self, lon, lat):
        """Apply forward projection to a set of lon/lat positions.

        Convert from lon/lat to x/y.

        Parameters
        ----------
        lon : `float` or `list` or `np.ndarray`
            Array of longitude(s) (degrees).
        lat : `float` or `list` or `np.ndarray`
            Array of latitudes(s) (degrees).

        Returns
        -------
        x : `np.ndarray`
            Array of x values.
        y : `np.ndarray`
            Array of y values.
        """
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)
        proj_xyz = self.projection.transform_points(PlateCarree(), lon, lat)
        return proj_xyz[..., 0], proj_xyz[..., 1]

    def proj_inverse(self, x, y):
        """Apply inverse projection to a set of points.

        Convert from x/y to lon/lat.

        Parameters
        ----------
        x : `float` or `list` or `np.ndarray`
            Projected x values.
        y : `float` or `list` or `np.ndarray`
            Projected y values.

        Returns
        -------
        lon : `np.ndarray`
            Array of longitudes (degrees).
        lat : `np.ndarray`
            Array of latitudes (degrees).
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        proj_xyz = PlateCarree().transform_points(self.projection, x, y)
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
        self._create_axes(extent)
        self._set_axes_limits(extent, invert=self.do_celestial)

        self._ax.set_frame_on(False)
        if self.do_gridlines:
            self._aa.grid(True, linestyle=':', color='k', lw=0.5)

        self._extent = extent
        self._changed_x_axis = False
        self._changed_y_axis = False

    def get_extent(self):
        """Get the extent in lon/lat coordinates.

        Returns
        -------
        extent : `list`
            Extent as [lon_min, lon_max, lat_min, lat_max].
        """
        return self._extent

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
            lon_steps = extent[2:]
            if extent[2] < 0 and extent[3] > 0:
                lon_steps.append(0.0)

            lon, lat = np.meshgrid(
                np.arange(360.0),
                lon_steps
            )
            x, y = self.proj(lon.ravel(), lat.ravel())
            # Need to offset this by some small amount to ensure we don't get
            # out-of-bounds transformations.
            extent_xform = [0.9999*np.max(x),
                            0.9999*np.min(x),
                            0.9999*np.min(y),
                            0.9999*np.max(y)]
            self._ax.set_extent(extent_xform, crs=self.projection)
        else:
            self._ax.set_extent(extent, crs=PlateCarree())

        extreme_finder = ExtremeFinderWrapped(20, 20, self._wrap)
        extent_xform = self._ax.get_extent()
        lon_min, lon_max, lat_min, lat_max = extreme_finder(self.proj_inverse,
                                                            extent_xform[0],
                                                            extent_xform[2],
                                                            extent_xform[1],
                                                            extent_xform[3])

        # Draw the outer edges of the projection.  This needs to be forward-
        # projected and drawn in that space to prevent out-of-bounds clipping.
        # It also needs to be done just inside -180/180 to prevent the transform
        # from resolving to the same line.
        x, y = self.proj(np.linspace(self._lon_0 - 179.9999, self._lon_0 - 179.9999),
                         np.linspace(-90., 90.))
        self.plot(x, y, 'k-', transform=self.projection)
        x, y = self.proj(np.linspace(self._lon_0 + 179.9999, self._lon_0 + 179.9999),
                         np.linspace(-90., 90.))
        self.plot(x, y, 'k-', transform=self.projection)

        if self._aa is not None:
            self._aa.set_xlim(self._ax.get_xlim())
            self._aa.set_ylim(self._ax.get_ylim())

        if invert:
            self._ax.invert_xaxis()
            if self._aa is not None:
                self._aa.invert_xaxis()

        return self._ax.get_xlim(), self._ax.get_ylim()

    def _create_axes(self, extent):
        """Create axes and axis artist.

        Parameters
        ----------
        extent : `list`
            Axis extent [lon_min, lon_max, lat_min, lat_max] (degrees).
        """
        extreme_finder = ExtremeFinderWrapped(20, 20, self._wrap)
        grid_locator1 = angle_helper.LocatorD(10, include_last=True)
        grid_locator2 = angle_helper.LocatorD(6, include_last=True)

        # We always want the formatting to be wrapped at 180 (-180 to 180)
        tick_formatter1 = WrappedFormatterDMS(180.0)
        tick_formatter2 = angle_helper.FormatterDMS()

        grid_helper = GridHelperSkymap(
            (self.proj, self.proj_inverse),
            extreme_finder=extreme_finder,
            grid_locator1=grid_locator1,
            grid_locator2=grid_locator2,
            tick_formatter1=tick_formatter1,
            tick_formatter2=tick_formatter2
        )

        self._grid_helper = grid_helper

        fig = self._ax.figure
        rect = self._ax.get_position()
        self._aa = axisartist.Axes(fig, rect, grid_helper=grid_helper, frameon=False, aspect=1.0)
        fig.add_axes(self._aa)

        self._aa.format_coord = self._format_coord
        self._aa.axis['left'].major_ticklabels.set_visible(True)
        self._aa.axis['right'].major_ticklabels.set_visible(False)
        self._aa.axis['bottom'].major_ticklabels.set_visible(True)
        self._aa.axis['top'].major_ticklabels.set_visible(True)

        self.set_xlabel('Right Ascension', size=16)
        self.set_ylabel('Declination', size=16)

        fig.sca(self._ax)

        return fig, self._ax

    def _format_coord(self, x, y):
        """Return a coordinate format string.

        Parameters
        ----------
        x : `float`
            x position in projected coords.
        y : `float`
            y position in projected coords.

        Returns
        -------
        coord_string : `str`
            Formatted string.
        """
        lon, lat = self.proj_inverse(x, y)
        # FIXME: check for out-of-bounds here?
        coord_string = 'lon=%.6f, lat=%.6f' % (lon, lat)
        if np.isnan(lon) or np.isnan(lat):
            val = hp.UNSEEN
        elif self._redraw_dict['hspmap'] is not None:
            val = self._redraw_dict['hspmap'].get_values_pos(lon, lat)
        elif self._redraw_dict['hpxmap'] is not None:
            pix = hp.ang2pix(self._redraw_dict['nside'],
                             lon,
                             lat,
                             lonlat=True,
                             nest=self._redraw_dict['nest'])
            val = self._redraw_dict['hpxmap'][pix]
        else:
            return coord_string

        if np.isclose(val, hp.UNSEEN):
            coord_string += ', val=UNSEEN'
        else:
            coord_string += ', val=%f' % (val)
        return coord_string

    def _change_axis(self, ax):
        """Callback for axis change.

        Parameters
        ----------
        ax : `cartopy.mpl.geoaxes.GeoAxesSubplot`
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            extent = ax.get_extent(crs=PlateCarree())
        if not np.isclose(extent[0], self._extent[0]) or not np.isclose(extent[1], self._extent[1]):
            self._changed_x_axis = True
        if not np.isclose(extent[2], self._extent[2]) or not np.isclose(extent[3], self._extent[3]):
            self._changed_y_axis = True

        if not self._changed_x_axis or not self._changed_y_axis:
            # Nothing to do yet.
            return

        # Reset to new extent
        self._changed_x_axis = False
        self._changed_y_axis = False
        self._extent = extent

        lon_range = [extent[0], extent[1]]
        lat_range = [extent[2], extent[3]]

        if self._redraw_dict['hpxmap'] is not None:
            lon_raster, lat_raster, values_raster = hpxmap_to_xy(self._redraw_dict['hpxmap'],
                                                                 lon_range,
                                                                 lat_range,
                                                                 nest=self._redraw_dict['nest'],
                                                                 xsize=self._redraw_dict['xsize'])
        elif self._redraw_dict['hspmap'] is not None:
            lon_raster, lat_raster, values_raster = hspmap_to_xy(self._redraw_dict['hspmap'],
                                                                 lon_range,
                                                                 lat_range,
                                                                 xsize=self._redraw_dict['xsize'])
        else:
            # Nothing to do
            return

        im = self.pcolormesh(lon_raster, lat_raster, values_raster,
                             vmin=self._redraw_dict['vmin'],
                             vmax=self._redraw_dict['vmax'],
                             **self._redraw_dict['kwargs_pcolormesh'])
        self._ax._sci(im)

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
        # FIXME: do we want to set the extent automatically here?
        return self._ax.hexbin(*args, transform=transform, **kwargs)

    def legend(self, *args, loc='upper left', **kwargs):
        """Add legend to the axis with ax.legend(*args, **kwargs)."""
        return self._ax.legend(*args, loc='upper left', **kwargs)

    def draw_line_lonlat(self, lon, lat,
                         color='black', linestyle='solid', nsamp=100,
                         **kwargs):
        """Draw a line assuming a Geodetic transform.

        Parameters
        ----------
        lon : `np.ndarray`
            Array of longitude points in the line segments.
        lat : `np.ndarray`
            Array of latitude points in the line segments.
        color : `str`, optional
            Color of line segments.
        linestyle : `str`, optional
            Line style for segments.
        nsamp : `int`, optional
            Number of samples for each line segment.
        label : `str`, optional
            Legend label string.
        **kwargs : `dict`
            Additional keywords passed to plot.
        """
        _lon = np.atleast_1d(lon)
        _lat = np.atleast_1d(lat)
        g = Geod(a=RADIUS)
        for i in range(len(_lon) - 1):
            lonlats = np.array(g.npts(_lon[i], _lat[i], _lon[i + 1], _lat[i + 1], nsamp,
                                      initial_idx=0, terminus_idx=0))
            # Check for lines that wrap around and clip these in two...
            lon_test = (lonlats[:, 0] + 180.) % 360. - 180.
            delta = lon_test[: -1] - lon_test[1:]
            cut, = np.where(delta > 180.0)
            if cut.size == 0:
                # No wrap
                self.plot(lonlats[:, 0], lonlats[:, 1], color=color, linestyle=linestyle,
                          **kwargs)
                # Only add label to first line segment.
                kwargs.pop('label', None)
            else:
                # We have a wrap
                cut = cut[0] + 1
                self.plot(lonlats[0: cut, 0], lonlats[0: cut, 1], color=color, linestyle=linestyle,
                          **kwargs)
                # Only add label to first line segment.
                kwargs.pop('label', None)
                self.plot(lonlats[cut:, 0], lonlats[cut:, 1], color=color, linestyle=linestyle,
                          **kwargs)

    def draw_polygon_lonlat(self, lon, lat, color='red', linestyle='solid',
                            **kwargs):
        """Plot a polygon from a list of lon, lat coordinates.

        Parameters
        ----------
        lon : `np.ndarray`
            Array of longitude points in polygon.
        lat : `np.ndarray`
            Array of latitude points in polygon.
        color : `str`, optional
            Color of polygon boundary.
        linestyle : `str`, optional
            Line style for boundary.
        label : `str`, optional
            Legend label string.
        **kwargs : `dict`, optional
            Additional keywords passed to plot.
        """
        lon = np.atleast_1d(lon).ravel()
        lat = np.atleast_1d(lat).ravel()
        # Add the first point at the end to ensure a closed polygon.
        lon = np.append(lon, lon[0])
        lat = np.append(lat, lat[0])

        self.draw_line_lonlat(lon, lat, color=color, linestyle=linestyle, **kwargs)

    def draw_polygon_file(self, filename, reverse=True,
                          color='red', linestyle='solid', **kwargs):
        """Draw a text file containing lon, lat coordinates of polygon(s).

        Parameters
        ----------
        filename : `str`
            Name of file containing the polygon(s) [lon, lat, poly]
        reverse : `bool`
            Reverse drawing order of points in each polygon.
        color : `str`
            Color of polygon boundary.
        linestyle : `str`, optional
            Line style for boundary.
        **kwargs : `dict`
            Additional keywords passed to plot.
        """
        try:
            data = np.genfromtxt(filename, names=['lon', 'lat', 'poly'])
        except ValueError:
            from numpy.lib.recfunctions import append_fields
            data = np.genfromtxt(filename, names=['lon', 'lat'])
            data = append_fields(data, 'poly', np.zeros(len(data)))

        for p in np.unique(data['poly']):
            poly = data[data['poly'] == p]
            lon = poly['lon'][::-1] if reverse else poly['lon']
            lat = poly['lat'][::-1] if reverse else poly['lat']
            self.draw_polygon_lonlat(lon,
                                     lat,
                                     color=color,
                                     linestyle=linestyle,
                                     **kwargs)
            # Only add the label to the first polygon plotting.
            kwargs.pop('label', None)

    def draw_hpxmap(self, hpxmap, nest=False, zoom=True, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Use pcolormesh to draw a healpix map.

        Parameters
        ----------
        hpxmap : `np.ndarray`
            Healpix map to plot, with length 12*nside*nside and UNSEEN for
            illegal values.
        nest : `bool`, optional
            Map in nest ordering?
        zoom : `bool`, optional
            Optimally zoom in projection to computed map.
        xsize : `int`, optional
            Number of rasterized pixels in the x direction.
        vmin : `float`, optional
            Minimum value for color scale.  Defaults to 2.5th percentile.
        vmax : `float`, optional
            Maximum value for color scale.  Defaults to 97.5th percentile.
        rasterized : `bool`, optional
            Plot with rasterized graphics.
        lon_range : `tuple` [`float`, `float`], optional
            Longitude range to plot [``lon_min``, ``lon_max``].
        lat_range : `tuple` [`float`, `float`], optional
            Latitude range to plot [``lat_min``, ``lat_max``].
        **kwargs : `dict`
            Additional args to pass to pcolormesh.

        Returns
        -------
        im : `cartopy.mpl.geocollection.GeoQuadMesh`
            Image that was displayed
        lon_raster : `np.ndarray`
            2D array of rasterized longitude values.
        lat_raster : `np.ndarray`
            2D array of rasterized latitude values.
        values_raster : `np.ma.MaskedArray`
            Masked array of rasterized values.
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
            self.set_extent(extent)

        im = self.pcolormesh(lon_raster, lat_raster, values_raster, vmin=vmin, vmax=vmax, **kwargs)
        self._ax._sci(im)

        # Link up callbacks
        self._xlc = self._ax.callbacks.connect('xlim_changed', self._change_axis)
        self._ylc = self._ax.callbacks.connect('ylim_changed', self._change_axis)
        self._redraw_dict['hspmap'] = None
        self._redraw_dict['hpxmap'] = hpxmap
        self._redraw_dict['nside'] = nside
        self._redraw_dict['nest'] = nest
        self._redraw_dict['vmin'] = vmin
        self._redraw_dict['vmax'] = vmax
        self._redraw_dict['xsize'] = xsize
        self._redraw_dict['kwargs_pcolormesh'] = kwargs

        return im, lon_raster, lat_raster, values_raster

    def draw_hpxpix(self, nside, pixels, values, nest=False, zoom=True, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Use pcolormesh to draw a healpix map made of pixels and values.

        Parameters
        ----------
        nside : `int`
            Healpix nside of pixels to plot.
        pixels : `np.ndarray`
            Array of pixels to plot.
        values : `np.ndarray`
            Array of values associated with pixels.
        nest : `bool`, optional
            Map in nest ordering?
        zoom : `bool`, optional
            Optimally zoom in projection to computed map.
        xsize : `int`, optional
            Number of rasterized pixels in the x direction.
        vmin : `float`, optional
            Minimum value for color scale.  Defaults to 2.5th percentile.
        vmax : `float`, optional
            Maximum value for color scale.  Defaults to 97.5th percentile.
        rasterized : `bool`, optional
            Plot with rasterized graphics.
        lon_range : `tuple` [`float`, `float`], optional
            Longitude range to plot [``lon_min``, ``lon_max``].
        lat_range : `tuple` [`float`, `float`], optional
            Latitude range to plot [``lat_min``, ``lat_max``].
        **kwargs : `dict`
            Additional args to pass to pcolormesh.

        Returns
        -------
        im : `cartopy.mpl.geocollection.GeoQuadMesh`
            Image that was displayed
        lon_raster : `np.ndarray`
            2D array of rasterized longitude values.
        lat_raster : `np.ndarray`
            2D array of rasterized latitude values.
        values_raster : `np.ma.MaskedArray`
            Masked array of rasterized values.
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

        im = self.pcolormesh(lon_raster, lat_raster, values_raster, vmin=vmin, vmax=vmax, **kwargs)
        self._ax._sci(im)
        return im, lon_raster, lat_raster, values_raster

    def draw_hspmap(self, hspmap, zoom=True, xsize=1000, vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Use pcolormesh to draw a healsparse map.

        Parameters
        ----------
        hspmap : `healsparse.HealSparseMap`
            Healsparse map to plot.
        zoom : `bool`, optional
            Optimally zoom in projection to computed map.
        xsize : `int`, optional
            Number of rasterized pixels in the x direction.
        vmin : `float`, optional
            Minimum value for color scale.  Defaults to 2.5th percentile.
        vmax : `float`, optional
            Maximum value for color scale.  Defaults to 97.5th percentile.
        rasterized : `bool`, optional
            Plot with rasterized graphics.
        lon_range : `tuple` [`float`, `float`], optional
            Longitude range to plot [``lon_min``, ``lon_max``].
        lat_range : `tuple` [`float`, `float`], optional
            Latitude range to plot [``lat_min``, ``lat_max``].
        **kwargs : `dict`
            Additional args to pass to pcolormesh.

        Returns
        -------
        im : `cartopy.mpl.geocollection.GeoQuadMesh`
            Image that was displayed
        lon_raster : `np.ndarray`
            2D array of rasterized longitude values.
        lat_raster : `np.ndarray`
            2D array of rasterized latitude values.
        values_raster : `np.ma.MaskedArray`
            Masked array of rasterized values.
        """
        self._hspmap = hspmap
        self._hpxmap = None

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

        im = self.pcolormesh(lon_raster, lat_raster, values_raster, vmin=vmin, vmax=vmax, **kwargs)
        self._ax._sci(im)

        # Link up callbacks
        self._xlc = self._ax.callbacks.connect('xlim_changed', self._change_axis)
        self._ylc = self._ax.callbacks.connect('ylim_changed', self._change_axis)
        self._redraw_dict['hspmap'] = hspmap
        self._redraw_dict['hpxmap'] = None
        self._redraw_dict['vmin'] = vmin
        self._redraw_dict['vmax'] = vmax
        self._redraw_dict['xsize'] = xsize
        self._redraw_dict['kwargs_pcolormesh'] = kwargs

        return im, lon_raster, lat_raster, values_raster

    def draw_hpxbin(self, lon, lat, C=None, nside=256, nest=False, zoom=True, xsize=1000,
                    vmin=None, vmax=None,
                    rasterized=True, lon_range=None, lat_range=None, **kwargs):
        """Create a healpix histogram of counts in lon, lat.

        Related to ``hexbin`` from matplotlib.

        If ``C`` array is specified then the mean is taken from the C values.

        Parameters
        ----------
        lon : `np.ndarray`
            Array of longitude values.
        lat : `np.ndarray`
            Array of latitude values.
        C : `np.ndarray`, optional
            Array of values to average in each pixel.
        nside : `int`, optional
            Healpix nside resolution.
        nest : `bool`, optional
            Compute map in nest ordering?
        zoom : `bool`, optional
            Optimally zoom in projection to computed map.
        xsize : `int`, optional
            Number of rasterized pixels in the x direction.
        vmin : `float`, optional
            Minimum value for color scale.  Defaults to 2.5th percentile.
        vmax : `float`, optional
            Maximum value for color scale.  Defaults to 97.5th percentile.
        rasterized : `bool`, optional
            Plot with rasterized graphics.
        lon_range : `tuple` [`float`, `float`], optional
            Longitude range to plot [``lon_min``, ``lon_max``].
        lat_range : `tuple` [`float`, `float`], optional
            Latitude range to plot [``lat_min``, ``lat_max``].
        **kwargs : `dict`
            Additional args to pass to pcolormesh.

        Returns
        -------
        hpxmap : `np.ndarray`
            Computed healpix map.
        im : `cartopy.mpl.geocollection.GeoQuadMesh`
            Image that was displayed.
        """
        hpxmap = healpix_bin(lon, lat, C=C, nside=nside, nest=nest)

        return self.draw_hpxmap(hpxmap, nest=nest, zoom=zoom, xsize=xsize, vmin=vmin,
                                vmax=vmax, rasterized=rasterized, lon_range=lon_range,
                                lat_range=lat_range, **kwargs)

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


# The following skymaps include the equal-area projections that are tested
# and known to work.

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


class EqualEarthSkymap(Skymap):
    def __init__(self, **kwargs):
        super().__init__(projection_name='eqearth', **kwargs)
