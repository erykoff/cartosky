import numpy as np

from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

__all__ = ['WrappedFormatterDMS', 'ExtremeFinderWrapped', 'GridHelperSkymap']


class WrappedFormatterDMS(angle_helper.FormatterDMS):
    def __init__(self, wrap):
        self._wrap = wrap
        super().__init__()

    def _wrap_values(self, factor, values):
        """Wrap the values according to the wrap angle.

        Parameters
        ----------
        factor : `float`
            Scaling factor for input values
        values : `list`
            List of values to format

        Returns
        -------
        wrapped_values : `np.ndarray`
            Array of wrapped values, scaled by factor.
        """
        _values = np.atleast_1d(values)/factor
        return factor*((_values + self._wrap) % 360 - self._wrap)

    def __call__(self, direction, factor, values):
        return super().__call__(direction, factor, self._wrap_values(factor, values))


class ExtremeFinderWrapped(ExtremeFinderSimple):
    # docstring inherited

    def __init__(self, nx, ny, wrap_angle):
        """
        Find extremes with configurable wrap angle and correct limits.

        Parameters
        ----------
        nx : `int`
            Number of samples in x direction.
        ny : `int`
            Number of samples in y direction.
        wrap_angle : `float`
            Angle at which the 360-degree cycle should be wrapped.
        """
        self.nx, self.ny = nx, ny
        self._wrap = wrap_angle

    def __call__(self, transform_xy, x1, y1, x2, y2):
        # docstring inherited
        x, y = np.meshgrid(
            np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        lon, lat = transform_xy(np.ravel(x), np.ravel(y))

        with np.errstate(invalid='ignore'):
            lon = (lon + self._wrap) % 360. - self._wrap

        lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
        lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)

        lon_min, lon_max, lat_min, lat_max = \
            self._add_pad(lon_min, lon_max, lat_min, lat_max)

        lat_min = np.clip(lat_min, -90.0, 90.0)
        lat_max = np.clip(lat_max, -90.0, 90.0)

        lon_min = np.clip(lon_min, -self._wrap, self._wrap)
        lon_max = np.clip(lon_max, -self._wrap, self._wrap)

        return lon_min, lon_max, lat_min, lat_max


class GridHelperSkymap(GridHelperCurveLinear):
    """GridHelperCurveLinear with tick overlap protection.
    """
    def __init__(self, *args, extent_xy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extent_xy = extent_xy

    def get_tick_iterator(self, nth_coord, axis_side, minor=False):

        try:
            _grid_info = self._grid_info
        except AttributeError:
            _grid_info = self.grid_info

        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[axis_side]
        lon_or_lat = ["lon", "lat"][nth_coord]
        if lon_or_lat == "lon":
            delta_x = abs(self._extent_xy[1] - self._extent_xy[0])
        if not minor:  # major ticks
            prev_xy = None
            for ctr, ((xy, a), l) in enumerate(zip(
                    _grid_info[lon_or_lat]["tick_locs"][axis_side],
                    _grid_info[lon_or_lat]["tick_labels"][axis_side])):
                angle_normal = a

                if ctr > 0 and lon_or_lat == 'lon':
                    # Check if this is too close to the last label.
                    if abs(xy[0] - prev_xy[0])/delta_x < 0.05:
                        continue
                prev_xy = xy
                yield xy, angle_normal, angle_tangent, l
        else:
            for (xy, a), l in zip(
                    _grid_info[lon_or_lat]["tick_locs"][axis_side],
                    _grid_info[lon_or_lat]["tick_labels"][axis_side]):
                angle_normal = a
                yield xy, angle_normal, angle_tangent, ""
