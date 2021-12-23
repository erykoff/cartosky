import numpy as np

from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
import mpl_toolkits.axisartist.angle_helper as angle_helper

__all__ = ['WrappedFormatterDMS', 'ExtremeFinderWrapped']


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
