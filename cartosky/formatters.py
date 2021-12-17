import numpy as np
import mpl_toolkits.axisartist.angle_helper as angle_helper

__all__ = ['WrappedFormatterDMS']


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
