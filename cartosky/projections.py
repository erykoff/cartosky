import warnings

import cartopy.crs
from cartopy.crs import _WarpedRectangularProjection, Globe
from .utils import setdefaults

# Default radius of the Globe.
# Eventually should be possible to use the unit sphere.
RADIUS = 6378137.0  # meters

__all__ = ['Aitoff', 'Hammer', 'McBrydeThomasFlatPolarQuartic', 'KavrayskiyVII',
           'get_projection', 'get_available_projections', 'SkySphere']


class SkySphere(Globe):
    """
    Spherical ccrs.Globe for sky plotting.
    """

    def __init__(self, *args, **kwargs):
        defaults = dict(ellipse=None, semimajor_axis=RADIUS, semiminor_axis=RADIUS)
        kwargs = setdefaults(kwargs, defaults)
        super().__init__(*args, **kwargs)


class Aitoff(_WarpedRectangularProjection):
    """
    The `Aitoff <https://en.wikipedia.org/wiki/Aitoff_projection>`__
    projection.
    """
    # Registered projection name.
    name = 'aitoff'

    def __init__(
        self, central_longitude=0, central_latitude=0, globe=None,
        false_easting=None, false_northing=None
    ):
        if globe is None:
            globe = SkySphere()

        if globe.ellipse is not None:
            warnings.warn(
                'The {} projection does not handle elliptical globes.'.format(self.name)
            )

        proj4_params = {'proj': 'aitoff', 'lon_0': central_longitude,
                        'lat_0': central_latitude}
        super().__init__(
            proj4_params, central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
            globe=globe
        )

    @property
    def threshold(self):  # how finely to interpolate line data, etc.
        return 1e4


class Hammer(_WarpedRectangularProjection):
    """
    The `Hammer <https://en.wikipedia.org/wiki/Hammer_projection>`__
    projection.
    """
    # Registered projection name.
    name = 'hammer'

    def __init__(
        self, central_longitude=0, central_latitude=0, globe=None,
        false_easting=None, false_northing=None
    ):
        if globe is None:
            globe = SkySphere()

        if globe.ellipse is not None:
            warnings.warn(
                f'The {self.name!r} projection does not handle '
                'elliptical globes.'
            )

        proj4_params = {'proj': 'hammer', 'lon_0': central_longitude,
                        'lat_0': central_latitude}
        super().__init__(
            proj4_params, central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
            globe=globe
        )

    @property
    def threshold(self):  # how finely to interpolate line data, etc.
        return 1e4


class McBrydeThomasFlatPolarQuartic(_WarpedRectangularProjection):
    """
    The `McBryde-Thomas Flat Polar Quartic
    <https://it.wikipedia.org/wiki/File:McBryde-Thomas_flat-pole_quartic_projection_SW.jpg>`__
    projection.
    """
    # Registered projection name.
    name = 'mbtfpq'

    def __init__(
        self, central_longitude=0, central_latitude=0, globe=None,
        false_easting=None, false_northing=None
    ):
        if globe is None:
            globe = SkySphere()

        if globe.ellipse is not None:
            warnings.warn(
                f'The {self.name!r} projection does not handle '
                'elliptical globes.'
            )

        proj4_params = {'proj': 'mbtfpq', 'lon_0': central_longitude,
                        'lat_0': central_latitude}
        super().__init__(
            proj4_params, central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
            globe=globe
        )

    @property
    def threshold(self):  # how finely to interpolate line data, etc.
        return 1e4


class KavrayskiyVII(_WarpedRectangularProjection):
    """
    The `Kavrayskiy VII \
<https://en.wikipedia.org/wiki/Kavrayskiy_VII_projection>`__ projection.
    """
    #: Registered projection name.
    name = 'kavrayskiyVII'

    def __init__(
        self, central_longitude=0, central_latitude=None, globe=None,
        false_easting=None, false_northing=None
    ):
        if globe is None:
            globe = SkySphere()

        if globe.ellipse is not None:
            warnings.warn(
                f'The {self.name!r} projection does not handle '
                'elliptical globes.'
            )
        if central_latitude is not None:
            warnings.warn(f'The {self.name!r} projection ignores central_latitude parameter.')

        proj4_params = {'proj': 'kav7', 'lon_0': central_longitude}
        super().__init__(
            proj4_params, central_longitude,
            false_easting=false_easting,
            false_northing=false_northing,
            globe=globe
        )

    @property
    def threshold(self):
        return 1e5


_projections = {
    # cartosky implementations from proj4
    'aitoff': ('Aitoff', Aitoff),
    'hammer': ('Hammer', Hammer),
    'mbtfpq': ('McBryde-Thomas Flat Polar Quartic', McBrydeThomasFlatPolarQuartic),
    'kav7': ('Kavrayskiy-VII', KavrayskiyVII),
    # cartopy implementations from proj4
    'aea': ('Albers Equal Area', 'AlbersEqualArea'),
    'aeqd': ('Azimuthal Equidistant', 'AzimuthalEquidistant'),
    'cyl': ('Plate Carree', 'PlateCarree'),
    'eck1': ('Eckert-I', 'EckertI'),
    'eck2': ('Eckert-II', 'EckertII'),
    'eck3': ('Eckert-III', 'EckertIII'),
    'eck4': ('Eckert-IV', 'EckertIV'),
    'eck5': ('Eckert-V', 'EckertV'),
    'eck6': ('Eckert-VI', 'EckertVI'),
    'eqc': ('Plate Carree', 'PlateCarree'),
    'eqdc': ('Equidistant Conic', 'EquidistantConic'),
    'eqearth': ('Equal Earth', 'EqualEarth'),
    'euro': ('UTM Zone 32 projection for EuroPP domain', 'EuroPP'),
    'geos': ('Geostationary', 'Geostationary'),
    'gnom': ('Gnominic', 'Gnomonic'),
    'igh': ('Interrupted Goode Homolosine', 'InterruptedGoodeHomolosine'),
    'laea': ('Lambert Azimuthal Equal Area', 'LambertAzimuthalEqualArea'),
    'lcc': ('Lambert Conformal', 'LambertConformal'),
    'lcyl': ('Lambert Cylindrical', 'LambertCylindrical'),
    'merc': ('Mercator', 'Mercator'),
    'mill': ('Miller', 'Miller'),
    'moll': ('Mollweide', 'Mollweide'),
    'npstere': ('North Polar Stereo', 'NorthPolarStereo'),
    'nsper': ('Nearside Perspective', 'NearsidePerspective'),
    'ortho': ('Orthographic', 'Orthographic'),
    'osgb': ('Ordinance Survey of Great Britain (UK only)', 'OSGB'),
    'osni': ('Ordinance Survey of Northern Ireland (Ireland only)', 'OSNI'),
    'pcarree': ('Plate Carree', 'PlateCarree'),
    'robin': ('Robinson', 'Robinson'),
    'rotpole': ('Rotated Pole', 'RotatedPole'),
    'sinu': ('Sinusoidal', 'Sinusoidal'),
    'spstere': ('South Polar Stereo', 'SouthPolarStereo'),
    'stere': ('Stereographic', 'Stereographic'),
    'tmerc': ('Transverse Mercator', 'TransverseMercator'),
    'utm': ('Universal Transverse Mercator', 'UTM'),
}


def get_projection(name, **kwargs):
    """Return a cartosky projection.

    For list of projections available, use cartosky.get_available_projections().

    Parameters
    ----------
    name : str
        Cartosky name of projection.
    **kwargs :
        Additional kwargs appropriate for given projection.

    Returns
    -------
    proj : `cartopy.crs.Projection`
    """
    # Define keyword aliases to allow proj4 shorthand names instead
    # of verbose cartopy names.
    aliases = {
        'lat_0': 'central_latitude',
        'lon_0': 'central_longitude',
        'lat_min': 'min_latitude',
        'lat_max': 'max_latitude',
    }
    kwproj = {aliases.get(key, key): value for key, value in kwargs.items()}
    # Set the default globe
    kwproj = setdefaults(kwproj,
                         {'globe': SkySphere()})

    # Is this a listed projection?
    if name not in _projections:
        raise ValueError(f'{name} projection name is not recognized.  See get_available_projections()')

    descr, projtype = _projections[name]
    if isinstance(projtype, str):
        # This is a cartopy projection
        if not hasattr(cartopy.crs, projtype):
            raise ValueError(f'{name} projection is not available in version of cartopy. '
                             'Consider updating.')
        crs = getattr(cartopy.crs, projtype)
    else:
        # This is a cartosky projection direct from proj.
        crs = projtype

    return crs(**kwproj)


def get_available_projections():
    """Return dict of available projections.

    Returns
    -------
    available_projections: `dict`
        Available projections.  Key is cartosky name, value is brief description.
    """
    available_projections = {}
    for name, (descr, projtype) in _projections.items():
        if isinstance(projtype, str):
            if hasattr(cartopy.crs, projtype):
                available_projections[name] = descr
        else:
            available_projections[name] = descr

    return available_projections
