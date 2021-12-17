import numpy as np
import healpy as hp

__all__ = ['healpix_pixels_range', 'hspmap_to_xy', 'hpxmap_to_xy', 'healpix_to_xy']


def healpix_pixels_range(nside, pixels, wrap, nest=False):
    """Find lon/lat range of healpix pixels, using wrap angle.

    Parameters
    ----------
    nside : `int`
        Healpix nside
    pixels : `np.ndarray`
        Array of pixel numbers
    wrap : `float`
        Wrap angle.
    nest : `bool`, optional
        Nest ordering?

    Returns
    -------
    lon_range : `tuple` [`float`, `float`]
        Longitude range of pixels (min, max)
    lat_range : `tuple` [`float`, `float`]
        Latitude range of pixels (min, max)
    """
    lon, lat = hp.pix2ang(nside, pixels, nest=nest, lonlat=True)

    eps = hp.max_pixrad(nside, degrees=True)

    lat_range = (np.clip(np.min(lat) - eps, -90.0, None),
                 np.clip(np.max(lat) + eps, None, 90.0))

    lon_wrap = (lon + wrap) % 360 - wrap

    lon_0 = (wrap + 180.0) % 360.0

    lon_range = (np.clip(np.min(lon_wrap) - eps, lon_0 - 180.0, None),
                 np.clip(np.max(lon_wrap) + eps, None, lon_0 + 180.0))

    return lon_range, lat_range


def hspmap_to_xy(hspmap, lon_range, lat_range, xsize=1000, aspect=1.0):
    """Convert healsparse map to rasterized x/y positions and values.

    Parameters
    ----------
    hspmap : `healsparse.HealSparseMap`
        Healsparse map
    lon_range : `tuple` [`float`, `float`]
        Longitude range for rasterization, (min, max).
    lat_range : `tuple` [`float`, `float`]
        Latitude range for rasterization, (min, max).
    xsize : `int`, optional
        Number of rasterized pixels in the x direction.
    aspect : `float`, optional
        Aspect ratio for ysize.

    Returns
    -------
    lon_raster : `np.ndarray`
        Rasterized longitude values (length xsize).
    lat_raster : `np.ndarray`
        Rasterized latitude values (length xsize*aspect).
    values_raster : `np.ma.maskedarray`
        Rasterized values (2-d).  Invalid values are masked.
    """
    lon_raster, lat_raster = np.meshgrid(np.linspace(lon_range[0], lon_range[1], xsize),
                                         np.linspace(lat_range[0], lat_range[1], int(aspect*xsize)))

    # For pcolormesh we need the central locations
    clon = (lon_raster[1:, 1:] + lon_raster[:-1, :-1])/2.
    clat = (lat_raster[1:, 1:] + lat_raster[:-1, :-1])/2.

    values = hspmap.get_values_pos(clon, clat)

    mask = np.isclose(values, hspmap._sentinel)

    return lon_raster, lat_raster, np.ma.array(values, mask=mask)


def hpxmap_to_xy(hpxmap, lon_range, lat_range, nest=False, xsize=1000, aspect=1.0):
    """Convert healpix map to rasterized x/y positions and values.

    Parameters
    ----------
    hpxmap : `np.ndarray`
        Healpix map
        lon_range : `tuple` [`float`, `float`]
        Longitude range for rasterization, (min, max).
    lat_range : `tuple` [`float`, `float`]
        Latitude range for rasterization, (min, max).
    xsize : `int`, optional
        Number of rasterized pixels in the x direction.
    aspect : `float`, optional
        Aspect ratio for ysize.

    Returns
    -------
    lon_raster : `np.ndarray`
        Rasterized longitude values (length xsize).
    lat_raster : `np.ndarray`
        Rasterized latitude values (length xsize*aspect).
    values_raster : `np.ma.maskedarray`
        Rasterized values (2-d).  Invalid values are masked.
    """
    lon_raster, lat_raster = np.meshgrid(np.linspace(lon_range[0], lon_range[1], xsize),
                                         np.linspace(lat_range[0], lat_range[1], int(aspect*xsize)))

    # For pcolormesh we need the central locations
    clon = (lon_raster[1:, 1:] + lon_raster[:-1, :-1])/2.
    clat = (lat_raster[1:, 1:] + lat_raster[:-1, :-1])/2.

    pix_raster = hp.ang2pix(hp.npix2nside(hpxmap.size), clon, clat, nest=nest, lonlat=True)
    values = hpxmap[pix_raster]

    mask = np.isclose(values, hp.UNSEEN)

    return lon_raster, lat_raster, np.ma.array(values, mask=mask)


def healpix_to_xy(nside, pixels, values, lon_range, lat_range,
                  nest=False, xsize=1000, aspect=1.0):
    """Convert healpix pixels to rasterized x/y positions and values.

    Parameters
    ----------
    nside : `int`
        Healpix nside.
    pixels : `np.ndarray`
        Array of pixel numbers
    values : `np.ndarray`
        Array of pixel values
    lon_range : `tuple`, optional
        Longitude range to do rasterization (min, max).
    lat_range : `tuple`, optional
        Latitude range to do rasterization (min, max).
    nest : `bool`, optional
        Nest ordering?
    xsize : `int`, optional
        Number of rasterized pixels in the x direction.
    aspect : `float`, optional
        Aspect ratio to compute ysize.

    Returns
    -------
    lon_raster : `np.ndarray`
        Rasterized longitude values (length xsize).
    lat_raster : `np.ndarray`
        Rasterized latitude values (length xsize*aspect).
    values_raster : `np.ndarray`
        Rasterized values (2-d).
    """
    test = np.unique(pixels)
    if test.size != pixels.size:
        raise ValueError("The pixels array must be unique.")

    lon_raster, lat_raster = np.meshgrid(np.linspace(lon_range[0], lon_range[1], xsize),
                                         np.linspace(lat_range[0], lat_range[1], int(aspect*xsize)))

    # For pcolormesh we need the central locations
    clon = (lon_raster[1:, 1:] + lon_raster[:-1, :-1])/2.
    clat = (lat_raster[1:, 1:] + lat_raster[:-1, :-1])/2.

    pix_raster = hp.ang2pix(nside, clon, clat, nest=nest, lonlat=True)

    st = np.argsort(pixels)
    sub1 = np.searchsorted(pixels, pix_raster, sorter=st)

    if pix_raster.max() > pixels.max():
        bad = np.where(sub1 == pixels.size)
        sub1[bad] = pixels.size - 1

    sub2 = np.where(pixels[st[sub1]] == pix_raster)
    sub1 = st[sub1[sub2]]

    mask = np.ones(pix_raster.shape, dtype=bool)
    mask[sub2] = False
    values_raster = np.zeros(pix_raster.shape, dtype=values.dtype)
    values_raster[sub2] = values[sub1]

    return lon_raster, lat_raster, np.ma.array(values_raster, mask=mask)
