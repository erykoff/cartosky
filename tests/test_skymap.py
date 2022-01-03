import os
import pytest

import matplotlib
matplotlib.use("Agg")

from matplotlib.testing.compare import compare_images, ImageComparisonFailure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import cartosky  # noqa: E402


ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize("skymap", [cartosky.Skymap,
                                    cartosky.McBrydeSkymap,
                                    cartosky.MollweideSkymap,
                                    cartosky.HammerSkymap,
                                    cartosky.EqualEarthSkymap])
def test_skymaps_basic(tmp_path, skymap):
    """Test full sky maps."""
    # Full image
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skymap(ax=ax)
    fname = f'{m.projection_name}_full.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.parametrize("skymap", [cartosky.Skymap,
                                    cartosky.McBrydeSkymap,
                                    cartosky.MollweideSkymap,
                                    cartosky.HammerSkymap,
                                    cartosky.EqualEarthSkymap,
                                    cartosky.LaeaSkymap])
def test_skymaps_zoom(tmp_path, skymap):
    # Simple zoom
    fig = plt.figure(1, figsize=(8, 5))
    fig.clf()
    ax = fig.add_subplot(111)
    m = skymap(ax=ax, extent=[0, 50, 0, 50])
    fname = f'{m.projection_name}_zoom.png'
    fig.savefig(tmp_path / fname)
    err = compare_images(os.path.join(ROOT, 'data', fname), tmp_path / fname, 10.0)
    if err:
        raise ImageComparisonFailure(err)
