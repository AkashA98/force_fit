import os
import astropy
import argparse
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.table import Table
from astropy.nddata import Cutout2D
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from matplotlib.backends.backend_pdf import PdfPages
from footprints import get_correct_file, fetch_epochs
from fitter import fit_gaussian, convert_fit_values_to_astrometric

warnings.filterwarnings(
    action="ignore", category=astropy.wcs.FITSFixedWarning, module="astropy"
)


def get_cut_out_data(hdu, s, size=2 * u.arcmin):
    """
    Helper function to get the cutout data of the main TILE image
    """
    w = WCS(hdu[0].header)
    data = hdu[0].data
    ind = w.celestial.world_to_pixel(s)

    # Check the diemsions and flatten it
    dim = data.ndim
    if dim == 4:
        data = data[0][0]
    try:
        cut_data = Cutout2D(
            data, [int(ind[0].item()), int(ind[1].item())], wcs=w.celestial, size=size
        )
    except astropy.nddata.utils.NoOverlapError:
        cut_data = None

    return cut_data


def save_cutout_image(cut_data, coord, plotfile, model=None, title=None):
    """
    Helper function to save the source finding image
    """
    plt.clf()
    _, ax = plt.subplot(figsize=(8, 6), projection=cut_data.wcs)
    ax.imshow(cut_data.data, aspect="auto", origin="lower", cmap="gray_r")
    if model is not None:
        X, Y, Z, levels = model
        ax.contour(X, Y, Z, levels=levels, colors="r")
    src_pos = cut_data.wcs.celestial.world_to_pixel(coord)
    ax.plot(
        [src_pos[0], src_pos[0]],
        [src_pos[1] + 1, src_pos[1] + 2],
        color="r",
    )
    ax.plot(
        [src_pos[0], src_pos[0]],
        [src_pos[1] - 2, src_pos[1] - 1],
        color="r",
    )
    ax.plot(
        [src_pos[0] - 2, src_pos[0] - 1],
        [src_pos[1], src_pos[1]],
        color="r",
    )
    ax.plot(
        [src_pos[0] + 1, src_pos[0] + 2],
        [src_pos[1], src_pos[1]],
        color="r",
    )
    ax.tick_params(labelsize=15)
    ax.grid(True, color="k", ls="--")
    if title is None:
        title = coord.to_string("hmsdms")
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("RA", fontsize=20)
    ax.set_ylabel("DEC", fontsize=20)
    plotfile.savefig(bbox_inches="tight")
    return plotfile


def get_racs_flux(coord, epoch="all", path="./", size=2 * u.arcmin, save_images=True):
    """
    Function to do the main source finding and fitting
    """

    if epoch == "all":
        epochs = fetch_epochs()
        epoch_names = list(epochs.keys())

    # Fetch all the files in all the epochs that have the source in
    # their primary beam
    all_files = []
    racs = []
    for e in epoch_names:
        epoch_id = e[-2:]
        match_files = get_correct_file(epoch_id, coord)
        if match_files is not None:
            all_files.append(match_files["primary"])
            if epoch_id in ["00", "14", "29"]:
                racs.append(True)
            else:
                racs.append(False)

    if len(all_files) > 0:
        res = []
        for i, p in enumerate(all_files):
            if racs[i]:
                p_rms = p.replace("STOKESV_IMAGES", "STOKESV_RMSMAPS")
                p_bkg = p_rms.replace(".fits", ".bkg.fits")
                p_rms = p_rms.replace(".fits", ".rms.fits")
            else:
                p_rms = p.replace("STOKESI_IMAGES", "STOKESI_RMSMAPS")
                p_bkg = p_rms.replace("VAST_", "meanMap.VAST_")
                p_rms = p_rms.replace("VAST_", "noiseMap.VAST_")

            hdu_im = fits.open(p)
            cut_data_im = get_cut_out_data(hdu_im, coord, size=size)

            rms_mask = 0

            if os.path.isfile(p_bkg):
                hdu_bkg = fits.open(p_bkg)
                cut_data_bkg = get_cut_out_data(hdu_bkg, coord, size=size)
            else:
                rms_mask += 1

            if os.path.isfile(p_rms):
                hdu_rms = fits.open(p_rms)
                cut_data_rms = get_cut_out_data(hdu_rms, coord, size=size)
            else:
                rms_mask += 1

            if np.all(np.isnan(cut_data_im.data)):
                return None
            else:
                # Do the actual fitting
                if rms_mask != 0:
                    X, model, fit_values, fit_errs, condon_err = fit_gaussian(
                        cut_data_im.data
                    )
                else:
                    X, model, fit_values, fit_errs, condon_err = fit_gaussian(
                        cut_data_im.data, cut_data_rms.data, cut_data_bkg.data
                    )
                rms = fit_errs[0]
                if condon_err is None:
                    condon_err = np.nan
                    snr = fit_values[0] / rms
                else:
                    snr = fit_values[0] / condon_err
                pos, fwhm_maj, fwhm_min = convert_fit_values_to_astrometric(
                    fit_values, cut_data_im.wcs, hdu_im[0].header
                )

                if save_images:
                    # Do the plotting
                    plotfile = PdfPages(f"{path}/{coord.to_string('hmsdms')}.pdf")
                    title = f"Stokes I image : epoch {e}, {np.round(pos.separation(coord).arcsec, 2)} arcsec offset"
                    levels = (
                        np.array(
                            [
                                0.049787068367863944,
                                0.1353352832366127,
                                0.36787944117144233,
                            ]
                        )
                        * fit_values[0]
                    )
                    if fit_values[0] < 0:
                        levels = np.flip(levels)
                    model = [X[0], X[1], model, levels]
                    plotfile = save_cutout_image(
                        cut_data_im, coord, plotfile, model=model, title=title
                    )
                res.append(
                    [
                        hdu_im[0].header["DATE-OBS"],
                        fit_values[0] * 1000,
                        rms * 1000,
                        condon_err * 1000,
                        snr,
                        pos.separation(coord).acrsec,
                    ]
                )
        if len(res) > 0:
            return res
        else:
            return None
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Script to to targetted searches for VAST data.
        Does force fitting for a given position and if needed searches around the 
        given coordinates to look for sources"""
    )
    parser.add_argument(
        "coords",
        help="Source coordinates to search for",
        nargs=2,
        action="append",
        type=list,
    )
    parser.add_argument(
        "--e",
        help="""Epochs to search for, by default searches in all the epochs.
        Should be provoided in str format, for eg. '00', or '23'. Default is 
        search for all epochs using 'all'.""",
        type="str",
        default="all",
    )
    parser.add_argument(
        "--outdir",
        help="""Output directory to save the plots/light curves to.
        Default is the current directory""",
        type=str,
        default="./",
    )

    args = parser.parse_args()

    tab = Table(
        names=["Name", "date", "flux", "rms", "err", "snr", "sep"],
        dtype=["U32", "U32", "f4", "f4", "f4", "f4", "f4", "f4"],
    )

    res = get_racs_flux(args.coord, args.epoch, args.path)
