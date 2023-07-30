import os
import astropy
import argparse
import warnings
import numpy as np
from loguru import logger
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.table import Table
from astropy.nddata import Cutout2D
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from matplotlib.backends.backend_pdf import PdfPages
from footprints import vast_footprint, pickle_pointing_info, get_correct_file
from fitter import fitter

warnings.filterwarnings(
    action="ignore", category=astropy.wcs.FITSFixedWarning, module="astropy"
)


def validate_files(all_files):
    """Helper function that asserts if the files are present in the paths

    Args:
        all_files (dict): Dictionary of all the files, epoch wise

    Returns:
        dict: Modified dictionary that returns images, bkg, rms files
    """
    racs_epochs = ["EPOCH00", "EPOCH29"]
    all_epochs = list(all_files.keys())
    structured_files = {}
    for e in all_epochs:
        if len(all_files[e]["primary"]) > 0:
            try:
                image_files = np.concatenate(
                    (all_files[e]["primary"], all_files[e]["other"])
                )
            except KeyError:
                image_files = all_files[e]["primary"]

            bkg_files = []
            rms_files = []
            for f in image_files:
                if e in racs_epochs:
                    aux_file = f.replace("_IMAGES", "_BANE")
                    bkg_file = aux_file.replace(".fits", "_bkg.fits")
                    rms_file = aux_file.replace(".fits", "_rms.fits")
                else:
                    aux_file = f.replace("_IMAGES", "_RMSMAPS")
                    bkg_file = aux_file.replace("image.", "meanMap.image.")
                    rms_file = aux_file.replace("image.", "noiseMap.image.")
                if not os.path.isfile(bkg_file):
                    bkg_file = None
                if not os.path.isfile(rms_file):
                    rms_file = None
                bkg_files.append(bkg_file)
                rms_files.append(rms_file)
            structured_files[e] = {
                "primary_field": all_files[e]["primary_field"],
                "images": image_files,
                "bkg": bkg_files,
                "rms": rms_files,
            }
    return structured_files


class source:
    """Source class that is created for a given target"""

    def __init__(self, coords) -> None:
        """Takes the source coordinates as the input
        Args:
            coords (astropy.coordinates.sky_coordinate.SkyCoord): source coordinates
        """
        self.coords = coords
        # Create a table to store the results
        self.result = Table(
            names=["date", "flux", "rms", "err", "snr", "sep"],
            dtype=["U32", "f4", "f4", "f4", "f4", "f4"],
        )

    def _add_files(
        self,
        img_path,
        bkg_path=None,
        rms_path=None,
        stokes="I",
        size=2 * u.arcmin,
        epoch="xx",
    ):
        """Takes the source coordinates as the input
        Args:
            img_path (str): Path to the input image
            bkg_path (str, optional): Path to the mean map. Defaults to None.
            rms_path (str, optional): Path to the noise map. Defaults to None.
            stokes (str, optional): Stokes parameter to search for. Defaults to "I".
            size (astropy.units.quantity.Quantity, optional): Size of the cutout image
                Defaults to 2*u.arcmin.
            stokes (str, optional): _description_. Defaults to "I".
            size (_type_, optional): _description_. Defaults to 2*u.arcmin.
            epoch (str, optional): _description_. Defaults to "xx".

        Raises:
            Exception: If the source coordinates are not in the given files
        """

        self.stokes = stokes
        self.size = size
        self.epoch = epoch
        self.image = img_path
        self.bkg = bkg_path
        self.rms = rms_path

        self.image_hdu = fits.open(self.image)
        self.image_hdr = self.image_hdu[0].header
        self.image_wcs = WCS(self.image_hdr)
        # Check of the required coordinates are inside the given file
        if self.image_wcs.celestial.footprint_contains(self.coords):
            self.image_data = self.image_hdu[0].data

            logger.info(f"Reading the image file for {self.image}")

            if self.bkg is None:
                logger.warning(
                    f"No background file is provided for the image {self.image}"
                )
                self.bkg_data = None
            else:
                self.bkg_data = fits.getdata(self.bkg, 0)
            if self.rms is None:
                logger.warning(f"No RMS file is provided for the image {self.image}")
                self.rms_data = None
            else:
                self.rms_data = fits.getdata(self.rms, 0)
        else:
            logger.error("The given file does not contain the source coordinates")
            raise NotImplementedError(
                "The given file does not contain the source coordinates"
            )

    def get_cut_out_data(self):
        """Helper function to get the cutout data of the main TILE image"""

        ind = self.image_wcs.celestial.world_to_pixel(self.coords)

        # Check the diemsions and flatten it
        dim = self.image_data.ndim
        bkg_dim = self.bkg_data.ndim
        if dim == 2:
            slc = (slice(None), slice(None))
        else:
            slc = [0] * (dim - 2)
            slc += [slice(None), slice(None)]
            slc = tuple(slc)
        if dim != bkg_dim:
            bkg_slc = (slice(None), slice(None))
        else:
            bkg_slc = slc
        self.image_cut = Cutout2D(
            self.image_data[slc],
            [int(ind[0].item()), int(ind[1].item())],
            wcs=self.image_wcs.celestial,
            size=self.size,
        )
        if self.bkg_data is not None:
            self.bkg_cut = Cutout2D(
                self.bkg_data[bkg_slc],
                [int(ind[0].item()), int(ind[1].item())],
                wcs=self.image_wcs.celestial,
                size=self.size,
            )
        else:
            self.bkg_cut = None

        if self.rms_data is not None:
            self.rms_cut = Cutout2D(
                self.rms_data[bkg_slc],
                [int(ind[0].item()), int(ind[1].item())],
                wcs=self.image_wcs.celestial,
                size=self.size,
            )
        else:
            self.rms_cut = None

    def get_fluxes(self):
        """Function that does the fitting and flux estimation"""
        logger.info("Starting the source fitting")
        try:
            if (self.bkg_cut is None) and (self.rms_cut is None):
                fit = fitter(
                    self.image_cut.data,
                    None,
                    None,
                    self.image_cut.wcs,
                    self.image_hdr,
                    self.coords,
                )
            elif (self.bkg_cut is not None) and (self.rms_cut is None):
                fit = fitter(
                    self.image_cut.data,
                    self.bkg_cut.data,
                    None,
                    self.image_cut.wcs,
                    self.image_hdr,
                    self.coords,
                )
            elif (self.bkg_cut is None) and (self.rms_cut is not None):
                fit = fitter(
                    self.image_cut.data,
                    None,
                    self.rms_cut.data,
                    self.image_cut.wcs,
                    self.image_hdr,
                    self.coords,
                )
            else:
                fit = fitter(
                    self.image_cut.data,
                    self.bkg_cut.data,
                    self.rms_cut.data,
                    self.image_cut.wcs,
                    self.image_hdr,
                    self.coords,
                )
        except NotImplementedError:
            raise NotImplementedError("The given file has no good data")
        self.fit = fit
        self.fit.fit_gaussian()

        self.fit_offset = np.round(self.fit.fit_pos.separation(self.coords).arcsec, 2)
        self.result.add_row(
            [
                self.image_hdr["DATE"],
                self.fit.fit[0],
                self.fit.rms_err,
                self.fit.condon_err,
                np.abs(self.fit.fit[0] / self.fit.rms_err),
                self.fit_offset,
            ]
        )

    def save_cutout_image(self, plotfile=None):
        """
        Helper function to save the source finding image
        """
        plt.clf()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection=self.image_cut.wcs)
        ax.imshow(self.image_cut.data, aspect="auto", origin="lower", cmap="gray_r")
        X, Y = self.fit.grid
        Z = self.fit.psf_model
        levels = np.array(
            [0.049787068367863944, 0.1353352832366127, 0.36787944117144233]
        )
        if self.fit.fit[0] < 0:
            levels = self.fit.fit[0] * np.flip(levels)
        else:
            levels = levels * self.fit.fit[0]
        ax.contour(X, Y, Z, levels=levels, colors="r")
        src_pos = self.image_cut.wcs.celestial.world_to_pixel(self.coords)
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

        title = f"Stokes{self.stokes} image, epoch {self.epoch}, with {self.fit_offset} arcsec offset"
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("RA", fontsize=20)
        ax.set_ylabel("DEC", fontsize=20)
        if plotfile is None:
            plt.show()
        else:
            plotfile.savefig(bbox_inches="tight")
            plt.close("all")
            return plotfile

    def _clean(self):
        """Close all the opened files and remove the ouput"""
        self.image_hdu.close()
        del (
            self.image_cut,
            self.image_wcs,
            self.image_hdr,
            self.bkg_cut,
            self.rms_cut,
            self.bkg_data,
            self.rms_data,
            self.fit,
        )


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
        "--stokes",
        help="""Stokes parameter to search for, by default searches for "I".
        Should be provoided in str format, for eg. "I", or "V" or "all".""",
        type="str",
        default="I",
    )
    parser.add_argument(
        "--outdir",
        help="""Output directory to save the plots/light curves to.
        Default is the current directory""",
        type=str,
        default="./",
    )

    args = parser.parse_args()

    coord = SkyCoord(f"{args.coords[0]} {args.coords[1]}", unit=(u.hourangle, u.dgeree))

    # First get all the vast epochs and files
    vast = vast_footprint(stokes=args.stokes)
    vast._fetch_epochs()  # Get all the epochs and their paths
    pickle_pointing_info(vast)  # Update the pickle files

    # Get all the files to search for
    all_files = {}
    for e in vast.epoch_names:
        epoch_id = e[-2:]
        files = get_correct_file(vast, e, coord)
        all_files[e] = files

    all_files = validate_files(all_files=all_files)
    epochs = list(all_files.keys())

    # Now create a source class
    src = source(coords=coord)
    plotfile = PdfPages(f"{coord.to_string('hmsdms')}.pdf")

    for e in epochs:
        epoch_files = all_files[e]
        for i in range(len(epoch_files["images"])):
            src._add_files(
                epoch_files["images"][i],
                bkg_path=epoch_files["bkg"][i],
                rms_path=epoch_files["rms"][i],
                epoch=e[-2:],
            )
            src.get_cut_out_data()
            src.get_fluxes()
            src.save_cutout_image(plotfile=plotfile)
            src._clean
