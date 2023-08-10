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

from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
rc("text", usetex=True)

warnings.filterwarnings(
    action="ignore", category=astropy.wcs.FITSFixedWarning, module="astropy"
)


def parse_coordinates(ra, dec):
    """Helper function to parse the input coordinates

    Args:
        ra (str/list): Right Ascension
        dec (str/list): Declination

    Returns:
        astropy.coordinates.sky_coordinate.SkyCoord: coordinate object
    """
    if type(ra) is str:
        ra = np.array([ra])
        dec = np.array([dec])
    try:
        coord = SkyCoord(
            ra=ra.astype(float) * u.degree, dec=dec.astype(float) * u.degree
        )
    except ValueError:
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.degree))

    return coord


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
            names=[
                "epoch",
                "date",
                "freq",
                "primary",
                "flux",
                "rms",
                "err",
                "snr",
                "sep",
            ],
            dtype=["U8", "U32", "f4", "bool", "f4", "f4", "f4", "f4", "f4"],
        )

    def _add_files(
        self,
        img_path,
        bkg_path=None,
        rms_path=None,
        stokes="I",
        size=2 * u.arcmin,
        epoch="xx",
        primary=True,
        search=True,
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
        self.is_primary = primary
        self.search = search

        self.image_hdu = fits.open(self.image)
        self.image_hdr = self.image_hdu[0].header
        self.image_wcs = WCS(self.image_hdr)

        # get the observation frequency
        # try:
        #     key = [i for i in self.image_hdr if self.image_hdr == "FREQ"][0]
        #     key = key.replace("TYPE", "RVAL")
        #     self.freq = self.image_hdr[key] / 1e6
        # except IndexError:
        #     self.freq = np.nan
        #     logger.warning("Observation frequency not found")
        self.freq = self.image_hdr["CRVAL3"] / 1e6
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

        if dim == 2:
            slc = (slice(None), slice(None))
        else:
            slc = [0] * (dim - 2)
            slc += [slice(None), slice(None)]
            slc = tuple(slc)

        if (self.bkg_data is not None) or (self.rms_data is not None):
            bkg_dim = self.bkg_data.ndim
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
        self.fit.fit_gaussian(search=self.search)

        self.fit_offset = np.round(self.fit.fit_pos.separation(self.coords).arcsec, 2)
        self.result.add_row(
            [
                self.epoch,
                self.image_hdr["DATE"],
                np.round(self.freq, 1),
                self.is_primary,
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


def get_vast_flux(coords, names, args):
    """Helper function that runs the source findign specific to RACS/VAST

    Args:
        coords (astropy.coordinates.sky_coordinate.SkyCoord): Source coordinates
        args (argparse.ArgumentParser): parser to handle cli arguments
    """

    # First get all the vast epochs and files
    vast = vast_footprint(stokes=args.stokes)
    vast._fetch_epochs()  # Get all the epochs and their paths
    pickle_pointing_info(vast)  # Update the pickle files

    # Now work on each source

    for ind, s in enumerate(coords):
        # Get all the files to search for
        all_files = {}
        for e in vast.epoch_names:
            epoch_id = e[-2:]
            files = get_correct_file(vast, epoch_id, s)
            all_files[e] = files

        all_files = validate_files(all_files=all_files)
        epochs = list(all_files.keys())

        # Now create a source class
        src = source(coords=s)
        if args.plot:
            plotfile = PdfPages(f"{args.outdir}/{names[ind]}_stokes{args.stokes}.pdf")
        else:
            plotfile = None

        for e in epochs:
            epoch_files = all_files[e]
            for i in range(len(epoch_files["images"])):
                field_name = [
                    i
                    for i in epoch_files["images"][i].split("/")[-1].split(".")
                    if (("RACS" in i) or ("VAST" in i))
                ][0]
                is_primary_field = (
                    True if (field_name == epoch_files["primary_field"]) else False
                )
                try:
                    src._add_files(
                        epoch_files["images"][i],
                        bkg_path=epoch_files["bkg"][i],
                        rms_path=epoch_files["rms"][i],
                        epoch=e[-2:],
                        size=3 * u.arcmin,
                        primary=is_primary_field,
                        search=args.search,
                    )
                    src.get_cut_out_data()
                    src.get_fluxes()
                    src.save_cutout_image(plotfile=plotfile)
                    src._clean()
                except NotImplementedError:
                    continue

        if args.plot:
            plotfile.close()
        if args.flux:
            src.result.write(
                f"{args.outdir}/{names[ind]}_stokes{args.stokes}_measurements.txt",
                format="ascii",
                overwrite=True,
            )
        else:
            print(src.result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Script to to targetted searches for VAST data.
        Does force fitting for a given position and if needed searches around the 
        given coordinates to look for sources"""
    )
    parser.add_argument(
        "coords",
        help="""Source coordinates to search for, can be a csv file having the columns
         RA and DEC or a pair of strings with RA and DEC of the source (ascii format).
          For eg. (both of the following work, but beware of the '-' in declination)
         python force_fit.py 12:54:24.3 +34:12:05.3 (can be hmsdms or degrees) or 
         python force_fit.py my_list_of_coords.csv""",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--e",
        help="""Epochs to search for, by default searches in all the epochs.
        Should be provoided in str format, for eg. '00', or '23'. Default is 
        search for all epochs using 'all'.""",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--stokes",
        help="""Stokes parameter to search for, by default searches for "I".
        Should be provoided in str format, for eg. "I", or "V" or "all".""",
        type=str,
        default="I",
    )
    parser.add_argument(
        "--search",
        help="""Flag that decides whether to search around in the coordinate space
        or to just fit for flux with the position fixed""",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--outdir",
        help="""Output directory to save the plots/light curves to.
        Default is the current directory""",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--plot",
        help="""Flag to control whether to save the plots""",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--flux",
        help="""Flag to control whether to save the measurements""",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    logger.remove()  # Remove all handlers added so far, including the default one.
    logger.add("testlog.log", level="INFO")

    coord = args.coords
    try:
        mask = os.path.isfile(args.coords[0])
        # This means a file with RA and DEC are given
        tab = Table.read(args.coords[0])
        coord = parse_coordinates(tab["RA"].data, tab["DEC"].data)
        try:
            names = tab["Name"]
            names = [i.replace(" ", "") for i in names]
        except NameError:
            names = coord.to_string("hmsdms")
    except (FileNotFoundError, TypeError):
        if len(coord) == 2:
            # This means a pair of coords is given
            coord = parse_coordinates(coord[0], coord[1])
        else:
            ra, dec = coord[0].split(" ")
            coord = parse_coordinates(ra, dec)
        names = coord.to_string("hmsdms")

    get_vast_flux(coords=coord, names=names, args=args)
