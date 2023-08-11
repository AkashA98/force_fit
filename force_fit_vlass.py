import os
import numpy as np
from loguru import logger
import subprocess as subp
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS
from fitter import fitter
from matplotlib import pyplot as plt


class vlass:
    vlass_base_url = "https://archive-new.nrao.edu/vlass/quicklook/"
    base_dir = "../vlass_files/"

    def __init__(self, coord):
        """Initalize a vlass class taking the source coordinates

        Args:
            coord (astropy.coordinates.sky_coordinate.SkyCoord): Source coordinates
        """
        self.coord = coord
        self.all_epochs = ["1.1", "1.2", "2.1", "2.2", "3.1"]

        epoch_tile_info = {}
        for e in self.all_epochs:
            if e in ["1.1", "1.2"]:
                e += "v2"
            name = f"VLASS{e}"
            tile_info = Table.read(f"./vlass_pointing_files/{name}.csv")
            epoch_tile_info[e] = tile_info
        self.epoch_tile_info = epoch_tile_info
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

    def get_vlass_file(self, epoch="all"):
        """Function to download VLASS observation files, given a
        sky-coordinate and epoch

        Args:
            epoch (str, list): Obsevring epoch. Should be one of
            (1.1, 1.2, 2.1, 2.2, 3.1, all) or a list of a subset of these
        """

        coord = self.coord
        if epoch == "all":
            epoch = self.all_epochs
        elif type(epoch) is str:
            epoch = [epoch]
        self.epochs = [i + "v2" if i in ["1.1", "1.2"] else i for i in epoch]

        files_per_epoch = {}
        links_per_epoch = {}
        for e in self.epochs:
            name = f"VLASS{e}"
            epoch_path = f"{vlass.vlass_base_url}/{name}/"

            self.match_file(e=e)
            if self.match_field is None:
                logger.warning(
                    f"VLASS observations are not available for postion {coord.to_string('hmsdms')} for epoch {e}"
                )
                files_per_epoch[e] = None
            else:
                basename = (
                    epoch_path
                    + self.match_tile
                    + self.match_field
                    + self.match_field.replace("/", "")
                )
                img_path = basename + ".I.iter1.image.pbcor.tt0.subim.fits"
                rms_path = basename + ".I.iter1.image.pbcor.tt0.rms.subim.fits"
                files_per_epoch[e] = [img_path.split("/")[-1], rms_path.split("/")[-1]]
                links_per_epoch[e] = [img_path, rms_path]
        self.matched_files = files_per_epoch
        self.matched_links = links_per_epoch

    def match_file(self, e):
        """Helper function to get the pointing center from VLASS filename
        and get the file with the source closest to the center

        Args:
            e (str): observation epoch, one of (1.1, 1.2, 2.1, 2.2, 3.1)
        """
        coord = self.coord
        tiles = self.epoch_tile_info[e]["tile"]
        fields = self.epoch_tile_info[e]["field"]
        centers = [i.split(".")[4][1:] for i in fields]
        centers_coords = SkyCoord(
            ra=[i[:2] + " " + i[2:4] + " " + i[4:6] for i in centers],
            dec=[i[6:9] + " " + i[9:11] + " " + i[11:13] for i in centers],
            unit=(u.hourangle, u.degree),
        )

        sep = coord.separation(centers_coords)
        mask = sep.degree <= 1

        if np.any(mask):
            ind = np.nanargmin(sep.degree)
            path = fields[ind]
            logger.info(
                f"The file with pointing center closest to the target is {path}"
            )
            self.match_tile = tiles[ind]
            self.match_field = path
        else:
            self.match_tile = None
            self.match_field = None

    def download_files(self):
        """Helper function to download VLASS image files"""

        if not os.path.isdir(vlass.base_dir):
            os.mkdir(vlass.base_dir)
        files = self.matched_files
        links = self.matched_links
        for e in list(files.keys()):
            if files[e] is not None:
                for i, l in enumerate(files[e]):
                    # if not os.path.isfile(file_path):
                    logger.info(f"Downloading data from {links[e][i]}")
                    subp.call(
                        [f"wget -t 2 -q -nc -c -O {vlass.base_dir}{l} {links[e][i]}"],
                        shell=True,
                    )

    def get_cutout_data(self, e, size=2 * u.arcmin):
        """Get the cut-out data within given size

        Args:
            e (str): Observing epoch, one of (1.1, 1.2, 2.1, 2.2, 3.1)
            size (astropy.units.quantity.Quantity, optional): Cutout size. Defaults to 2*u.arcmin.
        """
        if self.matched_files[e] is not None:
            image = fits.open(f"{vlass.base_dir}/{self.matched_files[e][0]}")
            rms = fits.open(f"{vlass.base_dir}/{self.matched_files[e][1]}")

            hdr = image[0].header
            wcs = WCS(hdr)
            ind = wcs.celestial.world_to_pixel(self.coord)

            # Check the diemsions and flatten it
            dim = image[0].data.ndim

            if dim == 2:
                slc = (slice(None), slice(None))
            else:
                slc = [0] * (dim - 2)
                slc += [slice(None), slice(None)]
                slc = tuple(slc)

            self.image_cut = Cutout2D(
                image[0].data[slc],
                [int(ind[0].item()), int(ind[1].item())],
                wcs=wcs.celestial,
                size=size,
            )

            self.rms_cut = Cutout2D(
                rms[0].data[slc],
                [int(ind[0].item()), int(ind[1].item())],
                wcs=wcs.celestial,
                size=size,
            )
            self.header = hdr
            self.wcs = wcs
        else:
            self.image_cut = None

    def get_vlass_flux(self, e):
        """Get the forced fit for the given VLASS epoch

        Args:
            e (str): Observation epoch, one of (1.1, 1.2, 2.1, 2.2, 3.1)
        """
        self.get_cutout_data(e=e)
        if self.image_cut is not None:
            try:
                fit = fitter(
                    self.image_cut.data,
                    None,
                    self.rms_cut.data,
                    self.image_cut.wcs,
                    self.header,
                    self.coord,
                )
            except NotImplementedError:
                raise NotImplementedError("The given file has no good data")
            self.fit = fit
            self.fit.fit_gaussian(search=True)

            self.fit_offset = np.round(
                self.fit.fit_pos.separation(self.coord).arcsec, 2
            )
            self.result.add_row(
                [
                    e,
                    self.header["DATE-OBS"],
                    np.round(self.header["CRVAL3"] / 1e6, 1),
                    True,
                    self.fit.fit[0],
                    self.fit.rms_err,
                    self.fit.condon_err,
                    np.abs(self.fit.fit[0] / self.fit.rms_err),
                    self.fit_offset,
                ]
            )
            self.save_cutout_image()

    def save_cutout_image(self, plotfile=None):
        """Helper function to save the source finding image"""
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

    def clean(self):
        """Clean the variables from epoch to epoch"""
        try:
            del (
                self.image_cut,
                self.rms_cut,
                self.header,
                self.wcs,
                self.fit,
                self.fit_offset,
            )
        except AttributeError:
            pass


if __name__ == "__main__":
    s = SkyCoord("20:39:09.12 -30:45:20.84", unit=(u.hourangle, u.degree))

    v = vlass(s)
    v.get_vlass_file()
    v.download_files()
    for e in v.epochs:
        v.get_vlass_flux(e=e)
        v.clean()
