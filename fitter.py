import numpy as np
from loguru import logger
from astropy import units as u
from lmfit import Model, create_params
from astropy.nddata import NoOverlapError

fwhm_to_sig = np.sqrt(8 * np.log(2))
nan_mask_value = 1e8


def gaussian_kernel(X, A, x0, y0, sx, sy, pa):
    """
    Function that defines the 2D Gaussian in terms of the vector
    position, semi-major/minor axes and the position angle and the
    peak flux (amplitude)
    """
    x, y = X
    theta = np.radians(pa)

    a = np.cos(theta) ** 2 / (2 * sx**2) + np.sin(theta) ** 2 / (2 * sy**2)
    b = np.sin(theta) ** 2 / (2 * sx**2) + np.cos(theta) ** 2 / (2 * sy**2)
    c = np.sin(2 * theta) / (2 * sx**2) - np.sin(2 * theta) / (2 * sy**2)

    power = -a * (x - x0) ** 2 - b * (y - y0) ** 2 - c * (x - x0) * (y - y0)

    return np.ravel(A * np.exp(power))


class fitter:
    """A fitter class that does the 2D gaussian fitting"""

    def __init__(self, data, bkg, rms, wcs, meta, coords) -> None:
        """Intialize a fitter class. Takes the image, bkg, rms data
        as inputs along with the header that has the metadata

        Args:
            data (ndarray): image data
            bkg (ndarray): mean map (background data)
            rms (ndarray): noise map (rms data)
            wcs (astropy.wcs.wcs.WCS): WCS object of the image
            meta (astropy.io.fits.header.Header): header of the image
            coords (astropy.coordinates.sky_coordinate.SkyCoord): source coordinates
        """

        self.image = data
        self.bkg = bkg
        self.bkg_flag = False if bkg is None else True
        self.rms = rms
        self.rms_flag = False if rms is None else True
        self.hdr = meta
        self.wcs = wcs
        self.coords = coords
        center = self.wcs.world_to_pixel(self.coords)
        self.center = np.round(np.array(center), 0).astype(int)

        hdr = self.hdr
        if hdr["BUNIT"] == "Jy/beam":
            logger.info("Given fluxes are in Jy, converting them to mJy")
            self.fac = 1000
        else:
            self.fac = 1

        if np.all(np.isnan(data)):
            logger.error("All input data are NaN's, so doing nothing.")
            raise ValueError("The given file has no good data")

        if np.any(self.center < 0):
            logger.error("The given source cooridnates are outside the image")
            raise NoOverlapError("Source is not in the image")

        # Correct the data for nan's
        # Handle nan values. This is a bad way of handling nan values
        # but currently the only way
        self.image[np.isnan(self.image)] = 0
        self.image *= self.fac

        # Do the same for noise data
        if not self.bkg_flag:
            logger.warning("No background data provided, so assuming mean map is 0")
            self.bkg = np.zeros_like(self.image)
        else:
            # Deal with nan values again
            self.bkg[np.isnan(self.bkg)] = 0
            self.bkg *= self.fac

        if not self.rms_flag:
            logger.warning("No RMS data provided, assuming equal weights -- 1 mJy")
            self.rms = np.ones_like(data) * 0.001
        else:
            # Set nan values in rms map so high, that weight is 0
            self.rms[np.isnan(self.rms)] = nan_mask_value
            self.rms[self.rms == 0] = nan_mask_value
            self.rms *= self.fac

    def _get_psf_info(self):
        """
        Function to get the information about the point-spread function
        """
        hdr = self.hdr
        try:
            self.pix_scl1 = abs(hdr["CDELT1"])
            self.pix_scl2 = abs(hdr["CDELT2"])
        except KeyError:
            self.pix_scl1 = abs(hdr["CD1_1"])
            self.pix_scl2 = abs(hdr["CD2_2"])

        self.bmaj = abs(hdr["BMAJ"])
        self.bmin = abs(hdr["BMIN"])

        self.bmaj_pix = abs(hdr["BMAJ"]) / self.pix_scl1
        self.bmin_pix = abs(hdr["BMIN"]) / self.pix_scl2

        self.pos_ang = hdr["BPA"] % 180

    def get_rms_from_image(self):
        # Calculate errors from the image itself
        # Since there is no error map, calculate by hand
        data = self.image
        l = len(data)
        rms1 = np.nanstd(data[0 : l // 2 - 10, 0 : l // 2 - 10])
        rms2 = np.nanstd(data[0 : l // 2 - 10, l // 2 + 10 : l])
        rms3 = np.nanstd(data[l // 2 + 10 : l, 0 : l // 2 - 10])
        rms4 = np.nanstd(data[l // 2 + 10 : l, l // 2 + 10 : l])
        rms = 0.25 * (rms1 + rms2 + rms3 + rms4)
        return rms

    def calculate_errors(self, fit):
        """
        Function to calculate fit errors according eqn 41 of Condon 1997
        """

        _, _, _, sx, sy, _ = fit
        sx, sy = sx * fwhm_to_sig, sy * fwhm_to_sig
        snr_2_c1 = sx * sy / (4 * self.bmaj_pix * self.bmin_pix)
        snr_2_c2 = 1 + (self.bmaj_pix * self.bmin_pix / (sx**2))
        snr_2_c3 = 1 + (self.bmaj_pix * self.bmin_pix / (sy**2))

        # estimate RMS of the image
        if self.rms_flag:
            mu = self.rms[self.center[0], self.center[0]]
        else:
            mu = self.get_rms_from_image()

        snr_2 = snr_2_c1 * (snr_2_c2 ** (1.5)) * (snr_2_c3 ** (1.5)) / (mu**2)
        amp_err = np.sqrt(2 / snr_2)

        self.rms_err = mu
        self.condon_err = amp_err

    def convert_fit_values_to_astrometric(self):
        """
        Convert from pixel space to astrometric space
        """
        _, x0, y0, sx, sy, pa = self.fit
        pos = self.wcs.celestial.pixel_to_world([x0], [y0])[0]
        maj = np.max([sx, sy])
        minor = np.min([sx, sy])

        fwhm_maj = maj * self.pix_scl1 * fwhm_to_sig
        fwhm_min = minor * self.pix_scl2 * fwhm_to_sig

        self.fit_pos = pos
        self.fit_shape = np.array([fwhm_maj, fwhm_min, pa])

    def make_params(self, data, center):
        """Helper function to make the fot parameters"""
        data_max = np.max(np.abs(data))
        peak_value = np.mean(
            data[center[0] - 1 : center[0] + 2, center[1] - 1 : center[1] + 1]
        )

        sx = self.bmaj_pix / fwhm_to_sig
        sy = self.bmin_pix / fwhm_to_sig
        pa = self.pos_ang
        p0 = (peak_value, center[0], center[1], sx, sy, pa)
        pmax = (
            2 * data_max,
            center[0] + 2 * sx,
            center[1] + 2 * sy,
            1.5 * sx,
            1.5 * sy,
            180,
        )
        pmin = (
            -2 * data_max,
            center[0] - 2 * sx,
            center[1] - 2 * sy,
            0.5 * sx,
            0.5 * sy,
            0,
        )

        self.p0 = p0
        self.p0_bounds = (np.array(pmin), np.array(pmax))

        params = create_params(
            A=dict(value=p0[0], min=pmin[0], max=pmax[0]),
            x0=dict(value=p0[1], min=pmin[1], max=pmax[1]),
            y0=dict(value=p0[2], min=pmin[2], max=pmax[2]),
            sx=dict(value=p0[3], min=pmin[3], max=pmax[3]),
            sy=dict(value=p0[4], min=pmin[4], max=pmax[4]),
            pa=dict(value=p0[5], min=pmin[5], max=pmax[5]),
        )
        return params

    def revise_params(self, fit, errs=None):
        """Helper function to revise the fit."""
        p0 = fit
        if errs is None:
            sx = self.bmaj_pix / fwhm_to_sig
            sy = self.bmin_pix / fwhm_to_sig
            pa = self.pos_ang
            errs = np.array([0.25, sx, sy, np.sqrt(sx), np.sqrt(sy), pa])

        params = create_params(
            A=dict(value=p0[0], min=p0[0] - 2 * errs[0], max=p0[0] + 2 * errs[0]),
            x0=dict(value=p0[1], min=p0[1] - 2 * errs[1], max=p0[1] + 2 * errs[1]),
            y0=dict(value=p0[2], min=p0[2] - 2 * errs[2], max=p0[2] + 2 * errs[2]),
            sx=dict(value=p0[3], min=p0[3] - 2 * errs[3], max=p0[3] + 2 * errs[3]),
            sy=dict(value=p0[4], min=p0[4] - 2 * errs[4], max=p0[4] + 2 * errs[4]),
            pa=dict(value=p0[5], min=p0[5] - 2 * errs[5], max=p0[5] + 2 * errs[5]),
        )
        return params

    def get_forced_flux(self):
        """
        Function to get the forced flux instead of fitting the position.
        Similar to the Kaplan+Andrew force fitting routine.
        """
        data = np.copy(self.image)

        # Make a pixel grid for the image
        x, y = np.arange(len(data)), np.arange(len(data[0]))
        X, Y = np.meshgrid(x, y)
        sx, sy, pa = (
            self.bmaj_pix / fwhm_to_sig,
            self.bmin_pix / fwhm_to_sig,
            self.pos_ang,
        )
        norm = 1  # snp.cos(2 * pa) / 2 / np.pi / sx / sy
        kernel = gaussian_kernel(
            (X, Y), norm, self.center[0], self.center[1], sx, sy, pa
        )
        kernel = kernel.reshape(X.shape)
        # Do a noise weighted sum
        flux = np.nansum((data - self.bkg) * kernel / self.rms**2) / np.nansum(
            kernel**2 / self.rms**2
        )
        flux_err = np.nansum((self.rms) * kernel / self.rms**2) / np.nansum(
            kernel / self.rms**2
        )

        return flux, flux_err

    def fit_gaussian(self, search=True):
        """
        Do the 2D gaussian fitting using the lmfit package
        """

        data = np.copy(self.image)
        bkg_data = np.copy(self.bkg)
        rms_data = np.copy(self.rms)
        sub_data = data - bkg_data

        self._get_psf_info()

        # Make a pixel grid for the image
        x, y = np.arange(len(data)), np.arange(len(data[0]))
        X, Y = np.meshgrid(x, y)
        self.grid = (X, Y)

        center = self.center
        # Make an initial guess and the bounds for the parameters
        # Source will always be at the center of the data

        # decide whether to search around or not
        self.fit_success = False
        if search:
            logger.info("Fitting for position as well as the flux")
            # Fit it using lmfit
            fmodel = Model(gaussian_kernel, independent_vars=("X"))

            params = self.make_params(data=sub_data, center=center)

            res = fmodel.fit(
                np.ravel(sub_data),
                params,
                X=(X, Y),
                weights=1 / np.ravel(rms_data),
            )

            # try revising the fit (2 iterations)
            niter = 0
            while niter < 2:
                if res.covar is None:
                    errs = None
                else:
                    errs = np.sqrt(np.diag(res.covar))
                new_params = self.revise_params(list(res.best_values.values()), errs)
                logger.info(f"Updating fit -- trail {niter+1}/2")
                res = fmodel.fit(
                    np.ravel(sub_data),
                    new_params,
                    X=(X, Y),
                    weights=1 / np.ravel(rms_data),
                )
                niter += 1

            self.res = res
            conv_success = (1 <= res.summary()["ier"] <= 4) and (
                res.summary()["success"]
            )

            covar_success = res.covar is not None
            # Calculate errors
            fit_values = np.array(list(res.best_values.values()))
            self.calculate_errors(fit_values)
            det_success = np.abs(fit_values[0] / self.rms_err) >= 3
            fit_success = conv_success & covar_success & det_success

            if fit_success:
                self.fit_success = True
                logger.info("Fit is success")
                # This means that the fit has converged and is successful
                fit_errs = np.sqrt(np.diag(res.covar))

                self.fit = fit_values
                self.fit_err = fit_errs

                self.convert_fit_values_to_astrometric()
                logger.info(
                    f"""A component is fit {np.round(self.fit_pos.separation(self.coords).arcsec, 2)}
                    arcsec away from the requested coordinates, with a peak flux of {fit_values[0]} +/- 
                    {self.rms_err} mJy, translating to an SNR of {np.abs(fit_values[0]/self.rms_err)}.
                    """
                )

        if (not search) or (not self.fit_success):
            if not self.fit_success:
                # If not the fit has failed to converge, so then
                # do a beam average of the flux and err
                logger.warning("Fit failed to converge")
            if not search:
                logger.info("Fitting for just the flux at the given position")
            # do a beam average of the flux and err
            if self.rms_flag:
                logger.info("Doing a simple forced fit for the given position")
                flux, flux_err = self.get_forced_flux()
                self.fit = np.array(
                    [
                        flux,
                        self.center[0],
                        self.center[1],
                        self.bmaj_pix / fwhm_to_sig,
                        self.bmin_pix / fwhm_to_sig,
                        self.pos_ang,
                    ]
                )
                self.fit_err = np.array(
                    [flux_err, np.nan, np.nan, np.nan, np.nan, np.nan]
                )
                self.rms_err = flux_err
            else:
                logger.info(
                    "No RMS data is given, so simply giving the flux at the position\
                    and the RMS calculated from the image itself"
                )
                self.fit = [
                    data[center[0], center[1]],
                    self.center[0],
                    self.center[1],
                    self.bmaj_pix / fwhm_to_sig,
                    self.bmin_pix / fwhm_to_sig,
                    self.pos_ang,
                ]
                rms = self.get_rms_from_image()
                self.fit_err = [rms, np.nan, np.nan, np.nan, np.nan, np.nan]
                self.rms_err = rms
            self.condon_err = None
            self.fit_pos = self.coords
            self.fit_shape = np.array([self.bmaj, self.bmin, self.pos_ang])

        self.psf_model = fmodel.func((X, Y), *self.fit).reshape(X.shape)
