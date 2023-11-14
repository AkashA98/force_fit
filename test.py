import numpy as np
from loguru import logger
from astropy import units as u
from lmfit import Model, create_params
from matplotlib import pyplot as plt
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
rc("text", usetex=True)


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


# def gaussian_kernel_multi(X, A, x0, y0, sx, sy, pa):
#     """Function that defines the 2D Gaussian in terms of the vector
#         position, semi-major/minor axes and the position angle and the
#         peak flux (amplitude)

#     Args:
#         X (np.array.ndarray): image grid
#         A (float/list): Peak amplitude
#         x0 (float/list): Image center (x)
#         y0 (float/list): Image center (y)
#         sx (float/list): PSF x
#         sy (float/list): PSF y
#         pa (float/list): polarization angle

#     Returns:
#         np.array.ndarray: Image
#     """
#     x, y = X
#     n = len(A)
#     x = np.repeat(x, n, axis=-1).reshape((n, 48, 48))
#     y = np.repeat(y, n, axis=-1).reshape((n, 48, 48))
#     theta = np.radians(pa)

#     x0 = x0[:, np.newaxis, np.newaxis]
#     y0 = y0[:, np.newaxis, np.newaxis]

#     a = np.cos(theta) ** 2 / (2 * sx**2) + np.sin(theta) ** 2 / (2 * sy**2)
#     b = np.sin(theta) ** 2 / (2 * sx**2) + np.cos(theta) ** 2 / (2 * sy**2)
#     c = np.sin(2 * theta) / (2 * sx**2) - np.sin(2 * theta) / (2 * sy**2)

#     a = a[:, np.newaxis, np.newaxis]
#     b = b[:, np.newaxis, np.newaxis]
#     c = c[:, np.newaxis, np.newaxis]
#     A = A[:, np.newaxis, np.newaxis]
#     power = -a * (x - x0) ** 2 - b * (y - y0) ** 2 - c * (x - x0) * (y - y0)

#     return A * np.exp(power)


def simulate_data(n_images=1, mean=0, std=0.25, shape=(48, 48)):
    """VAST data has a pixel scale of 2.5'', so simulate a 2' data set
        which is a 48x48 grid

    Args:
        n_images (int, optional): Simulate these number of images. Defaults to 1.
        mean (int, optional): Noise mean. Defaults to 0 (refers to background sub).
        std (float, optional): Typical RMS noise. Defaults to 0.25.
        shape (tuple, optional): image shape. Defaults to (48, 48).
    """

    # First generate background
    bkg = np.random.normal(loc=mean, scale=std, size=(n_images, shape[0], shape[1]))

    # Next add sources
    # Spread peak uniformly about 1 and 1000 sigma
    peak = np.random.uniform(low=1, high=100, size=(n_images,)) * std

    # Spread location around the center uniformly
    loc_x = np.random.randint(
        low=shape[0] // 2 - 3, high=shape[0] // 2 + 3, size=(n_images,)
    )
    loc_y = np.random.randint(
        low=shape[1] // 2 - 3, high=shape[1] // 2 + 3, size=(n_images,)
    )

    # Change shape parameters
    psf_x = np.random.normal(loc=17 / 2.5, scale=4 / 2.5, size=(n_images,))
    psf_y = np.random.normal(loc=15 / 2.5, scale=4 / 2.5, size=(n_images,))
    pol = np.random.uniform(low=0, high=360, size=(n_images,))

    grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    source = []
    for i in range(n_images):
        s = gaussian_kernel(
            X=grid,
            A=peak[i],
            x0=loc_x[i],
            y0=loc_y[i],
            sx=psf_x[i],
            sy=psf_y[i],
            pa=pol[i],
        )
        source.append(s.reshape((48, 48)))

    source = np.array(source)

    signal = bkg + source
    return peak, loc_x, loc_y, psf_x, psf_y, pol, signal


class fitter:
    """A fitter class that does the 2D gaussian fitting"""

    def __init__(
        self, data, bkg=None, rms=None, search=True, sx=3, sy=3, pol=90
    ) -> None:
        """Intialize a fitter class. Takes the image, bkg, rms data
        as inputs along with the header that has the metadata
        """

        self.image = data
        self.bkg = bkg
        self.bkg_flag = False if bkg is None else True
        self.rms = rms
        self.rms_flag = False if rms is None else True

        self.bmaj_pix = sx
        self.bmin_pix = sy

        self.pos_ang = pol % 180

        self.search = search
        center = np.array([data.shape[0] / 2, data.shape[1] / 2])
        self.center = np.round(np.array(center), 0).astype(int)

        # Correct the data for nan's
        # Handle nan values. This is a bad way of handling nan values
        # but currently the only way
        self.image[np.isnan(self.image)] = 0

        # Do the same for noise data
        if not self.bkg_flag:
            self.bkg = np.zeros_like(self.image)
        else:
            # Deal with nan values again
            self.bkg[np.isnan(self.bkg)] = 0

        if not self.rms_flag:
            self.rms = np.ones_like(data)
        else:
            # Set nan values in rms map so high, that weight is 0
            self.rms[np.isnan(self.rms)] = nan_mask_value
            self.rms[self.rms == 0] = nan_mask_value

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

    def make_params(self, data, center):
        """Helper function to make the fot parameters"""
        data_max = np.max(np.abs(data))
        peak_value = np.max(
            data[center[0] - 1 : center[0] + 2, center[1] - 1 : center[1] + 2]
        )

        sx = self.bmaj_pix
        sy = self.bmin_pix
        pa = self.pos_ang
        p0 = (peak_value, center[0], center[1], sx, sy, pa)
        pmax = (
            2 * data_max,
            center[0] + 2 * sx,
            center[1] + 2 * sy,
            2.5 * sx,
            2.5 * sy,
            180,
        )
        pmin = (
            -2 * data_max,
            center[0] - 2 * sx,
            center[1] - 2 * sy,
            0,
            0,
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
        # if errs is None:
        sx = self.bmaj_pix
        sy = self.bmin_pix
        pa = self.pos_ang
        errs = np.array([0.5, sx, sy, np.sqrt(sx), np.sqrt(sy), pa])

        params = create_params(
            A=dict(value=p0[0], min=p0[0] / 2, max=p0[0] * 2),
            x0=dict(value=p0[1], min=p0[1] - 2 * errs[1], max=p0[1] + 2 * errs[1]),
            y0=dict(value=p0[2], min=p0[2] - 2 * errs[2], max=p0[2] + 2 * errs[2]),
            sx=dict(value=sx, min=0.5 * sx, max=1.5 * sx),
            sy=dict(value=sy, min=0.5 * sy, max=1.5 * sy),
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
        sx = self.bmaj_pix
        sy = self.bmin_pix
        pa = self.pos_ang
        norm = 1
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

    def fit_gaussian(self):
        """
        Do the 2D gaussian fitting using the lmfit package
        """

        data = np.copy(self.image)
        bkg_data = np.copy(self.bkg)
        rms_data = np.copy(self.rms)
        sub_data = data - bkg_data
        search = self.search

        # Make a pixel grid for the image
        x, y = np.arange(len(data)), np.arange(len(data[0]))
        X, Y = np.meshgrid(x, y)
        self.grid = (X, Y)

        center = self.center
        # Make an initial guess and the bounds for the parameters
        # Source will always be at the center of the data
        fmodel = Model(gaussian_kernel, independent_vars=("X"))

        # decide whether to search around or not
        self.fit_success = False
        if search:
            # Fit it using lmfit

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
            rms = self.get_rms_from_image()
            self.rms_err = rms
            det_success = np.abs(fit_values[0] / self.rms_err) >= 3
            fit_success = conv_success & covar_success & det_success
            if fit_success:
                self.fit_success = True

                # This means that the fit has converged and is successful
                fit_errs = np.sqrt(np.diag(res.covar))

                self.fit = fit_values
                self.fit_err = fit_errs

        if (not search) or (not self.fit_success):
            if not self.fit_success:
                # If not the fit has failed to converge, so then
                # do a beam average of the flux and err
                pass
            # do a beam average of the flux and err
            if self.rms_flag:
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
                self.fit = [
                    data[center[0], center[1]],
                    self.center[0],
                    self.center[1],
                    self.bmaj_pix,
                    self.bmin_pix,
                    self.pos_ang,
                ]
                rms = self.get_rms_from_image()
                self.fit_err = [rms, np.nan, np.nan, np.nan, np.nan, np.nan]
                self.rms_err = rms
            self.condon_err = None
            # self.fit_pos = self.coords
            self.fit_shape = np.array([self.bmaj_pix, self.bmin_pix, self.pos_ang])

        self.psf_model = fmodel.func((X, Y), *self.fit).reshape(X.shape)


if __name__ == "__main__":
    # Get the simulated  data
    peak, loc_x, loc_y, psf_x, psf_y, pol, fake_data = simulate_data(n_images=1000)
    res = []
    res_err = []
    for i, im in enumerate(fake_data):
        f = fitter(data=fake_data[i], sx=psf_x[i], sy=psf_y[i], pol=pol[i])
        f.fit_gaussian()
        res.append(f.fit)
        res_err.append(f.fit_err)

    res = np.array(res)
    res_err = np.array(res_err)
    chi = ((res[:, 0] - peak) / res_err[:, 0]) ** 2

    plt.errorbar(
        peak,
        res[:, 0],
        yerr=res_err[:, 0],
        fmt=".",
        c="k",
        capsize=2,
        barsabove=True,
        label="this algorithm",
    )
    plt.plot(
        np.linspace(0, 25, 100),
        np.linspace(0, 25, 100),
        ls="--",
        color="r",
        label="expected",
    )
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=15)

    plt.ylabel("Recovered flux density", fontsize=20)
    plt.xlabel("Injected flux density", fontsize=20)

    plt.title(r"Recovery test for this routine ($\chi^2$=1.19, DOF=999)", fontsize=20)
    plt.tight_layout()
    plt.savefig("injection.pdf")
