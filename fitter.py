import numpy as np
from astropy import units as u
from lmfit import Model, create_params

nan_mask_value = 1e8
fwhm_to_sig = np.sqrt(8 * np.log(2))


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


def get_psf_info(hdr):
    """
    Function to get the information abou the point-spread function
    """
    try:
        bmaj_pix = abs(hdr["BMAJ"] / hdr["CDELT1"]) / fwhm_to_sig
        bmin_pix = abs(hdr["BMIN"] / hdr["CDELT2"]) / fwhm_to_sig
    except KeyError:
        bmaj_pix = abs(hdr["BMAJ"] / hdr["CD1_1"]) / fwhm_to_sig
        bmin_pix = abs(hdr["BMIN"] / hdr["CD2_2"]) / fwhm_to_sig
    pos_ang = hdr["BPA"] * u.degree

    return bmaj_pix, bmin_pix, pos_ang


def get_forced_flux(data_im, hdr, data_rms, data_bkg):
    """
    Function to get the forced flux instead of fitting the position.
    Similar to the Kaplan+Andrew force fitting routine.
    """
    data = np.copy(data_im)

    # Make a pixel grid for the image
    x, y = np.arange(len(data)), np.arange(len(data[0]))
    X, Y = np.meshgrid(x, y)
    sx, sy, pa = get_psf_info(hdr=hdr)
    norm = 1  # snp.cos(2 * pa) / 2 / np.pi / sx / sy
    kernel = gaussian_kernel(
        (X, Y), norm, data.shape[0] / 2, data.shape[1] / 2, sx, sy, pa
    )

    # Do a noise weighted sum
    flux = ((data - data_bkg) * kernel / data_rms**2).sum() / (
        kernel**2 / data_rms**2
    ).sum()
    flux_err = ((data_rms) * kernel / data_rms**2).sum() / (
        kernel / data_rms**2
    ).sum()

    return flux, flux_err


def calculate_errors(hdr, data, fit, rms_data=None):
    """
    Function to calculate fit errors according to Condon 1997
    """
    bmaj_pix, bmin_pix, _ = get_psf_info(hdr=hdr)

    # Now calculate errors using eqn 41 of Condon 1997
    _, x0, y0, sx, sy, _ = fit
    snr_2_c1 = sx * sy / (4 * bmaj_pix * bmin_pix)
    snr_2_c2 = 1 + (bmaj_pix * bmin_pix / (sx**2))
    snr_2_c3 = 1 + (bmaj_pix * bmin_pix / (sy**2))

    # estimate RMS of the image
    if (x0 > len(data)) or (y0 > len(data[0])):
        mu = np.nan
        print("Something wrong with the source fitting, the Gaussian is out of bounds")
    elif rms_data is not None:
        mu = rms_data[int(len(rms_data) // 2), int(len(rms_data[0]) // 2)]
    else:
        mu = get_rms_from_image(data=data)

    snr_2 = snr_2_c1 * (snr_2_c2 ** (1.5)) * (snr_2_c3 ** (1.5)) / (mu**2)
    amp_err = np.sqrt(2 / snr_2)

    return mu, amp_err


def convert_fit_values_to_astrometric(fit, wcs, hdr):
    """
    Convert from pixel space to astrometric space
    """
    _, x0, y0, sx, sy, _ = fit
    pos = wcs.celestial.pixel_to_world([x0], [y0])
    maj = np.max([sx, sy])
    minor = np.min([sx, sy])
    try:
        pix_scl_maj = hdr["CDELT1"]
        pix_scl_min = hdr["CDELT2"]
    except KeyError:
        pix_scl_maj = hdr["CD1_1"]
        pix_scl_min = hdr["CD2_2"]
    fwhm_maj = maj * pix_scl_maj * fwhm_to_sig
    fwhm_min = minor * pix_scl_min * fwhm_to_sig

    return pos, fwhm_maj, fwhm_min


def get_rms_from_image(data):
    # Calculate errors from the image itself
    # Since there is no error map, calculate by hand
    l = len(data)
    rms1 = np.nanstd(data[0 : l // 2 - 10, 0 : l // 2 - 10])
    rms2 = np.nanstd(data[0 : l // 2 - 10, l // 2 + 10 : l])
    rms3 = np.nanstd(data[l // 2 + 10 : l, 0 : l // 2 - 10])
    rms4 = np.nanstd(data[l // 2 + 10 : l, l // 2 + 10 : l])
    rms = 0.25 * (rms1 + rms2 + rms3 + rms4)

    return rms


def fit_gaussian(data_im, hdr, data_rms=None, data_bkg=None, search=True):
    """
    Do the 2D gaussian fitting using the lmfit package
    """

    data = np.copy(data_im)

    # Make a pixel grid for the image
    x, y = np.arange(len(data)), np.arange(len(data[0]))
    X, Y = np.meshgrid(x, y)

    # Make an initial guess and the bounds for the parameters
    # Source will always be at the center of the data
    data_max = np.max(np.abs(data))
    if (
        np.mean(
            data[
                data.shape[0] // 2 - 2 : data.shape[0] // 2 + 3,
                data.shape[1] // 2 - 2 : data.shape[1] // 2 + 3,
            ]
        )
        < 0
    ):
        p0 = (-data_max, len(data) / 2, len(data[0]) / 2, 1, 1, 45)
    else:
        p0 = (data_max, len(data) / 2, len(data[0]) / 2, 1, 1, 45)
    pmax = (2 * data_max, len(data) / 2 + 2, len(data[0]) / 2 + 2, 3, 3, 360)
    pmin = (-2 * data_max, len(data) / 2 - 2, len(data[0]) / 2 - 2, 0, 0, 0)

    # Correct the data for nan's
    # Handle nan values. This is a bad way of handling nan values
    # but currently the only way
    data[np.isnan(data)] = 0

    # Do the same for noise data
    if data_bkg is None:
        data_bkg = np.zeros_like(data)
    else:
        # Deal with nan values again
        data_bkg[np.isnan(data_bkg)] = 0

    sub_data = data - data_bkg

    if data_rms is None:
        data_rms = np.ones_like(data)
    else:
        # Set nan values in rms map so high, that weight is 0
        data_rms[np.isnan(data_rms)] = nan_mask_value

    # Dont fit anything if the data is nan
    if np.sum(np.abs(data)) == 0:
        return (
            (X, Y),
            None,
            np.ones_like(X) * np.nan,
            np.zeros_like(p0) * np.nan,
            np.zeros_like(p0),
        )
    else:
        # decide whether to search around or not
        if search:
            # Fit it using lmfit
            fmodel = Model(gaussian_kernel, independent_vars=("X"))

            params = create_params(
                A=dict(value=p0[0], min=pmin[0], max=pmax[0]),
                x0=dict(value=p0[1], min=pmin[1], max=pmax[1]),
                y0=dict(value=p0[2], min=pmin[2], max=pmax[2]),
                sx=dict(value=p0[3], min=pmin[3], max=pmax[3]),
                sy=dict(value=p0[4], min=pmin[4], max=pmax[4]),
                pa=dict(value=p0[5], min=pmin[5], max=pmax[5]),
            )

            res = fmodel.fit(
                np.ravel(sub_data), params, X=(X, Y), weights=1 / np.ravel(data_rms)
            )

            fit_success = [True if res.summary()["ier"] <= 1 else False][0]

            if fit_success:
                # This means that the fit has converged and is successful
                fit_values = np.array(list(res.best_values.values()))
                fit_errs = np.sqrt(np.diag(res.covar))
                if data_rms is None:
                    rms, condon_err = calculate_errors(hdr, data, fit_values)
                else:
                    rms, condon_err = calculate_errors(
                        hdr, data, fit_values, rms_data=data_rms
                    )
            else:
                # If not the fit has failed to converge, so then
                # do a beam average of the flux and err
                flux, rms = get_forced_flux(
                    data_im=data, hdr=hdr, data_bkg=data_bkg, data_rms=data_rms
                )
                sx, sy, pa = get_psf_info(hdr=hdr)
                fit_values = np.array(
                    [flux, data.shape[0] / 2, data.shape[1] / 2, sx, sy, pa]
                )
                fit_errs = np.array([rms, np.nan, np.nan, np.nan, np.nan, np.nan])
                condon_err = None
        else:
            # do a beam average of the flux and err
            flux, rms = get_forced_flux(
                data_im=data, hdr=hdr, data_bkg=data_bkg, data_rms=data_rms
            )
            sx, sy, pa = get_psf_info(hdr=hdr)
            fit_values = np.array(
                [flux, data.shape[0] / 2, data.shape[1] / 2, sx, sy, pa]
            )
            fit_errs = np.array([rms, np.nan, np.nan, np.nan, np.nan, np.nan])
            condon_err = None

        model = fmodel.func((X, Y), **fit_values).reshape(X.shape)

        return (X, Y), model, fit_values, fit_errs, condon_err
