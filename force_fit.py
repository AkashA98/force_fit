from astropy.table import Table
import numpy as np, glob, os
from astropy import units as u
from astropy import constants as c
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy, warnings
from matplotlib import pyplot as plt
import pickle
from lmfit import Model
from lmfit import create_params

warnings.filterwarnings(
    action="ignore", category=astropy.wcs.FITSFixedWarning, module="astropy"
)


def get_stokes_pointings(basepath, rms=False):
    if not rms:
        paths = glob.glob(f"{basepath}/*.fits")
    else:
        paths = glob.glob(f"{basepath}/*.fits")
    paths.sort()

    # Put all the WCS objects
    hdul = [fits.open(i) for i in paths]
    all_wcs = [WCS(hdu[0].header) for hdu in hdul]
    center = np.array(
        [
            w.celestial.pixel_to_world(w.pixel_shape[0] // 2, w.pixel_shape[1] // 2)
            for w in all_wcs
        ]
    )
    centers = np.array([[i.ra.degree, i.dec.degree] for i in center])
    all_centers = SkyCoord(ra=centers[:, 0] * u.degree, dec=centers[:, 1] * u.degree)
    for i in hdul:
        del i
    del hdul
    return all_centers, np.array(all_wcs), np.array(paths)


def get_correct_file(all_centers, all_wcs, all_paths, p, rms=False):
    mask = p.separation(all_centers).degree < 10
    right_file = 0
    correct_file_mask = []
    for i in range(len(all_paths[mask])):
        try:
            w = all_wcs[mask][i]
            ppos = w.celestial.world_to_pixel(p)
            ppos = np.array([int(ppos[0].item()), int(ppos[1].item())])
            data = fits.getdata(all_paths[mask][i], 0)
            cut = Cutout2D(data[0][0], ppos, size=10 * u.arcsec, wcs=w.celestial)
            right_file += 1
            im_center = w.celestial.pixel_to_world([w.wcs.crpix[0]], [w.wcs.crpix[1]])
            if p.separation(im_center).degree[0] <= 3.5:
                correct_file_mask.append(1)
            else:
                correct_file_mask.append(0)
        except astropy.nddata.utils.NoOverlapError:
            correct_file_mask.append(0)
            continue

    if right_file == 0:
        if not rms:
            return None
        else:
            return None
    else:
        print(all_paths[mask][correct_file_mask])
        return all_paths[mask][correct_file_mask]


def get_cut_out_data(hdu, s, size=3 * u.arcmin):
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


def gaussian(X, A, x0, y0, sx, sy, pa):
    x, y = X
    theta = np.radians(pa)

    a = np.cos(theta) ** 2 / (2 * sx**2) + np.sin(theta) ** 2 / (2 * sy**2)
    b = np.sin(theta) ** 2 / (2 * sx**2) + np.cos(theta) ** 2 / (2 * sy**2)
    c = np.sin(2 * theta) / (2 * sx**2) - np.sin(2 * theta) / (2 * sy**2)

    power = -a * (x - x0) ** 2 - b * (y - y0) ** 2 - c * (x - x0) * (y - y0)

    return np.ravel(A * np.exp(power))


def fit_gaussian(data_im, data_rms=None, data_bkg=None):
    data = np.copy(data_im)

    x, y = np.arange(len(data)), np.arange(len(data[0]))
    X, Y = np.meshgrid(x, y)
    p0 = (np.nanmax(data), len(data) / 2, len(data[0]) / 2, 1, 1, 45)
    pmax = (2 * np.nanmax(data), len(data) / 2 + 2, len(data[0]) / 2 + 2, 3, 3, 360)
    pmin = (0, len(data) / 2 - 2, len(data[0]) / 2 - 2, 0, 0, 0)

    # Dont fit anything if the data is nan
    if np.all(np.isnan(data)) or np.sum(data) == 0:
        return (
            (X, Y),
            None,
            np.ones_like(X) * np.nan,
            np.zeros_like(p0) * np.nan,
            np.zeros_like(p0),
        )
    else:
        # Fit it using lmfit
        fmodel = Model(gaussian, independent_vars=("X"))

        # Handle nan values
        data[np.isnan(data)] = 0

        params = create_params(
            A=dict(value=p0[0], min=pmin[0], max=pmax[0]),
            x0=dict(value=p0[1], min=pmin[1], max=pmax[1]),
            y0=dict(value=p0[2], min=pmin[2], max=pmax[2]),
            sx=dict(value=p0[3], min=pmin[3], max=pmax[3]),
            sy=dict(value=p0[4], min=pmin[4], max=pmax[4]),
            pa=dict(value=p0[5], min=pmin[5], max=pmax[5]),
        )

        if data_bkg is None:
            if data_rms is None:
                res = fmodel.fit(np.ravel(data), params, X=(X, Y))
            else:
                data_rms[np.isnan(data_rms)] = 1e8
                res = fmodel.fit(
                    np.ravel(data), params, X=(X, Y), weights=1 / np.ravel(data_rms)
                )
        else:
            data_bkg[np.isnan(data_bkg)] = 0
            if data_rms is None:
                res = fmodel.fit(np.ravel(data) - np.ravel(data_bkg), params, X=(X, Y))
            else:
                data_rms[np.isnan(data_rms)] = 1e8
                res = fmodel.fit(
                    np.ravel(data) - np.ravel(data_bkg),
                    params,
                    X=(X, Y),
                    weights=1 / np.ravel(data_rms),
                )

        fit_values = list(res.best_values.values())
        if res.covar is not None:
            fit_errs = np.sqrt(np.diag(res.covar))
        else:
            fit_errs = np.ones_like(fit_values) * np.nan

        model = fmodel.func((X, Y), **res.best_values).reshape(X.shape)

        return (X, Y), res, model, fit_values, fit_errs


def calculate_errors(hdr, data, rms_data, fit, err):
    bmaj_pix = abs(hdr["BMAJ"] / hdr["CDELT1"]) / 2.35482
    bmin_pix = abs(hdr["BMIN"] / hdr["CDELT2"]) / 2.35482

    # Now calculate errors using eqn 41 of Condon 1997
    snr_2_c1 = fit[3] * fit[4] / (4 * bmaj_pix * bmin_pix)
    snr_2_c2 = 1 + (bmaj_pix * bmin_pix / (fit[3] ** 2))
    snr_2_c3 = 1 + (bmaj_pix * bmin_pix / (fit[4] ** 2))

    # estimate RMS of the image
    if (fit[1] > len(data)) or (fit[2] > len(data[0])):
        mu = np.nan
        print("Something wrong with the source fitting, the Gaussian is out of bounds")
    else:
        mu = rms_data[int(len(rms_data) // 2), int(len(rms_data[0]) // 2)]
    snr_2 = snr_2_c1 * (snr_2_c2 ** (1.5)) * (snr_2_c3 ** (1.5)) / (mu**2)
    amp_err = np.sqrt(2 / snr_2)
    return mu, amp_err


def convert_fit_values_to_astrometric(fit, wcs, hdr):
    pos = wcs.celestial.pixel_to_world([fit[1]], [fit[2]])
    maj = np.max([fit[3], fit[4]])
    minor = np.min([fit[3], fit[4]])
    fwhm_maj = maj * hdr["CDELT1"] * 2.35482
    fwhm_min = minor * hdr["CDELT2"] * 2.35482

    return pos, fwhm_maj, fwhm_min


def get_rms_from_image(data):
    # Calculate errors
    # Since there is no error map, calculate by hand
    l = len(data)
    rms1 = np.nanstd(data[0 : l // 2 - 10, 0 : l // 2 - 10])
    rms2 = np.nanstd(data[0 : l // 2 - 10, l // 2 + 10 : l])
    rms3 = np.nanstd(data[l // 2 + 10 : l, 0 : l // 2 - 10])
    rms4 = np.nanstd(data[l // 2 + 10 : l, l // 2 + 10 : l])
    rms = 0.25 * (rms1 + rms2 + rms3 + rms4)
    print(rms1, rms2, rms3, rms4, rms)


def get_racs_flux(all_centers, all_wcs, all_paths, s, racs=True, size=1 * u.arcmin):
    all_files = get_correct_file(all_centers, all_wcs, all_paths, s)
    if all_files is not None:
        res = []
        for i, p in enumerate(all_files):
            if racs:
                p_rms = p.replace("STOKESV_IMAGES", "STOKESV_RMSMAPS")
                p_bkg = p_rms.replace(".fits", ".bkg.fits")
                p_rms = p_rms.replace(".fits", ".rms.fits")
            else:
                p_rms = p.replace("STOKESI_IMAGES", "STOKESI_RMSMAPS")
                p_bkg = p_rms.replace("VAST_", "meanMap.VAST_")
                p_rms = p_rms.replace("VAST_", "noiseMap.VAST_")

            hdu_im = fits.open(p)
            cut_data_im = get_cut_out_data(hdu_im, s, size=size)

            if os.path.isfile(p_bkg):
                hdu_bkg = fits.open(p_bkg)
                cut_data_bkg = get_cut_out_data(hdu_bkg, s, size=size)
            else:
                cut_data_bkg = None

            if os.path.isfile(p_rms):
                hdu_rms = fits.open(p_rms)
                cut_data_rms = get_cut_out_data(hdu_rms, s, size=size)
            else:
                cut_data_rms = None

            if np.all(np.isnan(cut_data_im.data)):
                return None
            else:
                # Do the actual fitting
                X, _, model, fit_values, fit_errs = fit_gaussian(
                    cut_data_im.data, cut_data_rms.data, cut_data_bkg.data
                )
                pos, fwhm_maj, fwhm_min = convert_fit_values_to_astrometric(
                    fit_values, cut_data_im.wcs, hdu_rms[0].header
                )
                rms, err = calculate_errors(
                    hdu_rms[0].header,
                    cut_data_im.data,
                    cut_data_rms.data,
                    fit_values,
                    fit_errs,
                )

                # Do the plotting
                plt.clf()
                fig = plt.figure(figsize=(6, 6))
                plt.subplot(projection=cut_data_im.wcs)
                plt.imshow(
                    cut_data_im.data, aspect="auto", origin="lower", cmap="gray_r"
                )
                levels = (
                    np.array(
                        [0.049787068367863944, 0.1353352832366127, 0.36787944117144233]
                    )
                    * fit_values[0]
                )
                if fit_values[0] < 0:
                    levels = np.flip(levels)
                plt.contour(X[0], X[1], model, levels=levels, colors="r")
                src_pos = cut_data_im.wcs.celestial.world_to_pixel(s)
                plt.plot(
                    [src_pos[0], src_pos[0]],
                    [src_pos[1] + 1, src_pos[1] + 2],
                    color="r",
                )
                plt.plot(
                    [src_pos[0], src_pos[0]],
                    [src_pos[1] - 2, src_pos[1] - 1],
                    color="r",
                )
                plt.plot(
                    [src_pos[0] - 2, src_pos[0] - 1],
                    [src_pos[1], src_pos[1]],
                    color="r",
                )
                plt.plot(
                    [src_pos[0] + 1, src_pos[0] + 2],
                    [src_pos[1], src_pos[1]],
                    color="r",
                )
                plt.tick_params(labelsize=15)
                plt.grid(color="k", ls="--")
                # plt.show()
                plt.savefig(f"J{s.to_string('hmsdms')}.pdf", bbox_inches="tight")

                # print(
                #     f"A component is fit at {pos.to_string('hmsdms')}; {pos.separation(s).arcsec} arcsec away from the requested coordinates"
                # )
                # print(f"The peak flux estimated is {fit_values[0]*1000} mJy")
                # print(f"The RMS value of the map is {rms*1000} mJy")
                # print(
                #     f"The error corrected for correlations (Condon 1997) is {err*1000} mJy"
                # )
                # print(f"SNR of the source is {fit_values[0]/rms}")

                res.append(
                    [
                        hdu_im[0].header["DATE-OBS"],
                        fit_values[0] * 1000,
                        rms * 1000,
                        err * 1000,
                        pos.separation(s).acrsec,
                    ]
                )
        if len(res) > 0:
            return res
        else:
            return None
    else:
        return None


def make_pickle_files():
    # all_centers = {}
    # all_wcs = {}
    # all_paths = {}
    centers, wcs, paths = get_stokes_pointings(
        "/home/aakash/Data/aakash/RACS/STOKESV_IMAGES"
    )
    # all_centers[e] = centers
    # all_wcs[e] = wcs
    # all_paths[e] = paths

    # write pickle files

    with open("all_centers.pkl", "wb") as f:
        pickle.dump(centers, f)

    with open("all_wcs.pkl", "wb") as f:
        pickle.dump(wcs, f)

    with open("all_paths.pkl", "wb") as f:
        pickle.dump(paths, f)
    return None


if __name__ == "__main__":
    # Read the pickle files
    # make_pickle_files()
    # Try reading them
    with open("all_centers.pkl", "rb") as f:
        all_centers = pickle.load(f)

    with open("all_wcs.pkl", "rb") as f:
        all_wcs = pickle.load(f)

    with open("all_paths.pkl", "rb") as f:
        all_paths = pickle.load(f)

    tdes = Table.read("tns_search.csv", format="ascii")
    tde_coords = SkyCoord(
        ra=tdes["RA"].data, dec=tdes["DEC"].data, unit=(u.hourangle, u.degree)
    )

    tab = Table(
        names=["Name", "date", "flux", "rms", "err", "snr", "sep"],
        dtype=["U32", "U32", "f4", "f4", "f4", "f4", "f4", "f4"],
    )
    for i in range(len(tdes)):
        res = get_racs_flux(
            all_centers=all_centers,
            all_wcs=all_wcs,
            all_paths=all_paths,
            s=tde_coords[i],
            size=2 * u.arcmin,
        )
        if res is not None:
            tab.add_row(
                [
                    tdes["Name"][i],
                    res[0],
                    res[1],
                    res[2],
                    res[3],
                    res[1] / res[3],
                    res[4],
                ]
            )
