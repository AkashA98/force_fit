import os
import glob
import pickle
import astropy
import warnings
from astropy.wcs import WCS
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

warnings.filterwarnings(
    action="ignore", category=astropy.wcs.FITSFixedWarning, module="astropy"
)


def validate_epochs(epochs):
    """
    Validate a given set of epochs. The epoch names provided
    will be of the form epoch_xx
    """
    names = [i.split("_")[-1] for i in epochs]
    proper_names = ["EPOCH" + i.rjust(2, "0") for i in names]
    proper_names[proper_names.index("EPOCH14")] = "EPOCH29"

    # Check for valid epochs
    mask = []
    for n in names:
        try:
            intn = int(n)
            mask.append(1)
        except ValueError:
            mask.append(0)
    mask = np.array(mask, dtype=bool)
    return mask, np.array(proper_names)


def fetch_epochs():
    """
    Function to fetch all the epochs in the vast data
    """
    root_dir = "/data/vast-survey/VAST/vast-data/TILES/STOKESI_IMAGES/"
    epochs_dir = np.array(glob.glob(root_dir + "/*"))
    epoch_names = [i.split("/")[-1] for i in epochs_dir]
    epoch_mask, alt_names = validate_epochs(epoch_names)

    order_mask = np.argsort(alt_names[epoch_mask])
    epochs = dict(
        zip(alt_names[epoch_mask][order_mask], epochs_dir[epoch_mask][order_mask])
    )

    # Modify the input directories for RACS
    epochs[
        "EPOCH00"
    ] = "/data/vast-survey/RACS/release-format/EPOCH00/TILES/STOKESI_IMAGES/"
    epochs[
        "EPOCH29"
    ] = "/data/vast-survey/RACS/release-format/EPOCH29/TILES/STOKESI_IMAGES/"
    return epochs


def get_vast_pointings(basepath):
    paths = glob.glob(f"{basepath}/*.fits")
    paths.sort()

    # Put all the WCS objects
    all_paths = []
    all_wcs = []
    all_centers = []

    for p in paths:
        hdu = fits.open(p)
        wcs = WCS(hdu[0].header)
        if wcs.naxis == 4:
            wcs_cel = wcs.celestial
        center = wcs_cel.pixel_to_world(
            wcs.pixel_shape[0] // 2, wcs.pixel_shape[1] // 2
        )
        all_paths.append(p)
        all_wcs.append(wcs)
        all_centers.append([center.ra.degree, center.dec.degree])
        hdu.close()

    all_centers = np.array(all_centers)
    all_centers = SkyCoord(
        ra=all_centers[:, 0] * u.degree, dec=all_centers[:, 1] * u.degree
    )

    return all_centers, np.array(all_wcs), np.array(paths)


def pickle_pointing_info(epochs):
    """
    Generate pickle files about the individual tiles and their
    pointings for all the epochs
    """
    epoch_names = np.array(list(epochs.keys()))
    mask = np.ones(len(epoch_names)).astype(bool)
    # Pickle them all
    if not os.path.isdir("pickles/"):
        os.mkdir("pickles/")
    else:
        for i, e in enumerate(epoch_names):
            if os.path.isfile(f"pickles/{e}.pkl"):
                continue
            else:
                mask[i] &= False

    # For epochs that are missing, write put the pickle files
    for e in epoch_names[~mask]:
        epoch_pontings = {}
        centers, wcs, paths = get_vast_pointings(epochs[e])
        epoch_pontings["centers"] = centers
        epoch_pontings["wcs"] = wcs
        epoch_pontings["paths"] = paths

        with open(f"pickles/{e}.pkl", "wb") as f:
            pickle.dump(epoch_pontings, f)
            f.close()

    return None


def get_correct_file(epoch: str, p: astropy.coordinates.sky_coordinate.SkyCoord):
    """
    Function to return the image files that have the given source
    coordinates. Read the pickle files and get the info
    """
    # First parse the epoch name correctly
    epoch = "EPOCH" + epoch.rjust(2, "0")
    with open(f"pickles/{epoch}.pkl", "rb") as f:
        epoch_pkl = pickle.load(f)
        f.close

    # First filter out files that are within 10 degrees to cut
    # short the computation
    centers = epoch_pkl["centers"]
    centers_sep = p.separation(centers).degree
    mask = centers_sep < 10

    num_files = 0
    all_files = {}

    primary_field = None
    primary_field_sep = 10

    for i, path in enumerate(epoch_pkl["paths"][mask]):
        w = epoch_pkl["wcs"][mask][i]
        source_pix = np.array(w.celestial.world_to_pixel(p))
        ref_pix = w.celestial.wcs.crpix

        if np.all(source_pix - 1.8 * ref_pix <= 0) and np.all(source_pix > 0):
            field_name = [
                i
                for i in path.split("/")[-1].split(".")
                if (("RACS" in i) or ("VAST" in i))
            ][0]
            try:
                all_files[field_name].append(path)
            except KeyError:
                all_files[field_name] = [path]

            num_files += 1
        else:
            continue

        # Check if it is primary field or not
        if centers_sep[mask][i] <= primary_field_sep:
            primary_field = field_name
            primary_field_sep = centers_sep[mask][i]

    # Now rewrite the dictionary into primary and other fields
    if num_files > 0:
        match_files = {}
        match_files["primary"] = all_files[primary_field]
        for i in list(all_files.keys()):
            if i != primary_field:
                try:
                    match_files["other"] = np.concatenate(
                        (match_files["other"], all_files[i])
                    )
                except KeyError:
                    match_files["other"] = all_files[i]

        return match_files
    else:
        return None
