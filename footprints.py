import os
import glob
import pickle
import astropy
import warnings
import numpy as np
from loguru import logger
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

warnings.filterwarnings(
    action="ignore", category=astropy.wcs.FITSFixedWarning, module="astropy"
)


class vast_footprint:
    """
    Class that takes care of the VAST footprints
    """

    def __init__(self, stokes="I") -> None:
        """Initialize a the class tha reads all the epochs

        Args:
            stokes (str, optional): sotesk parameter to search for.
            Defaults to "I". Can give "I" or "V".

        Returns:
            None : None
        """
        self.vast_root = (
            f"/data/vast-survey/VAST/vast-data/TILES/STOKES{stokes}_IMAGES/"
        )
        self.vast_rms_path = (
            f"/data/vast-survey/VAST/vast-data/TILES/STOKES{stokes}_IMAGES/"
        )
        self.racs_root = "/data/vast-survey/RACS/release-format/"
        self.racs_tail = f"/TILES/STOKES{stokes}_IMAGES/"
        self.racs_rms_tail = f"/TILES/STOKES{stokes}_BANE/"
        self.racs_epochs = ["EPOCH00", "EPOCH29"]
        self.stokes = stokes

        return None

    def _fetch_epochs(self, rms=True):
        """
        Function to fetch all the epochs in the vast data

        Returns:
        dict: dictionary of all the epochs and their paths
        """
        epochs_dir = np.array(glob.glob(f"{self.vast_root}/*"))
        epoch_names = np.array([i.split("/")[-1] for i in epochs_dir])
        self.raw_epoch_names = epoch_names
        logger.info(f"The following epochs are found: {epoch_names}")
        self._validate_epochs()  # Removes all the rubbish epochs
        logger.info(f"Removing the following epochs: {epoch_names[~self.epoch_mask]}")

        order_mask = np.argsort(self.epoch_names)
        epochs = dict(
            zip(self.epoch_names[order_mask], epochs_dir[self.epoch_mask][order_mask])
        )

        all_files = {}
        # Now assert for RMS MAPS
        for e in list(epochs.keys()):
            rmspath = epochs[e].replace(
                f"STOKES{self.stokes}_IMAGES", f"STOKES{self.stokes}_RMSMAPS"
            )
            if os.path.isdir(rmspath):
                all_files[e] = {"images": epochs[e], "rms": rmspath}
            else:
                all_files[e] = {"images": epochs[e], "rms": None}

        self.epochs = all_files
        logger.info("Asserting all the file paths")

        # Modify the input directories for RACS
        self._check_for_racs_epochs()  # change the paths for RACS epochs
        self.epoch_names = np.array(list(self.epochs.keys()))

        return None

    def _validate_epochs(self):
        """Validate a given set of epochs. The epoch names provided
        will be of the form epoch_xx

        Returns:
            array, array: boolean mask of legitimate epochs, epochs renamed
        """
        names = [i.split("_")[-1] for i in self.raw_epoch_names]
        proper_names = ["EPOCH" + i.rjust(2, "0") for i in names]

        # Check for valid epochs
        name_mask = []
        file_mask = []
        for i, n in enumerate(names):
            try:
                _ = int(n)
                name_mask.append(1)
            except ValueError:
                name_mask.append(0)
            
            # Check if images exist
            images = glob.glob(f"{self.vast_root}/{self.raw_epoch_names[i]}/*")
            if len(images)>0:
                file_mask.append(1)
            else:
                file_mask.append(0)
                
        name_mask = np.array(name_mask, dtype=bool)
        file_mask = np.array(file_mask, dtype=bool)
        self.epoch_mask = (name_mask & file_mask)
        self.epoch_names = np.array(proper_names)[self.epoch_mask]
        return None

    def _check_for_racs_epochs(self):
        """
        Check for the RACS epochs and get approriate file paths
        """
        for i in self.racs_epochs:
            racs_path = f"{self.racs_root}{i}{self.racs_tail}"
            racs_rms_path = f"{self.racs_root}{i}{self.racs_rms_tail}"
            if os.path.isdir(racs_path):
                d = {
                    "images": racs_path,
                    "rms": racs_rms_path if os.path.isdir(racs_rms_path) else None,
                }
                self.epochs[i] = d
                logger.info(f"Adding the following RACS epoch: {i}")
        return None


def get_vast_pointings(basepath):
    """Function to get the information about all the files in a given path

    Args:
        basepath (str): The base folder for all the image fits files

    Returns:
        array: An array of all the centers
        array: An array of all the WCS objects
        array: An array of all the paths to the files
    """
    # Now get info for the required epochs
    paths = glob.glob(f"{basepath}/*.fits")
    paths.sort()
    if paths==[]:
        raise FileNotFoundError
    else:    
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


def pickle_pointing_info(vast):
    """Generate pickle files about the individual tiles and their
    pointings for all the epochs

    Args:
        vast (class): The vast_footprint class object

    Returns:
        None: None
    """
    epoch_names = vast.epoch_names
    mask = np.ones(len(epoch_names)).astype(bool)
    # Pickle them all
    if not os.path.isdir(f"pickles_{vast.stokes}/"):
        os.mkdir(f"pickles_{vast.stokes}/")
        mask = ~mask
        logger.info(f"Making a pickle directory: pickles_{vast.stokes}/")
        logger.info("There are no pickle files, so making all of them")
    else:
        for i, e in enumerate(epoch_names):
            if os.path.isfile(f"pickles_{vast.stokes}/{e}.pkl"):
                continue
            else:
                mask[i] &= False
        logger.info(
            f"Pickle files for the follwoing epochs are missing: {epoch_names[~mask]}"
        )

    # For epochs that are missing, write put the pickle files
    for e in epoch_names[~mask]:
        try:
            centers, wcs, paths = get_vast_pointings(vast.epochs[e]["images"])
            epoch_pointings = {"centers": centers, "wcs": wcs, "paths": paths}

            with open(f"pickles_{vast.stokes}/{e}.pkl", "wb") as f:
                pickle.dump(epoch_pointings, f)
                logger.info(
                    f"Making pickle files for the epoch {e}: pickles_{vast.stokes}/{e}.pkl"
                )
                f.close()
        except FileNotFoundError:
            logger.info(
                f"Making pickle files for the epoch {e} failed, no images found"
            )

    return None


def get_correct_file(vast, epoch: str, p: astropy.coordinates.sky_coordinate.SkyCoord):
    """Function to return the image files that have the given source
    coordinates. Read the pickle files and get the info

    Args:
        vast (class): vast_footprint class object
        epoch (str): The epoch for which the files are searches
        p (astropy.coordinates.sky_coordinate.SkyCoord): source coordinates

    Returns:
        dict: dictionary of matched files
    """
    # First parse the epoch name correctly
    # logger.info(f"The follwoing coordinates are given {p.to_string('hmsdms')}")
    epoch = "EPOCH" + epoch.rjust(2, "0")
    logger.info(f"Searching for files in the epoch {epoch}")
    with open(f"pickles_{vast.stokes}/{epoch}.pkl", "rb") as f:
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
    match_files = {}
    nfiles = 0
    if num_files > 0:
        match_files["primary_field"] = primary_field
        match_files["primary"] = all_files[primary_field]
        nfiles += 1
        logger.info(f"The primary filed for the epoch {epoch} is {primary_field}")
        logger.info(
            f"Distance from center in the primary filed {primary_field_sep} deg"
        )
        for i in list(all_files.keys()):
            if i != primary_field:
                nfiles += len(all_files[i])
                try:
                    match_files["other"] = np.concatenate(
                        (match_files["other"], all_files[i])
                    )
                except KeyError:
                    match_files["other"] = all_files[i]
    else:
        match_files["primary"] = []

    logger.info(
        f"{nfiles} files found in total having the given position in their field"
    )
    return match_files
