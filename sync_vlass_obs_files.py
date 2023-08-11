import os
import requests
import numpy as np
from bs4 import BeautifulSoup
from astropy.table import Table, vstack

vlass_base_url = "https://archive-new.nrao.edu/vlass/quicklook/"


def get_links(url):
    """Takes an URL, unloads the contents using Beautifulsoup and then
        selects only the observing epochs for vlass

    Args:
        url (str): Webpage url
    """
    r = requests.get(url)
    data = BeautifulSoup(r.content, "html.parser")
    hrefs = data.find_all("a", href=True)

    links = []
    for h in hrefs:
        if h.attrs["href"] == h.contents[0]:
            links.append(h.attrs["href"])

    return links


def get_tiles(url):
    """Get the VLASS tiles per observing epoch

    Args:
        url (str): VLASS base url
    """
    epochs = get_links(vlass_base_url)
    epochs = [i for i in epochs if "VLASS" in i]

    tiles = {}
    for e in epochs:
        epoch_url = url + "/" + e
        epoch_tiles = get_links(epoch_url)
        epoch_tiles.remove("QA_REJECTED/")
        for x in ["images.lis", "new_keywords.csv", "old_keywords.csv"]:
            if x in epoch_tiles:
                epoch_tiles.remove(x)
        tiles[e] = np.array(epoch_tiles)
    return tiles


def get_files(url):
    """Get all the VLASS observing files per tile per epoch

    Args:
        url (str): VLASS base url
    """
    tiles = get_tiles(url=url)
    epochs = list(tiles.keys())

    if not os.path.isdir("./vlass_pointing_files"):
        os.mkdir("./vlass_pointing_files")

    for e in epochs:
        name = e.replace("/", "")
        tab = Table(names=["tile", "field"], dtype=["U10", "U80"])
        for t in tiles[e]:
            tile_url = url + "/" + e + "/" + t
            fields = get_links(tile_url)
            if len(fields) > 0:
                t = Table([np.repeat(t, len(fields)), fields], names=["tile", "field"])
                tab = vstack([tab, t], join_type="exact")
        tab.write(f"./vlass_pointing_files/{name}.csv", overwrite=True)


if __name__ == "__main__":
    get_files(vlass_base_url)
