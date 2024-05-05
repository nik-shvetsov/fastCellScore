import os
from pathlib import Path
import gdown
from zipfile import ZipFile
from glob import glob
import copy
import shutil
import urllib
import torch
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from xml.dom import minidom
from skimage import draw


def rm_n_mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def pad_array_to_patch_size(np_arr, patch_size):
    pad_x = (
        0
        if np_arr.shape[0] % patch_size == 0
        else patch_size - (np_arr.shape[0] % patch_size)
    )
    pad_y = (
        0
        if np_arr.shape[1] % patch_size == 0
        else patch_size - (np_arr.shape[1] % patch_size)
    )
    arr_padded = np.pad(np_arr, ((0, pad_x), (0, pad_y), (0, 0)), mode="reflect")
    return arr_padded


def download_from_url(url, download_dir, name=None):
    """
    Download a file from a url to destination directory, if file does not exist.

    Args:
        url (str): Url of file to download
        download_dir (str): Directory where file will be downloaded
        name (str, optional): Name of saved file. If ``None``, uses base name of url argument. Defaults to ``None``.
    """
    if name is None:
        name = os.path.basename(url)

    path = os.path.join(download_dir, name)

    if os.path.exists(path):
        return
    else:
        os.makedirs(download_dir, exist_ok=True)

        # Download the file from `url` and save it locally under `file_name`:
        with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


# Used for monusac
def download_data_url(url, rdir, wdir=Path("tmp")):
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    if os.path.exists(rdir):
        shutil.rmtree(rdir)

    zip_path = Path(wdir, "tmpdata.zip")
    ext_path = Path(wdir, "tmpdata")

    gdown.download(url, str(zip_path), quiet=True, fuzzy=True)
    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extractall(path=ext_path)

    for pdir in glob(f"{str(ext_path)}/*"):
        if os.path.isdir(pdir):
            shutil.move(pdir, rdir)
    shutil.rmtree(wdir)


def pannuke_multiclass_mask_to_nucleus_mask(multiclass_mask):
    """
    Convert multiclass mask from PanNuke to a single channel nucleus mask.
    Assumes each pixel is assigned to one and only one class. Sums across channels, except the last mask channel
    which indicates background pixels in PanNuke.
    Operates on a single mask.

    Args:
        multiclass_mask (torch.Tensor): Mask from PanNuke, in classification setting. (i.e. ``nucleus_type_labels=True``).
            Tensor of shape (6, 256, 256).

    Returns:
        Tensor of shape (256, 256).
    """
    # verify shape of input
    assert (
        multiclass_mask.ndim == 3 and multiclass_mask.shape[0] == 6
    ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    assert (
        multiclass_mask.shape[1] == 256 and multiclass_mask.shape[2] == 256
    ), f"Expecting a mask with dims (6, 256, 256). Got input of shape {multiclass_mask.shape}"
    # ignore last channel
    out = np.sum(multiclass_mask[:-1, :, :], axis=0)
    return out


def compute_hv_map(mask):
    """
    Preprocessing step for HoVer-Net architecture.
    Compute center of mass for each nucleus, then compute distance of each nuclear pixel to its corresponding center
    of mass.
    Nuclear pixel distances are normalized to (-1, 1). Background pixels are left as 0.
    Operates on a single mask.
    Can be used in Dataset object to make Dataloader compatible with HoVer-Net.

    Based on https://github.com/vqdang/hover_net/blob/195ed9b6cc67b12f908285492796fb5c6c15a000/src/loader/augs.py#L192

    Args:
        mask (np.ndarray): Mask indicating individual nuclei. Array of shape (H, W),
            where each pixel is in {0, ..., n} with 0 indicating background pixels and {1, ..., n} indicating
            n unique nuclei.

    Returns:
        np.ndarray: array of hv maps of shape (2, H, W). First channel corresponds to horizontal and second vertical.
    """
    assert (
        mask.ndim == 2
    ), f"Input mask has shape {mask.shape}. Expecting a mask with 2 dimensions (H, W)"

    out = np.zeros((2, mask.shape[0], mask.shape[1]))
    # each individual nucleus is indexed with a different number
    inst_list = list(np.unique(mask))

    try:
        inst_list.remove(0)  # 0 is background
    # TODO: change to specific exception
    except Exception:
        print(
            "No pixels with 0 label. This means that there are no background pixels. This may indicate a problem. Ignore this warning if this is expected/intended."
        )

    for inst_id in inst_list:
        # get the mask for the nucleus
        inst_map = mask == inst_id
        inst_map = inst_map.astype(np.uint8)
        contours, _ = cv2.findContours(
            inst_map, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE
        )

        # get center of mass coords
        mom = cv2.moments(contours[0])
        com_x = mom["m10"] / (mom["m00"] + 1e-6)
        com_y = mom["m01"] / (mom["m00"] + 1e-6)
        inst_com = (int(com_y), int(com_x))

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        # add to output mask
        # this works assuming background is 0, and each pixel is assigned to only one nucleus.
        out[0, :, :] += inst_x
        out[1, :, :] += inst_y
    return out


def process_xml_annotations(xml_file_path, img):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Generate n-ary mask for each cell-type
    count = 0
    # result = []
    result = {}
    for k in range(len(root)):
        label = [x.attrib["Name"] for x in root[k][0]]
        label = label[0]

        for child in root[k]:
            for x in child:
                r = x.tag
                if r == "Attribute":
                    label = x.attrib["Name"]
                    n_ary_mask = np.transpose(
                        np.zeros(
                            (img.read_region((0, 0), 0, img.level_dimensions[0]).size)
                        )
                    )

                if r == "Region":
                    regions = []
                    vertices = x[1]
                    coords = np.zeros((len(vertices), 2))
                    for i, vertex in enumerate(vertices):
                        coords[i][0] = vertex.attrib["X"]
                        coords[i][1] = vertex.attrib["Y"]
                    regions.append(coords)
                    try:
                        # may throw error if len(regions[0]) < 4
                        poly = Polygon(regions[0])
                        vertex_row_coords = regions[0][:, 0]
                        vertex_col_coords = regions[0][:, 1]
                        fill_row_coords, fill_col_coords = draw.polygon(
                            vertex_col_coords, vertex_row_coords, n_ary_mask.shape
                        )
                        # Keep track of giving unique values to each instance in an image
                        count = count + 1
                        n_ary_mask[fill_row_coords, fill_col_coords] = count

                    except Exception as e:
                        print(f"Error in {xml_file_path}, {label}, {e}")
                        print(
                            f"Wrong shape for {xml_file_path}, regions[0].shape: {regions[0].shape}"
                        )

        result[label] = n_ary_mask
    return result


def plot_batch_preproc(dataloader, n=4):
    images, masks, hvs, types = next(iter(dataloader))
    cols = ["H&E Image", "Nucleus Types", "Horizontal Map", "Vertical Map"]
    fig, ax = plt.subplots(nrows=n, ncols=len(cols), figsize=(8, 8))

    cm_mask = copy.copy(matplotlib.colormaps["tab10"])
    cm_mask.set_bad(color="white")

    for i in range(n):
        im = images[i, ...].numpy()
        ax[i, 0].imshow(np.moveaxis(im, 0, 2))
        m = masks.argmax(dim=1)[i, ...]
        m = np.ma.masked_where(m == 5, m)
        ax[i, 1].imshow(m, cmap=cm_mask)
        ax[i, 2].imshow(hvs[i, 0, ...], cmap="coolwarm")
        ax[i, 3].imshow(hvs[i, 1, ...], cmap="coolwarm")

    for a in ax.ravel():
        a.axis("off")
    for c, v in enumerate(cols):
        ax[0, c].set_title(v)

    plt.tight_layout()
    plt.show()
    plt.savefig("_preprocess.png")
