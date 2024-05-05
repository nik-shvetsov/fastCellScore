import os
import configparser
from glob import glob
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid
import pyvips
import numpy as np
import cv2
import matplotlib.pyplot as plt
import lovely_tensors as lt
from roi_segment.utils import print_gpu_utilization
import roi_segment.config as config
import roi_segment.augs as augs

lt.monkey_patch()


class DsWSIDataset(Dataset):
    """
    Dataset is using directory with .svs wsis and respective .itn files with annotations.
    WSI are rescaled to specified size. Masks are binary and returned as bool numpy array.
    """

    def __init__(
        self,
        wsis_path,
        itns_path,
        tf,
        scale=1,
        size=(1024, 1024),
        preproc=True,
        augment=True,
        for_dataloader=True,
    ):
        self.wsis_path = wsis_path
        self.itns_path = itns_path

        slides_glob = sorted(glob(f"{Path(wsis_path)}/*/*.svs"))
        slides_names = [os.path.basename(aps) for aps in slides_glob]
        itn_glob = sorted(
            [os.path.join(itns_path, ap.replace(".svs", ".itn")) for ap in slides_names]
        )
        self.paths_x_y = {
            i: (slides_glob[i], itn_glob[i]) for i in range(len(slides_glob))
        }

        self.transforms = tf

        self.scale = scale
        self.size = size if isinstance(size, tuple) else (size, size)

        self.augment = augment
        self.preproc = preproc

        self.for_dataloader = for_dataloader

    def __len__(self):
        return len(self.paths_x_y)

    @staticmethod
    def _get_polygons(config_polygon):
        """
        Return: polygon information, like
        {
            0:[(x0,y0), (x1,y1),...],
            1:[(x0,y0), (x1,y1),...],
            ...
        }
        """
        poly_points = defaultdict(list)
        l_it = list(((el.split("_")[2:4]) for el in (list(config_polygon))))
        for i in range(0, len(l_it), 2):
            poly, num = l_it[i]
            poly_points[int(poly)].append(
                (
                    float(config_polygon[f"poly_x_{poly}_{num}"]),
                    float(config_polygon[f"poly_y_{poly}_{num}"]),
                )
            )
        return poly_points

    @staticmethod
    def _scale_coords(reference_size, target_size, reference_coords):
        """
        Non-opencv version of _scale_coords()
        Ref: https://github.com/histolab/histolab/blob/master/histolab/util.py
        """
        reference_size = reference_size[0:2]
        target_size = target_size[0:2]
        result = {}
        for k in reference_coords.keys():
            result[k] = []
            for item in reference_coords[k]:
                updated_item = np.floor(
                    (np.asarray(item).ravel() * target_size) / reference_size
                ).astype("int64")
                result[k].append(updated_item)
        return result

    def __getitem__(self, idx):
        ref_vips_slide = pyvips.Image.new_from_file(self.paths_x_y[idx][0], level=0)
        # scaled_slide = pyvips.Image.thumbnail(self.paths_x_y[idx][0], max(self.size[0], self.size[1])) #  no more than max(self.size[0], self.size[1])
        scaled_slide = pyvips.Image.new_from_file(
            self.paths_x_y[idx][0],
            level=int(ref_vips_slide.get("openslide.level-count")) - 1,
        )

        vips_slide = scaled_slide.extract_band(0, n=3)

        config_parser = configparser.ConfigParser()
        config_parser.read(self.paths_x_y[idx][1])
        polygons_points = self._get_polygons(config_parser["Polygon"])

        scaled_polygons_points = self._scale_coords(
            (ref_vips_slide.width, ref_vips_slide.height),
            (vips_slide.width, vips_slide.height),
            polygons_points,
        )

        mask = np.zeros((vips_slide.height, vips_slide.width), dtype=np.uint8)
        for polygon_points in scaled_polygons_points.values():
            npcoords = np.array([polygon_points], np.int32)
            cv2.fillPoly(mask, npcoords, 1)

        ref_img = vips_slide.numpy()
        ref_mask = mask.astype(bool) * 255.0

        preproc_img = {"image": ref_img, "mask": ref_mask}

        if self.augment:
            preproc_img = self.transforms["aug"](
                image=preproc_img["image"], mask=preproc_img["mask"]
            )
        if self.preproc:
            preproc_img["image"] = self.transforms["preproc"](preproc_img["image"])

        preproc_img = self.transforms[f"resize_to_tensor"](
            image=preproc_img["image"], mask=preproc_img["mask"]
        )
        ###################

        img = preproc_img["image"]
        mask = preproc_img["mask"]

        # Convert pixel values to range [0, 1]: preproc_img['image'] /= 255.0
        # Not needed here due to self.preproc

        if self.for_dataloader:
            return img, torch.unsqueeze(mask, dim=0).bool()
        else:
            return (
                img,
                torch.unsqueeze(torch.Tensor(mask), dim=0).long(),
                ref_img,
                ref_mask,
            )


if __name__ == "__main__":
    n_show = 4
    batch_size = 8  # config.BATCH_SIZE

    dataset = DsWSIDataset(
        config.WSI_DIR,
        config.ITN_DIR,
        augs.ATF,
        preproc=False,
        augment=True,
        for_dataloader=True,
    )
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        dataset, [0.8, 0.1, 0.1], generator=generator
    )

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.ACCELERATOR == "cuda" else False,
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.ACCELERATOR == "cuda" else False,
    )
    test_dataloader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.ACCELERATOR == "cuda" else False,
    )

    print_gpu_utilization(device_idx=0)

    for batch_idx, (imgs, masks) in enumerate(test_dataloader):
        print(f"imgs tensor: {imgs}")
        print(f"masks tensor: {masks}")
        x = (
            imgs[:n_show] if n_show < batch_size else imgs[:batch_size]
        )  # config.BATCH_SIZE
        grid = make_grid(
            x.view(-1, config.IN_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
        )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.savefig("assets/dataloader_pbatch.png")
        break
