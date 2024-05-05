import cv2
import numpy as np
import openslide
import shutil
from pathlib import Path
from PIL import Image
from scipy.ndimage import label
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
from lovely_numpy import lo
from matplotlib.image import imread
from patchify import patchify

from hover_quant.datasets.data_utils import (
    compute_hv_map,
    pad_array_to_patch_size,
    process_xml_annotations,
    rm_n_mkdir,
    download_data_url,
)
import hover_quant.config as config


class MoNuSACDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transforms=None,
        nucleus_type_labels=True,
        hovernet_preprocess=True,
    ):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.nucleus_type_labels = nucleus_type_labels
        self.hovernet_preprocess = hovernet_preprocess

        # dirs for images, masks
        self.imgs_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "labels"

        # stop if the images and masks directories don't already exist
        assert (
            self.imgs_dir.is_dir()
        ), f"Error: 'images' directory not found: {self.imgs_dir}"
        assert (
            self.masks_dir.is_dir()
        ), f"Error: 'masks' directory not found: {self.masks_dir}"

        self.paths = list(self.imgs_dir.glob("*.png"))
        self.fnames = [p.stem for p in self.paths]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = self.imgs_dir / f"{fname}.png"
        mask_path = self.masks_dir / f"{fname}.npy"

        img_np = cv2.cvtColor(
            cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        mask_np = np.load(str(mask_path))

        if self.nucleus_type_labels is False:
            # only look at "background" mask in last channel
            mask_np = mask_np[0, :, :]
            # invert so that ones are nuclei pixels
            mask_np = 1 - mask_np
        else:
            # Move background to the last channel
            mask_np = np.roll(mask_np, shift=-1, axis=0)

        if self.transforms is not None:
            transformed = self.transforms(image=img_np, mask=mask_np)
            img_np = transformed["image"]
            mask_np = transformed["mask"]

        # swap channel dim to pytorch standard (C, H, W)
        img_np = img_np.transpose((2, 0, 1))

        # compute hv map
        if self.hovernet_preprocess:
            if self.nucleus_type_labels:
                # sum across mask channels to squash mask channel dim to size 1
                # don't sum the last channel, which is background!
                mask_1c = np.sum(mask_np[:-1, :, :], axis=0)
            else:
                mask_1c = mask_np
            hv_map = compute_hv_map(mask_1c)

        if self.hovernet_preprocess:
            out = (
                torch.from_numpy(img_np),
                torch.from_numpy(mask_np),
                torch.from_numpy(hv_map),
                "Not specified",
            )
        else:
            out = (torch.from_numpy(img_np), torch.from_numpy(mask_np), "Not specified")

        return out


class MoNuSACDataModule:
    def __init__(
        self,
        data_dir,
        download=False,
        train_val_size=0.85,
        nucleus_type_labels=True,
        hovernet_preprocess=True,
        include_ambiguous=False,
        transforms=None,
        shuffle=(True, False, False),
    ):
        np.random.seed(42)
        self.data_dir = Path(data_dir)
        self.modes = {
            "train": {
                "url": "https://drive.google.com/uc?id=1lxMZaAPSpEHLSxGA9KKMt_r-4S8dwLhq",
                "types_mapping": {
                    "Background": 0,
                    "Epithelial": 1,
                    "Lymphocyte": 2,
                    "Macrophage": 3,
                    "Neutrophil": 4,
                },
            },
            "test": {
                "url": "https://drive.google.com/uc?id=1G54vsOdxWY1hG7dzmkeK3r0xz9s-heyQ",
                "types_mapping": {
                    "Background": 0,
                    "Epithelial": 1,
                    "Lymphocyte": 2,
                    "Macrophage": 3,
                    "Neutrophil": 4,
                    "Ambiguous": 5,
                },
            },
        }
        if not include_ambiguous:
            for mode in self.modes.keys():
                self.modes[mode]["types_mapping"].pop("Ambiguous", None)
        self.train_val_size = train_val_size
        self.nucleus_type_labels = nucleus_type_labels
        self.hovernet_preprocess = hovernet_preprocess
        self.transforms = transforms
        self.shuffle = shuffle

        if download:
            self.setup()

    def setup(self, patch_size=256, verbose=False):
        for mode in self.modes.keys():
            print(f"Downloading {mode} data")
            download_data_url(
                self.modes[mode]["url"], Path(self.data_dir / "raw" / f"{mode}")
            )
            print(f"Processing {mode} data")
            self._process_monusac(
                mode,
                Path(self.data_dir / "raw"),
                Path(self.data_dir / "processed"),
                patch_size=patch_size,
                verbose=verbose,
            )
        return None
        # for mode in self.modes.keys():
        #         assert (
        #             Path(self.data_dir / "processed" / f"{mode}").is_dir()
        #         ), f"data folder is not found: {Path(self.data_dir, f'{mode}')}"

    def _process_monusac(
        self, mode, raw_folder, out_folder, patch_size=256, verbose=False
    ):
        types_mapping = self.modes[mode]["types_mapping"]
        patients_full_path = Path(raw_folder, mode).glob(
            "*"
        )  # [str(x) for x in Path(raw_folder).glob('*')]
        rm_n_mkdir(Path(out_folder, mode, "images"))
        rm_n_mkdir(Path(out_folder, mode, "labels"))
        for patient_path in tqdm(list(patients_full_path)):
            patient_name = Path(patient_path).stem
            sub_images = Path(patient_path).glob(
                "*.svs"
            )  # [str(x) for x in Path(patient_path).glob('*.svs')]
            for sub_image_path in sub_images:
                sub_image_name = Path(sub_image_path).stem
                img = openslide.OpenSlide(sub_image_path)
                np_img = np.array(
                    img.read_region((0, 0), 0, img.level_dimensions[0]).convert("RGB")
                )

                ### Process masks by available types from .xmls
                xml_file_path = Path(
                    Path(sub_image_path).parent, Path(sub_image_path).stem + ".xml"
                )
                inst_type_dict = process_xml_annotations(xml_file_path, img)
                for cell_type in types_mapping.keys():
                    if cell_type not in inst_type_dict.keys():
                        inst_type_dict[cell_type] = np.zeros(np.shape(np_img)[0:2])

                ### Generate instance map and type map
                inst_map = np.zeros(np.shape(np_img)[0:2])
                type_map = np.zeros(np.shape(np_img)[0:2])
                for k, v in types_mapping.items():
                    uniques = np.unique(inst_type_dict[k])[1:]  # exclude 0
                    for val in uniques:
                        inst_map[inst_type_dict[k] == val] = val
                        type_map[inst_type_dict[k] == val] = v

                ### Remap labels to be consecutive
                for i, inst in enumerate(
                    np.unique(inst_map)
                ):  ##### TODO check if 0 avoidance is needed
                    inst_map[inst_map == inst] = i

                ### Compute list of instance types
                inst_type = [
                    type_map[
                        np.where(inst_map == x)[0][0], np.where(inst_map == x)[1][0]
                    ]
                    for x in list(np.unique(inst_map))[1:]
                ]

                ### Compute instace centroids
                inst_centroid_list = []
                inst_id_list = list(np.unique(inst_map))
                for inst_id in inst_id_list[1:]:  # avoid 0 i.e background
                    mask = np.array(inst_map == inst_id, np.uint8)
                    inst_moment = cv2.moments(mask)
                    inst_centroid = [
                        (inst_moment["m10"] / inst_moment["m00"]),
                        (inst_moment["m01"] / inst_moment["m00"]),
                    ]
                    inst_centroid_list.append(inst_centroid)

                np_dict = {
                    "inst_map": inst_map.astype(np.int16),
                    "type_map": type_map.astype(np.int8),
                    "inst_type": (np.array([inst_type]).T).astype(np.int8),
                    "inst_centroid": np.array(inst_centroid_list),
                }

                ### Save images and labels
                # if save_old_format:
                #     cv2.imwrite(str(Path(out_folder, 'old_format', mode, 'images', f"{sub_image_name}.png")), np_img[:, :, ::-1])
                #     np.save(Path(out_folder, 'old_format', mode, 'labels', f"{sub_image_name}.npy"), np_dict)

                np_masks = np.zeros(
                    (
                        len(types_mapping),
                        np_dict["type_map"].shape[0],
                        np_dict["type_map"].shape[1],
                    )
                )
                for k, inst_labels in types_mapping.items():
                    inst_mask = (np_dict["type_map"] == inst_labels).astype(int)
                    if k != "Background":
                        inst_mask = inst_mask * np_dict["inst_map"]
                    np_masks[inst_labels] = inst_mask
                np_masks = np_masks.transpose(1, 2, 0)

                # np_img
                # np_masks

                if (
                    np_img.shape[0] % patch_size != 0
                    or np_img.shape[1] % patch_size != 0
                ):
                    np_img = pad_array_to_patch_size(np_img, patch_size)
                    np_masks = pad_array_to_patch_size(np_masks, patch_size)

                assert (
                    np_img.shape[0] % patch_size == 0
                    and np_img.shape[1] % patch_size == 0
                ), f"Image shape: {np_img.shape}, patch_size: {patch_size}"

                if np_img.shape[0] > patch_size or np_img.shape[1] > patch_size:
                    # Patchify the image and the label, using padding and mirroring
                    np_img_patches = patchify(
                        np_img, (patch_size, patch_size, 3), step=patch_size
                    ).reshape(-1, patch_size, patch_size, 3)
                    np_masks_patches = patchify(
                        np_masks,
                        (patch_size, patch_size, len(types_mapping)),
                        step=patch_size,
                    ).reshape(-1, patch_size, patch_size, len(types_mapping))

                    assert (
                        np_img_patches.shape[0] == np_masks_patches.shape[0]
                    ), f"Image patches: {np_img_patches.shape}, Mask patches: {np_masks_patches.shape}, Image shape: {np_img.shape}, Mask shape: {np_masks.shape}"

                    for patch_idx in range(np_img_patches.shape[0]):
                        relabeled_map = []
                        for map_idx in range(np_masks_patches.shape[-1]):
                            if map_idx == 0:  # skip relabeling for background
                                relabeled_map.append(
                                    np_masks_patches[patch_idx, :, :, map_idx]
                                )
                                continue
                            relabeled_map.append(
                                label(np_masks_patches[patch_idx, :, :, map_idx])[0]
                            )
                        np_masks_patches[patch_idx, :, :, :] = np.array(
                            relabeled_map
                        ).transpose(1, 2, 0)

                else:
                    np_img_patches = np.expand_dims(np_img, axis=0)
                    np_masks_patches = np.expand_dims(np_masks, axis=0)

                for i in range(np_img_patches.shape[0]):
                    sub_image_name_pn = f"{sub_image_name}_{i}"
                    if all(
                        len(np.unique(np_masks_patches[i, :, :, j])) == 1
                        for j in range(np_masks_patches[i, :, :, :].shape[-1])
                    ):
                        if verbose:
                            print(f"Warn: {sub_image_name_pn} is empty")
                        continue
                    cv2.imwrite(
                        str(
                            Path(out_folder, mode, "images", f"{sub_image_name_pn}.png")
                        ),
                        np_img_patches[i, :, :, ::-1],
                    )
                    np.save(
                        Path(out_folder, mode, "labels", f"{sub_image_name_pn}.npy"),
                        np_masks_patches[i, :, :, :].transpose(2, 0, 1),
                    )

        if mode == "train" and self.train_val_size is not None:
            rm_n_mkdir(Path(out_folder, "val", "images"))
            rm_n_mkdir(Path(out_folder, "val", "labels"))
            imgs = list(Path(out_folder, mode, "images").glob("*.png"))
            np.random.shuffle(imgs)
            split = int(len(imgs) * self.train_val_size)
            val_imgs = imgs[split:]
            for img in val_imgs:
                shutil.move(img, Path(out_folder, "val", "images", img.name))
                shutil.move(
                    Path(out_folder, mode, "labels", f"{img.stem}.npy"),
                    Path(out_folder, "val", "labels", f"{img.stem}.npy"),
                )

    def _get_dataset(self, data_folder, transforms):
        return MoNuSACDataset(
            data_dir=data_folder,
            transforms=transforms,
            nucleus_type_labels=self.nucleus_type_labels,
            hovernet_preprocess=self.hovernet_preprocess,
        )

    @property
    def train_dataloader(self):
        """
        Dataloader for training set.
        Yields (image, mask), or (image, mask, hv) for HoVer-Net
        """
        return DataLoader(
            dataset=self._get_dataset(
                data_folder=Path(self.data_dir / "processed" / "train"),
                transforms=self.transforms[0],
            ),
            batch_size=config.BATCH_SIZE,
            shuffle=self.shuffle[0],
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    @property
    def valid_dataloader(self):
        """
        Dataloader for val set.
        Yields (image, mask), or (image, mask, hv) for HoVer-Net
        """
        return DataLoader(
            dataset=self._get_dataset(
                data_folder=Path(self.data_dir / "processed" / "val"),
                transforms=self.transforms[1],
            ),
            batch_size=config.BATCH_SIZE,
            shuffle=self.shuffle[1],
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    @property
    def test_dataloader(self):
        """
        Dataloader for test set.
        Yields (image, mask), or (image, mask, hv) for HoVer-Net
        """
        return DataLoader(
            dataset=self._get_dataset(
                data_folder=Path(self.data_dir / "processed" / "test"),
                transforms=self.transforms[2],
            ),
            batch_size=config.BATCH_SIZE,
            shuffle=self.shuffle[2],
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
