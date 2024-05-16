import os
from glob import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import pyvips
import cv2
import polars as pl
from PIL import Image
import matplotlib.pyplot as plt
from sympy import Point, Ray, Line, Segment, deg, sqrt, cos, sin, N
import patch_class.config as config
from patch_class.model import ImgAugmentor
import lovely_tensors as lt

lt.monkey_patch()


class InferDF_PPTS_PatchDataset(Dataset):
    def __init__(self, ppts_pd, vips_slide, tf, extract_patch_height):
        self.ppts_info = ppts_pd
        self.vips_slide = vips_slide
        self.extract_patch_height = extract_patch_height
        self.transforms = tf

    def __len__(self):
        return len(self.ppts_info)

    def __getitem__(self, idx):
        img_info = self.ppts_info.row(idx)
        index = img_info[0]  # index of 'Pair_ID'
        pts = eval(str(img_info[1]))  # index of 'Points_pair'

        img = self.transforms["from_np.uint8_to_torch.float"](
            self.extract_patch(self.vips_slide, pts, self.extract_patch_height).numpy()
        )
        img = self.transforms["resize_for_infer"](img)

        # torch.Tensor -> (3, 512, 512)
        return img, index

    def rotate_point(self, orig_shape, rot_shape, point, angle_rad):
        org_center = np.array(orig_shape) / 2.0
        rot_center = np.array(rot_shape) / 2.0
        org = point - org_center
        new_coord = np.array(
            [
                (org[0] * cos(angle_rad)) + (org[1] * sin(angle_rad)),
                (-org[0] * sin(angle_rad)) + (org[1] * cos(angle_rad)),
            ]
        )
        rotated_point = new_coord + rot_center
        return rotated_point

    def find_rectangular_points(self, p0, p1, height):
        half_height = height / 2
        x_axis = Ray(Point(0, 0), Point(1, 0))
        ray = Ray(p0, p1)
        angle = ray.closing_angle(x_axis)

        if p0.x != p1.x:
            ray_rot = ray.rotate(-angle)
            upper_ray = ray_rot.translate(0, -half_height)
            lower_ray = ray_rot.translate(0, half_height)
            upper_line = Line(upper_ray.rotate(angle))
            lower_line = Line(lower_ray.rotate(angle))
        else:
            upper_line = Line(ray).translate(-half_height, 0)
            lower_line = Line(ray).translate(half_height, 0)

        upper_found_p0 = upper_line.projection(p0)
        lower_found_p0 = lower_line.projection(p0)
        upper_found_p1 = upper_line.projection(p1)
        lower_found_p1 = lower_line.projection(p1)

        return (upper_found_p0, upper_found_p1, lower_found_p1, lower_found_p0)

    def find_bounding_rect(self, points, local_offset=None):
        xs = list(zip(*points))[0]
        ys = list(zip(*points))[1]
        top_left = Point(min(xs), min(ys))
        top_right = Point(max(xs), min(ys))
        bottom_right = Point(max(xs), max(ys))
        bottom_left = Point(min(xs), max(ys))
        if local_offset is not None:
            tl = Point(top_left.x - local_offset, top_left.y - local_offset)
            tr = Point(top_right.x + local_offset, top_right.y - local_offset)
            br = Point(bottom_right.x + local_offset, bottom_right.y + local_offset)
            bl = Point(bottom_left.x - local_offset, bottom_left.y + local_offset)
            return (tl, tr, br, bl)
        else:
            return (top_left, top_right, bottom_right, bottom_left)

    def extract_patch(self, vips_img, pts, patch_height):
        p0, p1 = (Point(pts[0]), Point(pts[1]))
        angle = Ray((0, 0), (1, 0)).closing_angle(Ray(p0, p1))  # in radians

        # Find points of rectangle, it may be rotated
        rect_points = self.find_rectangular_points(p0, p1, patch_height)

        # Find bounding box area of the rectangular
        bbox_points = self.find_bounding_rect(rect_points, local_offset=0)

        # Crop area using bbox_points
        width = bbox_points[2].x - bbox_points[0].x
        height = bbox_points[2].y - bbox_points[0].y

        ### Extend image if bbox is out of image
        if (
            int(bbox_points[0].x) + int(width) > vips_img.width
            or int(bbox_points[0].y) + int(height) > vips_img.height
            or int(bbox_points[0].x) < 0
            or int(bbox_points[0].y) < 0
        ):
            dlt = self.calculate_extend_polygon(
                (bbox_points[0], bbox_points[1], bbox_points[2], bbox_points[3]),
                (vips_img.width, vips_img.height),
                eps=patch_height // 2,
            )
            ### Set global dlt offset
            vips_img = vips_img.embed(
                dlt,
                dlt,
                vips_img.width + (2 * dlt),
                vips_img.height + (2 * dlt),
                background=[242.0, 242.0, 242.0],
            )  # for sRGB slide background=[242.0, 242.0, 242.0, 255.0]
            p0 = Point(p0.x + dlt, p0.y + dlt)
            p1 = Point(p1.x + dlt, p1.y + dlt)
            rect_points = self.find_rectangular_points(p0, p1, patch_height)
            bbox_points = self.find_bounding_rect(rect_points, local_offset=0)

        rect_area_crop = vips_img.crop(
            int(bbox_points[0].x),  # int(N(bbox_points[0].x, 2)),
            int(bbox_points[0].y),  # int(N(bbox_points[0].y, 2)),
            int(width),  # int(N(width, 2)),
            int(height),  # int(N(height, 2))
        )

        # Recalculate points of rectangle on the cropped area
        img_crop_center = Segment(rect_points[0], rect_points[2]).midpoint
        crop_rect_center = Segment(
            Point(0, 0), Point(rect_area_crop.width, rect_area_crop.height)
        ).midpoint
        delta = (
            img_crop_center.x - crop_rect_center.x,
            img_crop_center.y - crop_rect_center.y,
        )
        rect_points_on_crop = [
            Point(p.x - delta[0], p.y - delta[1]) for p in rect_points
        ]

        # Rotate cropped area
        rotated_rect_area_crop = rect_area_crop.rotate(deg(angle))

        # Rotate points of rectangle on the cropped area
        rotated_rec_points = tuple(
            [
                (
                    self.rotate_point(
                        (rect_area_crop.width, rect_area_crop.height),
                        (rotated_rect_area_crop.width, rotated_rect_area_crop.height),
                        p,
                        -angle,
                    ).astype("double")
                )
                for p in rect_points_on_crop
            ]
        )

        # Crop area using rotated points
        rotated_bbox_points = self.find_bounding_rect(
            rotated_rec_points, local_offset=0
        )
        width = rotated_bbox_points[2].x - rotated_bbox_points[0].x
        height = rotated_bbox_points[2].y - rotated_bbox_points[0].y

        # assert round(width) == round(height), f"Width and height must be equal, width: {width}, height: {height}"

        return rotated_rect_area_crop.crop(
            int(N(rotated_bbox_points[0].x, 2)),
            int(N(rotated_bbox_points[0].y, 2)),
            round(width),  # int(N(width, 2)),
            round(height),  # int(N(height, 2))
        )

    def calculate_extend_polygon(self, rec_points, slide_size, eps=0):
        """
        rec_points: tuple(Point2D(x,y)); tl, tr, br, bl
        slide_size: (slide.width, slide.height)
        eps: int
        """
        tl = rec_points[0]
        br = rec_points[2]
        min_cord = 0
        max_cord = 0

        if tl.x < 0 or tl.y < 0:
            min_cord = abs(min(tl.x, tl.y))

        if br.x > slide_size[0] or br.y > slide_size[1]:
            max_cord = max(br.x - slide_size[0], br.y - slide_size[1])

        return int(sqrt(2 * pow(max(min_cord, max_cord), 2)) + eps)


class PatchDataset(Dataset):
    def __init__(
        self,
        data_path,
        tf,
        selected_folds,
        kfold=3,
        clip_to_min=False,
        kfold_seed=42,
        clip_seed=42,
    ):
        self.data_path = data_path
        try:
            self.dataframe_path = list(Path(data_path).glob("*.csv"))[0]
        except IndexError:
            raise FileNotFoundError(f"No csv file found in {data_path}")

        if not self.dataframe_path.exists():
            raise FileNotFoundError(f"Dataframe not found at {self.dataframe_path}")

        # assert kfold in [10, 5, 4, 3, 2], "kfold must be in in [10, 5, 3, 2]"
        assert kfold >= 1, "kfold must be equal or greater than 1"
        dataframe_all_fold = self._split_fold(
            pl.read_csv(self.dataframe_path), kfold, seed=kfold_seed
        )

        if isinstance(selected_folds, int):
            selected_folds = [selected_folds]
        # check if each element is in range [1, kfold]
        assert all(
            [0 < x <= kfold for x in selected_folds]
        ), "selected_fold must be in range [1, kfold]"
        self.dataframe = dataframe_all_fold.filter(pl.col("fold").is_in(selected_folds))

        classes = sorted([x for x in self.dataframe["label"].unique().to_list()])
        assert (
            len(classes) == config.NUM_CLASSES
        ), f"Number of classes in dataset ({len(classes)}) does not match config ({config.NUM_CLASSES})"

        self.class_to_idx = {
            class_label: idx for idx, class_label in enumerate(classes)
        }  # aka data_classes, {'necrosis': 0, 'normal_lung': 1, 'stroma_tls': 2}
        self.idx_to_class = {
            idx: class_label for idx, class_label in enumerate(classes)
        }

        ### self.dataframe: | slide_id | label | img_name | split | fold |
        self.imgs = sorted(
            [
                str(Path(self.data_path, row[1], row[2]))
                for row in self.dataframe.iter_rows()
            ]
        )

        self.targets = [self.class_to_idx[Path(img).parent.name] for img in self.imgs]
        self.imgs_per_class = {}
        for class_label in classes:
            self.imgs_per_class[class_label] = self.dataframe.filter(
                pl.col("label") == class_label
            ).shape[0]

        self.transforms = tf

        if clip_to_min:
            assert kfold == 1, "clip_to_min can only be used with kfold = 1"
            min_imgs_for_classes = min(self.imgs_per_class.values())
            self.imgs = []
            for class_label in classes:
                class_imgs = self.dataframe.filter(pl.col("label") == class_label)
                class_imgs = class_imgs.sample(
                    fraction=min_imgs_for_classes / class_imgs.shape[0], seed=clip_seed
                )
                class_imgs = [
                    str(Path(self.data_path, row[1], row[2]))
                    for row in class_imgs.iter_rows()
                ]
                self.imgs.extend(class_imgs)
            self.targets = [
                self.class_to_idx[Path(img).parent.name] for img in self.imgs
            ]
            self.imgs_per_class = {
                class_label: min_imgs_for_classes for class_label in classes
            }

    def _split_fold(self, dataframe_df, kfold, seed=42, kfold2_split_ratio=0.5):

        def assign_folds(group_df):
            shuffled_group_df = group_df.sample(fraction=1.0, seed=seed)
            fold_numbers = (pl.arange(0, shuffled_group_df.height) % kfold) + 1
            return shuffled_group_df.with_columns(fold_numbers.alias("fold"))

        out_dataframe = dataframe_df.group_by("label").map_groups(assign_folds)
        return out_dataframe.sort(["slide_id", "label"])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.resize(
            cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB),
            (config.INPUT_SIZE[0], config.INPUT_SIZE[1]),
        )
        img = self.transforms["from_np.uint8_to_torch.float"](img)
        label = torch.tensor(
            self.class_to_idx[Path(self.imgs[idx]).parent.name], dtype=torch.long
        )
        return img, label


class AugmentedDataLoader:
    def __init__(self, dataloader, augmentor):
        self.dataloader = dataloader
        self.augmentor = augmentor
        self.dataset = dataloader.dataset

    def __iter__(self):
        for batch_idx, (imgs, labels) in enumerate(self.dataloader):
            yield self.augmentor(imgs), labels

    def __len__(self):
        return len(self.dataloader)


if __name__ == "__main__":
    n_show = 4

    # Data info
    train_val_dataset = PatchDataset(
        config.TRAIN_DATA_DIR,
        config.ATF,
        selected_folds=(1),
        kfold=1,
    )

    print("=====================================")
    print(f"Train dataset info: {config.TRAIN_DATA_DIR}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Number of images: {len(train_val_dataset)}")
    print(f"Labels: {train_val_dataset.class_to_idx}")
    print(f"Number of images per class: {train_val_dataset.imgs_per_class}")
    print(f"Using normalization: {config.NORM_ARGS}")
    print("=====================================")

    ### Test dataset
    test_dataset = PatchDataset(
        config.TEST_DATA_DIR, config.ATF, selected_folds=(1), kfold=1, clip_to_min=True
    )

    print("=====================================")
    print(f"Test dataset info: {config.TEST_DATA_DIR}")
    print(f"Test dataset length: {len(test_dataset)}")
    print(f"Number of images per class: {test_dataset.imgs_per_class}")
    print(
        f"Targets for test_dataset: {dict(zip(*np.unique(torch.tensor(test_dataset.targets).numpy(), return_counts=True)))}"
    )
    print("=====================================")

    ### Augmentors
    train_augmentor = ImgAugmentor(
        config.ATF,
        p_augment=False,
        preproc=False,
        norm_args=config.NORM_ARGS,
        # color_augment_args=config.COLOR_AUG_ARGS,
        use_fast_color_aug=False,
        # clamp_values=True,
        train_mode=True,
        proc_device="cpu",
        target_device="cpu",
    )

    val_test_augmentor = ImgAugmentor(
        config.ATF,
        p_augment=False,
        preproc=False,
        norm_args=config.NORM_ARGS,
        color_augment_args=None,
        use_fast_color_aug=False,
        clamp_values=True,
        train_mode=False,
        proc_device="cpu",
        target_device="cpu",
    )

    train_dataloader = AugmentedDataLoader(
        DataLoader(
            train_val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        ),
        train_augmentor,
    )

    test_dataloader = AugmentedDataLoader(
        DataLoader(
            test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False
        ),
        val_test_augmentor,
    )

    for batch_idx, (imgs, labels) in enumerate(train_dataloader):
        print(f"Train imgs tensor: {imgs}")
        print(f"Train img labels: {labels[:n_show]}")
        x = imgs[:n_show] if n_show < config.BATCH_SIZE else imgs[: config.BATCH_SIZE]
        grid = torchvision.utils.make_grid(x.view(-1, *imgs.shape[1:]))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("assets/train_dataloader_pbatch.png")
        plt.close()
        break

    for batch_idx, (imgs, labels) in enumerate(test_dataloader):
        print(f"Test imgs tensor: {imgs}")
        print(f"Test imgs labels: {labels[:n_show]}")
        x = imgs[:n_show] if n_show < config.BATCH_SIZE else imgs[: config.BATCH_SIZE]
        grid = torchvision.utils.make_grid(x.view(-1, *imgs.shape[1:]))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("assets/test_dataloader_pbatch.png")
        plt.close()
        break
