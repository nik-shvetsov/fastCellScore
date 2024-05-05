import os
from glob import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import IterableDataset
import torchvision
import numpy as np
import pyvips
import cv2
import polars as pl
from PIL import Image
import matplotlib.pyplot as plt
from sympy import (
    Point,
    Point2D,
    Ray,
    Line,
    Segment,
    Polygon,
    deg,
    sqrt,
    cos,
    sin,
    N,
    solve,
)
import lovely_tensors as lt
from patchify import patchify, unpatchify

lt.monkey_patch()


class PatchGeneratorDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


def rotate_point(orig_shape, rot_shape, point, angle_rad):
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


def find_rectangular_points(p0, p1, height):
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


def find_bounding_rect(points, local_offset=None):
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


def _old_extract_patch(vips_img, pts, patch_height):
    p0, p1 = (Point(pts[0]), Point(pts[1]))
    angle = Ray((0, 0), (1, 0)).closing_angle(Ray(p0, p1))  # in radians
    rect_points = find_rectangular_points(p0, p1, patch_height)
    bbox_points = find_bounding_rect(rect_points, local_offset=0)
    width = bbox_points[2].x - bbox_points[0].x
    height = bbox_points[2].y - bbox_points[0].y
    rect_area_crop = vips_img.crop(
        int(bbox_points[0].x),
        int(bbox_points[0].y),
        int(width),
        int(height),
        # int(N(bbox_points[0].x, 2)),
        # int(N(bbox_points[0].y, 2)),
        # int(N(width, 2)),
        # int(N(height, 2)),
    )
    img_crop_center = Segment(rect_points[0], rect_points[2]).midpoint
    crop_rect_center = Segment(
        Point(0, 0), Point(rect_area_crop.width, rect_area_crop.height)
    ).midpoint
    delta = (
        img_crop_center.x - crop_rect_center.x,
        img_crop_center.y - crop_rect_center.y,
    )
    rect_points_on_crop = [Point(p.x - delta[0], p.y - delta[1]) for p in rect_points]
    rotated_rect_area_crop = rect_area_crop.rotate(deg(angle))
    rotated_rec_points = tuple(
        [
            (
                rotate_point(
                    (rect_area_crop.width, rect_area_crop.height),
                    (rotated_rect_area_crop.width, rotated_rect_area_crop.height),
                    p,
                    -angle,
                ).astype("double")
            )
            for p in rect_points_on_crop
        ]
    )
    rotated_bbox_points = find_bounding_rect(rotated_rec_points, local_offset=0)
    width = rotated_bbox_points[2].x - rotated_bbox_points[0].x
    height = rotated_bbox_points[2].y - rotated_bbox_points[0].y
    return rotated_rect_area_crop.crop(
        int(rotated_bbox_points[0].x),
        int(rotated_bbox_points[0].y),
        int(width),
        int(height),
        # int(N(rotated_bbox_points[0].x, 2)),
        # int(N(rotated_bbox_points[0].y, 2)),
        # int(N(width, 2)),
        # int(N(height, 2)),
    )


def extract_patch(vips_img, pts, patch_height):
    p0, p1 = (Point(pts[0]), Point(pts[1]))
    angle = Ray((0, 0), (1, 0)).closing_angle(Ray(p0, p1))  # in radians

    # Find points of rectangle, it may be rotated
    rect_points = find_rectangular_points(p0, p1, patch_height)

    # Find bounding box area of the rectangular
    bbox_points = find_bounding_rect(rect_points, local_offset=0)

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
        # print ("Extending slide...")
        # if int(bbox_points[0].x) < 0 or int(bbox_points[0].y) < 0:
        #     dlt = 2 * abs(min(int(bbox_points[0].x), int(bbox_points[0].y)))
        # else:
        #     dlt = 2 * (max(
        #         int(bbox_points[0].x) + int(width) - vips_img.width,
        #         int(bbox_points[0].y) + int(height) - vips_img.height
        #     ))
        # print (f"{dlt}")

        dlt = calculate_extend_polygon(
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
        rect_points = find_rectangular_points(p0, p1, patch_height)
        bbox_points = find_bounding_rect(rect_points, local_offset=0)

    # print ("---")
    # tmpx = int(bbox_points[0].x) + int(width)
    # tmpy = int(bbox_points[0].y) + int(height)
    # print (f"{vips_img.width} x {vips_img.height}")

    # print (f"bbox_points[0].x + width = {int(bbox_points[0].x) + int(width)}")
    # print (f"bbox_points[0].y + height = {int(bbox_points[0].y) + int(height)}")

    # print (
    #     int(bbox_points[0].x),
    #     int(bbox_points[0].y),
    #     int(width),
    #     int(height)
    # )
    # print ("---")

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
    rect_points_on_crop = [Point(p.x - delta[0], p.y - delta[1]) for p in rect_points]

    # Rotate cropped area
    rotated_rect_area_crop = rect_area_crop.rotate(deg(angle))

    # Rotate points of rectangle on the cropped area
    rotated_rec_points = tuple(
        [
            (
                rotate_point(
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
    rotated_bbox_points = find_bounding_rect(rotated_rec_points, local_offset=0)
    width = rotated_bbox_points[2].x - rotated_bbox_points[0].x
    height = rotated_bbox_points[2].y - rotated_bbox_points[0].y

    return rotated_rect_area_crop.crop(
        int(N(rotated_bbox_points[0].x, 2)),
        int(N(rotated_bbox_points[0].y, 2)),
        int(width),  # int(N(width, 2)),
        int(height),  # int(N(height, 2))
    )


def calculate_extend_polygon(rec_points, slide_size, eps=0):
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


def pachified_crops_generator(
    ppts_pd,
    vips_img,
    tf,
    crop_size=(768, 768),  # input crop to be patchified
    patch_size=(256, 256),  # used by quantification model
    preproc=True,
    augment=True,
    filter_cls=None,
    # filter_sparse=True,
):
    crop_size = (int(crop_size), int(crop_size))
    assert len(crop_size) == 2, "crop_size must be tuple of 2 elements"
    assert len(patch_size) == 2, "patch_size must be tuple of 2 elements"
    assert (
        crop_size[0] % patch_size[0] == 0
    ), "crop_size[0] must be divisible by patch_size[0]"
    assert (
        crop_size[1] % patch_size[1] == 0
    ), "crop_size[1] must be divisible by patch_size[1]"
    assert (
        crop_size[0] >= patch_size[0]
    ), "crop_size[0] must be greater than patch_size[0]"
    assert (
        crop_size[1] >= patch_size[1]
    ), "crop_size[1] must be greater than patch_size[1]"

    # print (f"Before: {len(crops_ppts_pd)}")
    pts_info = ppts_pd
    # if filter_sparse and 'sparse_patch' in ppts_pd.columns:
    #     pts_info = ppts_pd[ppts_pd['sparse_patch'] == False]
    if filter_cls is not None and "class" in ppts_pd.columns:
        if isinstance(filter_cls, tuple) or isinstance(filter_cls, list):
            ## pts_info = ppts_pd[ppts_pd['class'].isin(filter_cls)]
            pts_info = ppts_pd.filter(ppts_pd["class"].is_in(filter_cls))
        elif isinstance(filter_cls, int):
            pts_info = ppts_pd[ppts_pd["class"] == filter_cls]
        else:
            raise ValueError("filter_cls must be tuple or list of int or int")
    # print (f"After filters: {len(pts_info)}")

    crop_height = crop_size[1]
    patch_width = patch_size[0]

    ### Iterate
    ## for row in pts_info.itertuples():
    for row in pts_info.iter_rows():
        ## parent_crop_index = row.Pair_ID
        ## pts = eval(str(row.Points_pair))

        parent_crop_index = row[0]
        pts = eval(str(row[1]))

        # pts = eval(row.Points_pair) if type(row.Points_pair) is str else row.Points_pair

        crop = extract_patch(vips_img, pts, crop_height).numpy()

        # Fix an issue with border procudeure, when output patch shape is several pixels smaller in height
        if crop.shape[0:2] != crop_size:
            # cv2.imwrite(f"crop_{parent_crop_index}.png", crop)
            # print (f"crop.shape: {crop.shape}")
            crop = cv2.resize(crop, crop_size, interpolation=cv2.INTER_AREA)

        # print (f"crop.shape: {crop.shape}")
        if crop_size == patch_size:
            preproc_img = {"image": crop.squeeze()}
            if preproc:
                preproc_img["image"] = Image.fromarray(preproc_img["image"])
                preproc_img["image"] = tf["preproc"](preproc_img["image"])
                preproc_img["image"] = preproc_img["image"].permute(1, 2, 0)
                preproc_img["image"] = np.asarray(preproc_img["image"])
            if augment:
                preproc_img = tf["aug"](image=preproc_img["image"])
            preproc_img = tf["resize_to_tensor"](image=preproc_img["image"])
            img = preproc_img["image"]
            index = (parent_crop_index, 0, 0)
            yield (img, index)

        else:
            patches = patchify(crop, (*patch_size, 3), step=patch_width)
            # print (f"patches.shape: {patches.shape}")
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    preproc_img = {"image": patches[i, j].squeeze()}
                    if preproc:
                        preproc_img["image"] = Image.fromarray(preproc_img["image"])
                        preproc_img["image"] = tf["preproc"](preproc_img["image"])
                        preproc_img["image"] = preproc_img["image"].permute(1, 2, 0)
                        preproc_img["image"] = np.asarray(preproc_img["image"])
                    if augment:
                        preproc_img = tf["aug"](image=preproc_img["image"])
                    preproc_img = tf["resize_to_tensor"](image=preproc_img["image"])
                    img = preproc_img["image"]
                    index = (parent_crop_index, i, j)
                    yield (img, index)


class TmpPatchDataset(Dataset):
    def __init__(
        self,
        ppts_pd,
        vips_img,
        tf,
        patch_height=768,
        preproc=True,
        augment=True,
        filter_cls=None,
        # filter_sparse=True,
    ):

        self.pts_info = ppts_pd
        # if filter_sparse and 'sparse_patch' in ppts_pd.columns:
        #     self.pts_info = ppts_pd[ppts_pd['sparse_patch'] == False]
        if filter_cls is not None and "class" in ppts_pd.columns:
            if isinstance(filter_cls, tuple) or isinstance(filter_cls, list):
                self.pts_info = ppts_pd[ppts_pd["class"].isin(filter_cls)]
            elif isinstance(filter_cls, int):
                self.pts_info = ppts_pd[ppts_pd["class"] == filter_cls]
            else:
                raise ValueError("filter_cls must be tuple or list of int or int")

        self.vips_img = vips_img
        self.transforms = tf
        self.patch_height = patch_height
        self.preproc = preproc
        self.augment = augment

        # self.idx_to_label = {x[0]: x[1] for x in self.pts_info}

    def __len__(self):
        return len(self.pts_info)

    def __getitem__(self, idx):
        img_info = self.pts_info.iloc[idx]

        index = img_info["Pair_ID"]
        pts = (
            eval(img_info["Points_pair"])
            if type(img_info["Points_pair"]) is str
            else img_info["Points_pair"]
        )

        preproc_img = {
            "image": self.extract_patch(vips_img, pts, self.patch_height).numpy()
        }
        if self.preproc:
            preproc_img["image"] = Image.fromarray(preproc_img["image"])
            preproc_img["image"] = self.transforms["preproc"](preproc_img["image"])
            preproc_img["image"] = preproc_img["image"].permute(1, 2, 0)
            preproc_img["image"] = np.asarray(preproc_img["image"])
        if self.augment:
            preproc_img = self.transforms["aug"](image=preproc_img["image"])
        preproc_img = self.transforms[f"resize_to_tensor"](image=preproc_img["image"])
        img = preproc_img["image"]
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
        rect_points = self.find_rectangular_points(p0, p1, patch_height)
        bbox_points = self.find_bounding_rect(rect_points, local_offset=0)
        width = bbox_points[2].x - bbox_points[0].x
        height = bbox_points[2].y - bbox_points[0].y
        rect_area_crop = vips_img.crop(
            int(bbox_points[0].x),
            int(bbox_points[0].y),
            int(width),
            int(height),
            # int(N(bbox_points[0].x, 2)),
            # int(N(bbox_points[0].y, 2)),
            # int(N(width, 2)),
            # int(N(height, 2)),
        )
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
        rotated_rect_area_crop = rect_area_crop.rotate(deg(angle))
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
        rotated_bbox_points = self.find_bounding_rect(
            rotated_rec_points, local_offset=0
        )
        width = rotated_bbox_points[2].x - rotated_bbox_points[0].x
        height = rotated_bbox_points[2].y - rotated_bbox_points[0].y
        return rotated_rect_area_crop.crop(
            int(rotated_bbox_points[0].x),
            int(rotated_bbox_points[0].y),
            int(width),
            int(height),
            # int(N(rotated_bbox_points[0].x, 2)),
            # int(N(rotated_bbox_points[0].y, 2)),
            # int(N(width, 2)),
            # int(N(height, 2)),
        )
