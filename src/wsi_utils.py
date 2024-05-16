import os
import math
from math import ceil
import configparser
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from skimage.color import rgb2hed
from skimage.measure import regionprops

# from skimage.measure import label

import polars as pl
import torch
from torch_staintools.normalizer import NormalizerBuilder
from torchvision.transforms import v2 as tv2

import cv2
import pyvips
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sympy import Point, Ray, Line, Segment, Polygon, deg, sqrt, cos, sin, N


def get_n_points_polygons(config_polygon):
    n_poly_points = {}
    polygons = get_polypons(config_polygon)
    for p in polygons.keys():
        n_poly_points[p] = len(polygons[p])
    return n_poly_points


def get_polypons(config_polygon):
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


def _scale_coords(image_shape, scaled_image_shape, original_coords):
    """
    image: cv2 image
    corods: e.g.
        {
        0:[(x0,y0), (x1,y1),...],
        1:[(x0,y0), (x1,y1),...],
        }
    """
    ow, oh, och = image_shape
    w, h, ch = scaled_image_shape
    M = cv2.getAffineTransform(
        np.float32([[ow, 0], [ow, oh], [0, oh]]), np.float32([[w, 0], [w, h], [0, h]])
    )
    result = {}
    for k in original_coords.keys():
        npcoords = np.array([original_coords[k]], np.float32)  # np.int32?
        result[k] = cv2.transform(npcoords, M).squeeze()
    return result


def draw_all_polys(image, coords, fill=False, color=(0, 255, 0), lt=5):
    """
    #https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python
    #https://stackoverflow.com/questions/65522469/create-mask-from-roi?noredirect=1

    image: cv2 image, convert PIL using np.asarray()
    corods: e.g.
        {
        0:[(x0,y0), (x1,y1),...],
        1:[(x0,y0), (x1,y1),...],
        }
    """
    for k in coords.keys():
        npcoords = np.array([coords[k]], np.int32)
        if fill:
            cv2.fillPoly(image, npcoords, color)
        else:
            cv2.polylines(image, npcoords, True, color, lt)

    return image


def draw_polygon(image, points, fill=False, color=(0, 255, 0), lt=5, is_closed=False):
    pts = np.array([points], np.int32)
    if fill:
        cv2.fillPoly(image, pts, color)
    else:
        cv2.polylines(image, pts, is_closed, color, lt)
    return image


def plot_patches_on_slide(slide_info, points_pairs_list):
    slide_path = slide_info["path"]
    slide_vips = extract_slide(slide_path, slide_info, level=3)
    # create white background
    bg = pyvips.Image.black(slide_vips.width, slide_vips.height, bands=4)

    for points_pair in points_pairs_list:
        scaled_points_pair = scale_coords(
            points_pair, slide_info["raw_size"], (slide_vips.width, slide_vips.height)
        )
        left = min(scaled_points_pair[0][0], scaled_points_pair[1][0])
        top = min(scaled_points_pair[0][1], scaled_points_pair[1][1])
        width = abs(scaled_points_pair[0][0] - scaled_points_pair[1][0])
        height = abs(scaled_points_pair[0][1] - scaled_points_pair[1][1])
        if height < width:
            height = width

        bg = bg.draw_rect([0, 255, 0, 255], left, top, width, height, fill=False)

    plt.figure(figsize=(10, 10))
    # make it transparent
    plt.imshow(slide_vips, alpha=0.5)
    plt.imshow(bg)
    plt.show()


def get_annotated_image(
    slide, annotations, level, fill=False, color=(0, 255, 0), lt=5, verbose=False
):
    """
    slide: openslide object
    level: downsampling level from 1 to 4 usually, maps to downsampling factor in `slide.level_downsamples`
    annotations: points, obtained from annotation `.itn` file

    Return: np.array
    """
    original_size = (*slide.dimensions, 3)
    scaled_size = (*slide.level_dimensions[level - 1], 3)
    scaled_polys = _scale_coords(original_size, scaled_size, annotations)
    pil_img = slide.read_region(
        (0, 0), level - 1, slide.level_dimensions[level - 1]
    ).convert("RGB")

    if verbose:
        print(f"Shape of slide: {np.shape(pil_img)}")
        plt.figure(dpi=200)
        plt.imshow(pil_img)

    res_img = draw_all_polys(
        np.asarray(pil_img), scaled_polys, fill=fill, color=color, lt=lt
    )

    if verbose:
        plt.figure(dpi=200)
        plt.imshow(res_img)

    return res_img, scaled_polys


def get_annotated_image_vips(
    slide_path,
    slide_info,
    l0_annotations,
    level=None,
    fill=False,
    color=(0, 255, 0),
    lt=5,
    verbose=False,
):
    """
    slide: vips object
    level: downsampling level from 1 to 4 usually, maps to downsampling factor in `slide.level_downsamples`
    annotations: points, obtained from annotation `.itn` file

    Return: np.array
    """
    lvl = slide_info["level_count"] - 1 if level is None else level

    original_size = (*slide_info["raw_size"], 3)
    scaled_size = (*slide_info["level_dimensions"][lvl][1], 3)
    scaled_polys = _scale_coords(original_size, scaled_size, l0_annotations)

    pil_img = extract_slide(slide_path, slide_info, level=lvl).numpy()

    if verbose:
        print(f"Shape of slide: {np.shape(pil_img)}")
        plt.figure(dpi=200)
        plt.imshow(pil_img)

    res_img = draw_all_polys(
        np.asarray(pil_img), scaled_polys, fill=fill, color=color, lt=lt
    )

    if verbose:
        plt.figure(dpi=200)
        plt.imshow(res_img)

    return res_img, scaled_polys


def visualize_contours(
    slide_path,
    slide_info,
    l0_annotations,
    save_file_path,
    level=None,
    fill=False,
    color=(0, 255, 0),
    lt=5,
):
    """
    slide: vips object
    level: downsampling level from 1 to 4 usually, maps to downsampling factor in `slide.level_downsamples`
    annotations: points, obtained from annotation `.itn` file

    Return: np.array
    """
    lvl = slide_info["level_count"] - 1 if level is None else level
    original_size = (*slide_info["raw_size"], 3)
    scaled_size = (*slide_info["level_dimensions"][lvl][1], 3)
    scaled_polys = _scale_coords(original_size, scaled_size, l0_annotations)
    np_img = extract_slide(slide_path, slide_info, level=lvl).numpy()

    plt.figure(dpi=200)
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(draw_all_polys(np_img, scaled_polys, fill=fill, color=color, lt=lt))
    plt.savefig(save_file_path)
    plt.close()


def extract_patches_from_point(image, coords, size=24):
    """
    image: cv2 image, convert PIL using np.asarray()
    corods: e.g.
        {
        0:[(x0,y0), (x1,y1),...],
        1:[(x0,y0), (x1,y1),...],
        }
    """
    patches = []
    for k in coords.keys():
        npcoords = np.squeeze(np.array([coords[k]], np.int32))
        for x, y in npcoords:
            crop_img = image[y : y + size, x : x + size]
            patches.append(crop_img)
    return patches


def extract_patches_btw_points(img, p1, p2, patch_size=748):
    """
    Arguments:
        - img: cv2.image
        - p1, p2: (x, y) points
    """
    points = [p1, p2]
    points.sort(key=lambda x: x[0], reverse=True)

    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    dx = points[0][0] - points[1][0]
    dy = points[0][1] - points[1][1]

    angle = atan2(dy, dx)

    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=math.degrees(angle), scale=1
    )
    p1_r = np.array(rotate(points[0], center, math.degrees(angle)))
    p2_r = np.array(rotate(points[1], center, math.degrees(angle)))

    rotated_image = cv2.warpAffine(image, rotate_matrix, dsize=(width, height))

    dx = p2_r[0] - p1_r[0]
    dy = p2_r[1] - p1_r[1]
    distance_after = sqrt(dx**2 + dy**2)  # 850

    n_full_patches = int(distance_after // patch_size)
    residual = int(distance_after % patch_size)

    rot_points = [p1_r, p2_r]
    rot_points.sort(key=lambda x: x[0], reverse=False)

    line_stops = [(rot_points[0][0], rot_points[0][1])]
    for num in range(1, n_full_patches):
        line_stops.append((rot_points[0][0] + patch_size * num, rot_points[0][1]))

    rimgs = []
    for idx, line_stop in enumerate(line_stops):
        x, y = line_stop
        y = y - int(patch_size / 2)
        if n_full_patches > 0:
            crop_img = rotated_image[y : y + patch_size, x : x + patch_size]
            rimgs.append(crop_img)

            if idx == len(line_stops) - 1 and residual != 0:
                crop_img = rotated_image[
                    y : y + patch_size, x + patch_size : (x + patch_size) + residual
                ]
                rimgs.append(crop_img)
        else:
            crop_img = rotated_image[y : y + patch_size, x : x + residual]
            rimgs.append(crop_img)

    return rimgs


def extract_slide_info(slide_path):
    ref_slide = pyvips.Image.new_from_file(slide_path)
    levels = int(ref_slide.get("openslide.level-count"))
    level_dimensions = {}
    for i in range(levels):
        level_dimensions[i] = (
            ref_slide.get(f"openslide.level[{i}].downsample"),
            (
                int(ref_slide.get(f"openslide.level[{i}].width")),
                int(ref_slide.get(f"openslide.level[{i}].height")),
            ),
        )

    slide_id = str(Path(slide_path).stem)
    if (
        "aperio.ImageID" in ref_slide.get_fields()
        and "aperio.Filename" in ref_slide.get_fields()
    ):
        aperio_id = ref_slide.get("aperio.ImageID")
        aperio_file_name = ref_slide.get("aperio.Filename")

        if slide_id != aperio_file_name:
            if f"{aperio_file_name}" in slide_id:
                if f"{aperio_file_name}." in slide_id:
                    suffix = f"_{slide_id.split(f'{aperio_file_name}.')[1]}"
                else:
                    suffix = f"_{slide_id.split(f'{aperio_file_name}')[1]}"
                slide_id = f"{aperio_id}{suffix}"
            else:
                slide_id = f"{aperio_id}_{slide_id}"

        else:
            slide_id = aperio_id

    file_name = Path(slide_path).stem
    if "aperio.Filename" in ref_slide.get_fields():
        file_name = ref_slide.get("aperio.Filename")

    if "aperio.MPP" not in ref_slide.get_fields():
        raise ValueError(f"No MPP field ('aperio.MPP') in slide {slide_path}")

    mag_level = int(ref_slide.get("aperio.AppMag"))
    # if int(mag_level) != 40:
    #     raise ValueError(f"Mag level is ({mag_level}), which is not 40 in slide {slide_path}")

    return {
        "id": slide_id,
        "name": file_name,
        "level_count": levels,
        "raw_size": (int(ref_slide.get("width")), int(ref_slide.get("height"))),
        "mpp": float(ref_slide.get("aperio.MPP")),
        "mag_level": int(ref_slide.get("aperio.AppMag")),
        "level_dimensions": level_dimensions,
        "path": slide_path,
    }


def extract_slide(slide_path, slide_info, add_alpha=False, level=None):
    n_bands = 4 if add_alpha else 3
    if level is None:
        level = slide_info["level_count"] - 1
    slide = pyvips.Image.new_from_file(slide_path, level=level).extract_band(
        0, n=n_bands
    )
    return slide.crop(0, 0, *slide_info["level_dimensions"][level][1])


def scale_polygons_coords(reference_coords, reference_size, target_size):
    reference_size = reference_size[0:2] if len(reference_size) == 3 else reference_size
    target_size = target_size[0:2] if len(target_size) == 3 else target_size
    result = {}
    for k in reference_coords.keys():
        result[k] = []
        for item in reference_coords[k]:
            updated_item = np.floor(
                (np.asarray(item).ravel() * target_size) / reference_size
            ).astype("int64")
            result[k].append(tuple(updated_item))
    return result


def scale_coords(reference_coords, reference_size, target_size):
    result = []
    for item in reference_coords:
        updated_item = np.floor(
            (np.asarray(item).ravel() * target_size) / reference_size
        ).astype("int64")
        result.append(tuple(updated_item))
    return result


def remove_small_objects(binary_mask, min_obj_size=50):
    """
    Usage:
    predicted_mask = np.array([[0, 1, 0, 0],
                               [1, 1, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 0, 1]])
    refined_mask = remove_small_objects(predicted_mask, min_obj_size=4)
    """
    labeled_mask, num_features = label(binary_mask)
    filtered_mask = np.zeros_like(binary_mask)

    for region in regionprops(labeled_mask):
        if region.area >= min_obj_size:
            for coord in region.coords:
                filtered_mask[coord[0], coord[1]] = 1

    return filtered_mask


def save_poligons_to_itn(polygons_dict, output_filepath):
    # polygons_dict = {
    #     0: [(x1, y1), (x2, y2)],
    #     1: [(x3, y3), (x4, y4), (x5, y5)],
    #     ...
    # }
    with open(output_filepath, "w") as f:
        f.write(f"[Polygon]\n")
        for poly_number, points in polygons_dict.items():
            for point_number, (x, y) in enumerate(points):
                f.write(f"Poly_X_{poly_number}_{point_number}={x}\n")
                f.write(f"Poly_Y_{poly_number}_{point_number}={y}\n")


def get_polypons(config_polygon):
    """
    Get polygon information in a dictionary from parsed .itn file.

    Parameters:
    - config_polygon (configparser.SectionProxy):   config['Polygon'] object

    Returns:
    - poly_points (dict):   polygon information, like
        {
            0: [(x0,y0), (x1,y1),...],
            1: [(x0,y0), (x1,y1),...],
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


def find_point_on_line(p1, p2, threshold):
    # Convert points to numpy arrays
    point1 = np.array(p1, dtype=np.float32)
    point2 = np.array(p2, dtype=np.float32)

    # Calculate the vector from point1 to point2
    vector = point2 - point1

    # Normalize the vector
    vector_norm = vector / np.linalg.norm(vector)

    # Multiply the normalized vector by the distance x
    # Then add to the original point1 to get the new point

    # TODO check: rounding
    new_point = np.round(point1 + vector_norm * threshold)

    return new_point


def _reduce_polygons_points(polygons, threshold, close=True):
    result = {}

    for k, polygon in polygons.items():
        polygon = np.array(polygon)
        p1 = polygon[0]
        result[k] = [p1.tolist()]

        for p2 in polygon[1:]:
            dist = np.linalg.norm(p2 - p1)

            if dist > threshold:
                result[k].append(p2.tolist())
                p1 = p2

        if close:
            result[k].append(polygon[0].tolist())

        # if 2 points are the same - make new 2 points depending on threshold
        if len(set(map(tuple, result[k]))) == 1:
            np0, np1 = map(
                lambda x: (x + threshold / 2 * np.array([-1, 1])).tolist(),
                [result[k][0]] * 2,
            )
            result[k] = [np0, np1]

    return result


def calculate_closing_patch_points(first_point, last_point, threshold):
    fp = np.array(first_point, dtype=np.int32)
    lp = np.array(last_point, dtype=np.int32)
    midpoint = (fp + lp) / 2

    new_first_point = find_point_on_line(midpoint, first_point, threshold / 2)
    new_last_point = find_point_on_line(midpoint, last_point, threshold / 2)
    return new_last_point, new_first_point


def reduce_polygons_points(polygons, threshold, fix_last_missing=True):
    reduced_polygons = {}

    for key, polygon in polygons.items():
        polygon = np.array(polygon)
        initial_point = polygon[0]
        reduced_polygon = [initial_point]

        for next_point in polygon[1:]:
            distance = np.linalg.norm(next_point - initial_point)
            if np.isclose(distance, threshold) or abs(distance - threshold) < 1.0:
                reduced_polygon.append(next_point)
                initial_point = next_point
            elif distance - threshold >= 1.0:
                updated_point = find_point_on_line(initial_point, next_point, threshold)
                reduced_polygon.append(updated_point)
                initial_point = updated_point

        # if distance between last and first points is more than threshold, we add a point
        if fix_last_missing:
            first_point = reduced_polygon[0]
            last_point = reduced_polygon[-1]
            distance = np.linalg.norm(last_point - first_point)

            if distance != 0.0:
                while distance > threshold:
                    updated_point = find_point_on_line(
                        last_point, first_point, threshold
                    )
                    reduced_polygon.append(updated_point)
                    last_point = updated_point
                    distance = np.linalg.norm(last_point - first_point)

        if len(set(map(tuple, reduced_polygon))) == 1:
            np0 = [polygon[0][0] - int(threshold / 2), polygon[0][1]]
            np1 = [polygon[0][0] + int(threshold / 2), polygon[0][1]]
            reduced_polygon = [np0, np1]

        reduced_polygons[key] = reduced_polygon

    return reduced_polygons


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


def find_polygon_bounding_rect(polygons, local_offset=None):
    result = {}
    for k in polygons.keys():
        xs = list(zip(*polygons[k]))[0]
        ys = list(zip(*polygons[k]))[1]
        top_left = Point(min(xs), min(ys))
        top_right = Point(max(xs), min(ys))
        bottom_right = Point(max(xs), max(ys))
        bottom_left = Point(min(xs), max(ys))

        if local_offset is not None:
            tl = Point(top_left.x - local_offset, top_left.y - local_offset)
            tr = Point(top_right.x + local_offset, top_right.y - local_offset)
            br = Point(bottom_right.x + local_offset, bottom_right.y + local_offset)
            bl = Point(bottom_left.x - local_offset, bottom_left.y + local_offset)
            result[k] = (tl, tr, br, bl)
        else:
            result[k] = (top_left, top_right, bottom_right, bottom_left)
    return result


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


def get_points_pair_from_slide_reduced_polys(slide, reduced_polys, threshold):
    upd_reduced_polys = {}

    for k in reduced_polys.keys():
        upd_reduced_polys[k] = (reduced_polys[k], None)

    ### Stage 2 ###
    # Add closure if distance > threshold / 3
    for k in upd_reduced_polys.keys():
        pts = upd_reduced_polys[k][0]
        last_point, first_point = pts[-1], pts[0]
        distance = np.linalg.norm(
            np.array(last_point, dtype=np.float32)
            - np.array(first_point, dtype=np.float32)
        )
        if distance > (threshold / 3):
            upd_last_point, upd_first_point = calculate_closing_patch_points(
                first_point, last_point, threshold
            )
            upd_reduced_polys[k] = (
                upd_reduced_polys[k][0],
                (Point(upd_last_point), Point(upd_first_point)),
            )

    ### Stage 3 ###
    # Transform to list for dataloader
    slide_with_point_pair_list = []
    for k in upd_reduced_polys.keys():
        poly_pts = upd_reduced_polys[k][0]
        closing = upd_reduced_polys[k][1]

        for p1_idx in range(len(poly_pts) - 1):
            # lst_entry = (k, crop, (poly_pts[p1_idx], poly_pts[p1_idx+1]))
            lst_entry = (k, (Point(poly_pts[p1_idx]), Point(poly_pts[p1_idx + 1])))
            slide_with_point_pair_list.append(lst_entry)
        if closing is not None:
            # last_lst_entry = (k, crop, closing)
            last_lst_entry = (k, closing)
            slide_with_point_pair_list.append(last_lst_entry)

    # Add index
    slide_with_point_pair_list = [
        (idx,) + el for idx, el in enumerate(slide_with_point_pair_list)
    ]

    # crops_with_point_pair_list: [(index, crop_index, vips_crop, points_pair)]
    # crops_with_polygons: {crop_index: (vips_crop, [points])}
    return slide_with_point_pair_list


def rotate_point(orig_shape, rot_shape, point, angle_rad):
    """
    Get new coordinates for point on a rotated image.

    Parameters:
    - img_size (tuple): size of original image (w, h)
    - rotated_size (tuple): size of rotated image (w, h) by `angle`
    - point (tuple): point of type (x, y) that should be rotated
    - angle (float): angle of the rotation in radians

    Returns:
    - rotated_point (tuple): rotated point of type (x, y)
    """
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
        round(width),  # int(N(width, 2)),
        round(height),  # int(N(height, 2))
    )


def is_row_high_intensity(
    vips_crop_from, pts_pair, patch_height, ms={"mean": 210, "std": 30}
):
    # 222 30 - almost empty - high intensity
    # 143 50 - full
    vips_patch = extract_patch(vips_crop_from, pts_pair, patch_height)
    return bool(vips_patch.avg() >= ms["mean"] and vips_patch.deviate() <= ms["std"])


def is_row_empty_H(vips_img_from, pts_pair, patch_height, threshold=0.017):  # or 0.012
    vips_patch = extract_patch(vips_img_from, pts_pair, patch_height)
    patch_hed = rgb2hed(vips_patch.numpy())
    # if patch_hed.shape != (patch_height, patch_height, 3):
    #     print (f"Patch shape: {patch_hed.shape}")
    # # Extract the Hematoxylin channel and get avg value
    return bool(patch_hed[:, :, 0].mean() < threshold)


def is_row_empty_BW(vips_crop_from, pts_pair, patch_height, threshold=40.0):
    vips_patch = extract_patch(vips_crop_from, pts_pair, patch_height)
    w_component = vips_patch.colourspace(c)[1]
    return bool(w_component.avg() < threshold)


def get_hematoxilin_component(vips_patch_crop):
    """
    Returns True if the Hematoxylin component is enough
    """
    patch_hed = rgb2hed(vips_patch_crop.numpy())
    return patch_hed[:, :, 0].mean()


def is_hematoxilin_enough(vips_patch_crop, threshold=0.017):
    """
    Returns True if the Hematoxylin component is enough
    """
    # binary_mask = (img.colourspace('b-w') < 200).numpy()
    # masked_img = np.where(binary_mask[..., None], img.numpy(), 255).astype(np.uint8)
    # mask = (masked_img > 230).astype(np.uint8)
    # emphasized_img = masked_img * (1 - mask) + 255 * mask

    patch_hed = rgb2hed(vips_patch_crop.numpy())
    return bool(patch_hed[:, :, 0].mean() >= threshold)


def is_hematoxilin_enough_normed_H(vips_patch_crop, normalizer, tf, threshold=0.017):
    gray_mask = vips_patch_crop.colourspace("b-w")
    binary_mask = ~(gray_mask > 180).numpy()
    # select only binary mask on image
    masked_img = np.where(binary_mask[..., None], vips_patch_crop.numpy(), 255).astype(
        np.uint8
    )
    # apply normalization ?
    # masked_img = normalizer(masked_img)
    patch_hed = rgb2hed(masked_img)
    return bool(patch_hed[:, :, 0].mean() >= threshold)


def is_intensity_low(vips_patch_crop, ms={"mean": 210, "std": 30}):
    # 222 30 - almost empty - high intensity
    # 143 50 - full
    """
    Returns True if the intensity is low
    """
    return bool(
        vips_patch_crop.avg() < ms["mean"] and vips_patch_crop.deviate() > ms["std"]
    )


def is_bw_not_empty(vips_patch_crop, threshold=0.2):
    # 1 - object, 0 - background
    gray_mask = vips_patch_crop.colourspace("b-w")
    binary_mask = ~(gray_mask > 180).numpy()
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        area_p = area / (binary_mask.shape[0] * binary_mask.shape[1])
        return area / (binary_mask.shape[0] * binary_mask.shape[1]) > threshold
    else:
        return False


def is_within_mask(patch_params, mask, mask_threshold=0.4):
    """
    Returns True if the patch is within the mask
    """
    left, top, width, height = patch_params
    mask_patch = mask[top : top + height, left : left + width]
    return bool(np.mean(mask_patch) > mask_threshold)


def calculate_norm_mm2(slide_mpp, patch_size, abs_counts):
    """
    Calculate cell density in cells per mm^2 for ONE patch
    """
    area_in_microns_sq = (patch_size * slide_mpp) * (patch_size * slide_mpp)
    cell_density_in_cells_per_micron_sq = abs_counts / area_in_microns_sq
    conversion_factor = 1000 * 1000  # 1000 microns in a millimeter
    cell_density_in_cells_per_mm_sq = (
        cell_density_in_cells_per_micron_sq * conversion_factor
    )
    return cell_density_in_cells_per_mm_sq


def calculate_cells_score(
    slide_info, cell_types, s4_dataframe, patch_size=256, drop_zero=False
):
    result = {}
    dfo = s4_dataframe.clone()
    for cell_type in cell_types:
        norm_col = f"{cell_type}_norm_mm2"
        # Get normalized values using mpp for each sub-patch
        dfo = dfo.with_columns(
            pl.col(cell_type)
            .map_elements(
                lambda value: calculate_norm_mm2(
                    slide_info["mpp"], patch_size, int(value)
                ),
                return_dtype=pl.Float64,
            )
            .alias(norm_col)
        )

        # Summarize norm_mm2 column by "Pair_ID"
        dfo_sum = dfo.group_by(["Pair_ID"]).agg(
            [pl.col(norm_col).sum().alias(norm_col)]
        )

        if drop_zero:
            dfo_sum = dfo_sum.filter(pl.col(norm_col) != 0)

        # Divide by number of patches (with the same Pair_ID)
        pair_id_counts = dfo.group_by("Pair_ID").agg(pl.count("Pair_ID").alias("count"))
        dfo_sum = dfo_sum.join(pair_id_counts, on="Pair_ID")
        dfo_sum = dfo_sum.with_columns(
            (pl.col(norm_col) / pl.col("count")).alias(norm_col)
        )
        result[cell_type] = round(dfo_sum[norm_col].mean(), 3)

    return result


def generate_tissue_mask(
    slide_path,
    binary_threshold=170,
    min_area_threshold=5000,
    border_px=50,
    save_tissue_contour_image=True,
):
    slide_info = extract_slide_info(slide_path)
    available_levels = sorted(list(slide_info["level_dimensions"].keys()))
    extraction_level = 3 if 3 in available_levels else available_levels[-1]

    # slide_lowres = extract_slide(slide_path, slide_info, level=extraction_level)

    try:
        slide_lowres = extract_slide(
            slide_path, slide_info, level=extraction_level
        ).numpy()
        slide_lowres = extract_slide(slide_path, slide_info, level=extraction_level)
    except:
        slide_lowres = extract_slide(
            slide_path, slide_info, level=extraction_level - 1
        ).numpy()
        slide_lowres = extract_slide(slide_path, slide_info, level=extraction_level - 1)

    #########################
    gray_mask = slide_lowres.colourspace("b-w")
    binary_mask = gray_mask > binary_threshold

    # binary_mask = binary_mask.gaussblur(5)

    # 1 - object, 0 - background
    binary_mask = ~binary_mask.numpy()

    # zero out 50 pixels from all sides to remove border artifacts
    binary_mask[:border_px, :] = 0
    binary_mask[-border_px:, :] = 0
    binary_mask[:, :border_px] = 0
    binary_mask[:, -border_px:] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    filtered_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area_threshold:
            filtered_mask[labels == i] = 255
    contours, _ = cv2.findContours(
        filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # # only largest_contour
    # reference_polygons = {0: max(contours, key=cv2.contourArea).squeeze()}

    reference_polygons = {}
    for i, contour in enumerate(contours):
        reference_polygons[i] = contour.squeeze()

    reference_size = (binary_mask.shape[1], binary_mask.shape[0])
    target_size = slide_info["raw_size"]

    l0_polys = scale_polygons_coords(reference_polygons, reference_size, target_size)
    mask = np.zeros(
        (slide_info["raw_size"][1], slide_info["raw_size"][0]), dtype=np.uint8
    )

    for poly in l0_polys.values():
        mask = draw_polygon(mask, poly, fill=True, color=1)

    if save_tissue_contour_image:
        binary_mask_color = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)
        # largest_contour_image = np.zeros_like(binary_mask_color)
        largest_contour_image = slide_lowres.numpy()
        for contour in contours:
            cv2.drawContours(largest_contour_image, [contour], -1, (0, 255, 0), 10)
        save_img_path = Path("debug_imgs", "tissue_contours")
        if save_img_path.exists() is False:
            save_img_path.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 10), dpi=200)
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(largest_contour_image)
        plt.savefig(Path(save_img_path, f"_S2TC_{slide_info['name']}.png"))
        plt.close()

    return mask.astype(bool), l0_polys


def grid_H_patch_coord_generator(
    slide_path,
    roi_mask=None,
    patch_size=None,
    ignore_tissue_mask=False,
    sort_h_min_threshold=None,
    mask_threshold=0.4,
    s1_polygons=None,
    min_num_patches=20,
    save_plot=False,
):
    """ """
    print(
        f"Using 'grid_H_patch_coord_generator' with {sort_h_min_threshold} threshold."
    )
    assert s1_polygons is not None, "s1_polygons is not defined"
    assert patch_size is not None, "patch_size is not defined"

    slide_info = extract_slide_info(slide_path)
    vips_image = extract_slide(slide_path, slide_info, level=0)

    ### Calculate the number of patches in x and y directions
    num_patches_x = math.ceil(vips_image.width / patch_size)
    num_patches_y = math.ceil(vips_image.height / patch_size)
    n_total_patches = num_patches_x * num_patches_y

    ### Generate patch_candidates
    patch_candidates = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            if j == num_patches_x - 1 and vips_image.width % patch_size != 0:
                left = vips_image.width - patch_size
            else:
                left = j * patch_size

            if i == num_patches_y - 1 and vips_image.height % patch_size != 0:
                top = vips_image.height - patch_size
            else:
                top = i * patch_size

            width = min(patch_size, vips_image.width - left)
            height = min(patch_size, vips_image.height - top)

            patch_candidates.append((left, top, width, height))

    vips_patch_candidates = [
        vips_image.crop(left, top, width, height)
        for left, top, width, height in patch_candidates
    ]
    cords_vips_patch_candidates = list(zip(patch_candidates, vips_patch_candidates))

    ### Mask definition #####################
    if not ignore_tissue_mask:
        mask, tissue_countours = generate_tissue_mask(
            slide_path, save_tissue_contour_image=save_plot
        )
    #########################################

    if mask is not None:
        cords_vips_patch_candidates = [
            cords_vips_patch_candidates[i]
            for i in range(len(cords_vips_patch_candidates))
            if (is_within_mask(cords_vips_patch_candidates[i][0], mask, mask_threshold))
        ]

    if sort_h_min_threshold is not None:
        patch_candidates_with_hema_all = [
            (x[0], get_hematoxilin_component(x[1])) for x in cords_vips_patch_candidates
        ]
        patch_candidates_with_hema_all = sorted(
            patch_candidates_with_hema_all, key=lambda x: x[1], reverse=True
        )

        patch_candidates_with_hema_cut = [
            x for x in patch_candidates_with_hema_all if x[1] > sort_h_min_threshold
        ]

        patch_candidates_all = [x[0] for x in patch_candidates_with_hema_all]
        patch_candidates_cut = [x[0] for x in patch_candidates_with_hema_cut]
    else:
        cords_vips_patch_candidates = [
            cords_vips_patch_candidates[i]
            for i in range(len(cords_vips_patch_candidates))
            if (is_hematoxilin_enough(cords_vips_patch_candidates[i][1]))
        ]
        patch_candidates_cut = [x[0] for x in cords_vips_patch_candidates]
        patch_candidates_all = patch_candidates_cut

    sum_target_polygons_area = 0
    for target_polygon in s1_polygons.values():
        target_polygon = np.array(target_polygon).reshape(-1, 2).astype(int)
        sum_target_polygons_area += cv2.contourArea(target_polygon)

    tissue_mask_area = 0
    for tissue_countour in tissue_countours.values():
        tissue_mask_area += cv2.contourArea(np.array(tissue_countour))

    tumor_to_tissue_area_ratio = sum_target_polygons_area / (tissue_mask_area + 1e-6)

    n_to_yield = max(
        int(min(tumor_to_tissue_area_ratio, 1.0) * len(patch_candidates_cut)),
        min_num_patches,
    )
    if n_to_yield > len(patch_candidates_cut):
        candidates_to_yield = patch_candidates_all
    else:
        candidates_to_yield = patch_candidates_cut

    print(
        f"Total patches: {n_total_patches}, cut: {len(patch_candidates_cut)} Patches to yield: {tumor_to_tissue_area_ratio} * {len(candidates_to_yield)} : {n_to_yield}"
    )

    for i in range(n_to_yield):
        yield (candidates_to_yield[i][0], candidates_to_yield[i][1])


def grid_patch_coord_generator(
    slide_path,
    roi_mask=None,
    perform_checks=True,
    patch_size=None,
    ignore_tissue_mask=False,
    sort_h=False,
    mask_threshold=0.4,
    to_yield=1.0,
    s1_polygons=None,
    benchmark=True,
    save_plot=False,
    logger=None,
):
    """
    For using with ratio of patches.

    This is slower version of the patch generator, but it uses tissue mask AND H_component check to do initial filter of candidates.
    This results in smaller amount of candidates and is closer to the desired number of patches to be yeilded.

    TODO:
    - generate patches using ratio with respect to each mask polygon. Need to pass mask polygons as an argument
    - ? intensity filter: <210 mean and >30 std
    """
    assert patch_size is not None, "patch_size is not defined"

    if logger is not None:
        if isinstance(to_yield, float):
            logger.info(
                f"Using 'grid_patch_coord_generator' for to_yield = {to_yield} ({to_yield * 100:0.2f}%)."
            )
        elif isinstance(to_yield, int):
            logger.info(
                f"Using 'grid_patch_coord_generator' for to_yield = {to_yield} (int). Consider using 'grid_patch_coord_generator' for float ratio."
            )

    slide_info = extract_slide_info(slide_path)
    vips_image = extract_slide(slide_path, slide_info, level=0)

    ### Calculate the number of patches in x and y directions
    num_patches_x = math.ceil(vips_image.width / patch_size)
    num_patches_y = math.ceil(vips_image.height / patch_size)
    n_total_patches = num_patches_x * num_patches_y

    ### Generate patch_candidates
    patch_candidates = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            if j == num_patches_x - 1 and vips_image.width % patch_size != 0:
                left = vips_image.width - patch_size
            else:
                left = j * patch_size

            if i == num_patches_y - 1 and vips_image.height % patch_size != 0:
                top = vips_image.height - patch_size
            else:
                top = i * patch_size

            width = min(patch_size, vips_image.width - left)
            height = min(patch_size, vips_image.height - top)

            patch_candidates.append((left, top, width, height))

    ### Shuffle patch_candidates
    if benchmark:
        np.random.seed(42)
    np.random.shuffle(patch_candidates)

    vips_patch_candidates = [
        vips_image.crop(left, top, width, height)
        for left, top, width, height in patch_candidates
    ]
    cords_vips_patch_candidates = list(zip(patch_candidates, vips_patch_candidates))
    if perform_checks:
        if roi_mask is not None:
            cords_vips_patch_candidates = [
                cords_vips_patch_candidates[i]
                for i in range(len(cords_vips_patch_candidates))
                if (
                    is_within_mask(
                        cords_vips_patch_candidates[i][0], roi_mask, mask_threshold
                    )
                )
            ]

            cords_vips_patch_candidates = [
                cords_vips_patch_candidates[i]
                for i in range(len(cords_vips_patch_candidates))
                if (is_hematoxilin_enough(cords_vips_patch_candidates[i][1]))
            ]
        else:

            ### By ignoring mask, we expand the possible number of candidates, and take all possible patches and run checks on each
            if ignore_tissue_mask:
                cords_vips_patch_candidates = [
                    cords_vips_patch_candidates[i]
                    for i in range(len(cords_vips_patch_candidates))
                    if (is_hematoxilin_enough(cords_vips_patch_candidates[i][1]))
                ]
            else:
                ### If we have a generated mask, we drasticly reduce the number of candidates before checks, and thus speed up the process of checking candidates
                tissue_mask, _ = generate_tissue_mask(
                    slide_path, save_tissue_contour_image=save_plot
                )

                cords_vips_patch_candidates = [
                    cords_vips_patch_candidates[i]
                    for i in range(len(cords_vips_patch_candidates))
                    if (
                        is_within_mask(
                            cords_vips_patch_candidates[i][0],
                            tissue_mask,
                            mask_threshold,
                        )
                    )
                ]

                cords_vips_patch_candidates = [
                    cords_vips_patch_candidates[i]
                    for i in range(len(cords_vips_patch_candidates))
                    if (is_hematoxilin_enough(cords_vips_patch_candidates[i][1]))
                ]

    ### Sort by hematoxylin in descending order
    if sort_h is not None:
        patch_candidates_with_hema = [
            (x[0], get_hematoxilin_component(x[1])) for x in cords_vips_patch_candidates
        ]
        patch_candidates = sorted(
            patch_candidates_with_hema, key=lambda x: x[1], reverse=True
        )
        patch_candidates = [x[0] for x in patch_candidates]
    else:
        patch_candidates = [x[0] for x in cords_vips_patch_candidates]

    ### Determine the number of patches to yield
    if isinstance(to_yield, float):
        n_patches_to_yield = int(to_yield * len(patch_candidates))
    elif isinstance(to_yield, int):
        n_patches_to_yield = min(to_yield, len(patch_candidates))
    else:
        n_patches_to_yield = len(patch_candidates)

    if logger is not None:
        logger.info(
            f"Patches to yield: {n_patches_to_yield}, Total eligible patches: {len(patch_candidates)}, Total patches: {n_total_patches}"
        )

    for i in range(n_patches_to_yield):
        yield (patch_candidates[i][0], patch_candidates[i][1])


def fast_grid_patch_coord_generator(
    slide_path,
    roi_mask=None,
    perform_checks=True,
    patch_size=None,
    ignore_tissue_mask=False,
    sort_h=False,
    mask_threshold=0.4,
    to_yield=1.0,
    s1_polygons=None,
    benchmark=True,
    save_plot=False,
    logger=None,
):
    """
    For using with int's of yielded patches

    This is the fastest version of the patch generator, but it uses only tissue mask to do initial filter of candidates.
    This results in higher number of candidates, and if percentage of candidates is used, it may result in higher amount of patches to be yeilded.

    Though it should work perfectly if to_yield is set to a int number.
    """
    assert patch_size is not None, "patch_size is not defined"

    if logger is not None:
        if isinstance(to_yield, float):
            logger.info(
                f"Using 'fast_grid_patch_coord_generator' for to_yield = {to_yield} ({to_yield * 100:0.2f}%). Consider using 'grid_patch_coord_generator' for float ratio."
            )
        elif isinstance(to_yield, int):
            logger.info(
                f"Using 'fast_grid_patch_coord_generator' for to_yield = {to_yield} (int)."
            )

    slide_info = extract_slide_info(slide_path)
    vips_image = extract_slide(slide_path, slide_info, level=0)

    num_patches_x = math.ceil(vips_image.width / patch_size)
    num_patches_y = math.ceil(vips_image.height / patch_size)
    n_total_patches = num_patches_x * num_patches_y

    if logger is not None:
        logger.info(f"Total number of patches in WSI: {n_total_patches}")

    ### Mask definition #####################
    mask = roi_mask
    if roi_mask is None and not ignore_tissue_mask:
        mask, _ = generate_tissue_mask(slide_path, save_tissue_contour_image=save_plot)
    #########################################

    ### Generate patch_candidates #####################
    all_coord_pivots = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            if j == num_patches_x - 1 and vips_image.width % patch_size != 0:
                left = vips_image.width - patch_size
            else:
                left = j * patch_size

            if i == num_patches_y - 1 and vips_image.height % patch_size != 0:
                top = vips_image.height - patch_size
            else:
                top = i * patch_size

            width = min(patch_size, vips_image.width - left)
            height = min(patch_size, vips_image.height - top)
            all_coord_pivots.append((left, top, width, height))

    vips_patch_candidates = [
        vips_image.crop(left, top, width, height)
        for left, top, width, height in all_coord_pivots
    ]
    cords_vips_patch_candidates = list(zip(all_coord_pivots, vips_patch_candidates))

    ### Shuffle patch_candidates

    if benchmark:
        np.random.seed(42)

    np.random.shuffle(cords_vips_patch_candidates)
    ###################################################

    ### Adjust n_total_patches, using mask ############
    if mask is not None:
        cords_vips_patch_candidates = [
            cords_vips_patch_candidates[i]
            for i in range(len(cords_vips_patch_candidates))
            if (is_within_mask(cords_vips_patch_candidates[i][0], mask, mask_threshold))
        ]

        ### to make tighter selection, we can check for hematoxylin, but checking all of them will be time consuming
        ### but then we can just sample n candidates without checking

        # cords_vips_patch_candidates = [cords_vips_patch_candidates[i] for i in range(len(cords_vips_patch_candidates)) if
        # (
        #     is_hematoxilin_enough(cords_vips_patch_candidates[i][1])
        # )
        # ]

        n_total_patches = len(cords_vips_patch_candidates)
    ###################################################

    if isinstance(to_yield, float):
        n_patches_to_yield = int(to_yield * n_total_patches)
    elif isinstance(to_yield, int):
        n_patches_to_yield = min(to_yield, n_total_patches)
    else:
        n_patches_to_yield = n_total_patches

    if logger is not None:
        info_n_patches = (
            f"total WSI patches" if mask is None else "eligible patches, using mask"
        )
        logger.info(
            f"Patches to yield: {n_patches_to_yield} out of {n_total_patches} {info_n_patches}"
        )

    approved_candidates = []
    all_patches_idx = 0

    while len(approved_candidates) < n_patches_to_yield and all_patches_idx < len(
        cords_vips_patch_candidates
    ):
        candidate_subset = cords_vips_patch_candidates[
            all_patches_idx : all_patches_idx
            + (n_patches_to_yield - len(approved_candidates))
        ]
        all_patches_idx += n_patches_to_yield

        if perform_checks:
            approved_subset_candidates = [
                candidate_subset[i]
                for i in range(len(candidate_subset))
                if (is_hematoxilin_enough(candidate_subset[i][1]))
            ]
            approved_candidates.extend(approved_subset_candidates)
        else:
            approved_candidates = candidate_subset

    n_patches_to_yield = min(len(approved_candidates), n_patches_to_yield)

    ### Sort by hematoxylin in descending order
    if sort_h:
        patch_candidates_with_hema = [
            (x[0], get_hematoxilin_component(x[1]))
            for x in approved_candidates[:n_patches_to_yield]
        ]
        patch_candidates = sorted(
            patch_candidates_with_hema, key=lambda x: x[1], reverse=True
        )
        patch_candidates = [x[0] for x in patch_candidates]
    else:
        patch_candidates = [x[0] for x in approved_candidates[:n_patches_to_yield]]

    # assert len(patch_candidates) == n_patches_to_yield, f"{len(patch_candidates)} != {n_patches_to_yield}"

    for i in range(n_patches_to_yield):
        yield (patch_candidates[i][0], patch_candidates[i][1])


def fast_TM_coord_generator(
    slide_path,
    roi_mask=None,
    patch_size=None,
    mask_threshold=0.4,
    s1_polygons=None,
    save_plot=False,
):
    """
    For using with int's of yielded patches

    This is the fastest version of the patch generator, but it uses only tissue mask to do initial filter of candidates.
    This results in higher number of candidates, and if percentage of candidates is used, it may result in higher amount of patches to be yeilded.

    Though it should work perfectly if to_yield is set to a int number.
    """
    assert patch_size is not None, "patch_size is not defined"

    slide_info = extract_slide_info(slide_path)
    vips_image = extract_slide(slide_path, slide_info, level=0)

    num_patches_x = math.ceil(vips_image.width / patch_size)
    num_patches_y = math.ceil(vips_image.height / patch_size)
    n_total_patches = num_patches_x * num_patches_y

    print(f"Total number of patches in WSI: {n_total_patches}")

    ### Mask definition #####################
    mask = roi_mask
    if roi_mask is None and not ignore_tissue_mask:
        mask, _ = generate_tissue_mask(slide_path, save_tissue_contour_image=save_plot)
    #########################################

    ### Generate patch_candidates #####################
    all_coord_pivots = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            if j == num_patches_x - 1 and vips_image.width % patch_size != 0:
                left = vips_image.width - patch_size
            else:
                left = j * patch_size

            if i == num_patches_y - 1 and vips_image.height % patch_size != 0:
                top = vips_image.height - patch_size
            else:
                top = i * patch_size

            width = min(patch_size, vips_image.width - left)
            height = min(patch_size, vips_image.height - top)
            all_coord_pivots.append((left, top, width, height))

    vips_patch_candidates = [
        vips_image.crop(left, top, width, height)
        for left, top, width, height in all_coord_pivots
    ]
    cords_vips_patch_candidates = list(zip(all_coord_pivots, vips_patch_candidates))

    ### Adjust n_total_patches, using mask ############
    if mask is not None:
        cords_vips_patch_candidates = [
            cords_vips_patch_candidates[i]
            for i in range(len(cords_vips_patch_candidates))
            if (is_within_mask(cords_vips_patch_candidates[i][0], mask, mask_threshold))
        ]
    n_patches_to_yield = len(cords_vips_patch_candidates)
    print(f"To y: {n_patches_to_yield}")
    patch_candidates = [x[0] for x in cords_vips_patch_candidates]
    for i in range(n_patches_to_yield):
        yield (patch_candidates[i][0], patch_candidates[i][1])


def normH_patch_coord_generator(
    slide_path,
    roi_mask=None,
    perform_checks=True,
    patch_size=None,
    ignore_tissue_mask=False,
    mask_threshold=0.4,
    to_yield=1.0,
    s1_polygons=None,
    benchmark=True,
    save_plot=False,
    logger=None,
):
    """ """
    assert patch_size is not None, "patch_size is not defined"

    if logger is not None:
        if isinstance(to_yield, float):
            logger.info(
                f"Using 'normH_patch_coord_generator' for to_yield = {to_yield} ({to_yield * 100:0.2f}%). Consider using 'grid_patch_coord_generator' for float ratio."
            )
        elif isinstance(to_yield, int):
            logger.info(
                f"Using 'normH_patch_coord_generator' for to_yield = {to_yield} (int)."
            )

    slide_info = extract_slide_info(slide_path)
    vips_image = extract_slide(slide_path, slide_info, level=0)

    num_patches_x = math.ceil(vips_image.width / patch_size)
    num_patches_y = math.ceil(vips_image.height / patch_size)
    n_total_patches = num_patches_x * num_patches_y

    if logger is not None:
        logger.info(f"Total number of patches in WSI: {n_total_patches}")

    ### Mask definition #####################
    mask = roi_mask
    if roi_mask is None and not ignore_tissue_mask:
        mask, _ = generate_tissue_mask(slide_path, save_tissue_contour_image=save_plot)
    #########################################

    ### Generate patch_candidates #####################
    all_coord_pivots = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            if j == num_patches_x - 1 and vips_image.width % patch_size != 0:
                left = vips_image.width - patch_size
            else:
                left = j * patch_size

            if i == num_patches_y - 1 and vips_image.height % patch_size != 0:
                top = vips_image.height - patch_size
            else:
                top = i * patch_size

            width = min(patch_size, vips_image.width - left)
            height = min(patch_size, vips_image.height - top)
            all_coord_pivots.append((left, top, width, height))

    vips_patch_candidates = [
        vips_image.crop(left, top, width, height)
        for left, top, width, height in all_coord_pivots
    ]
    cords_vips_patch_candidates = list(zip(all_coord_pivots, vips_patch_candidates))

    ### Shuffle patch_candidates

    if benchmark:
        np.random.seed(42)

    np.random.shuffle(cords_vips_patch_candidates)
    ###################################################

    ### Adjust n_total_patches, using mask ############
    if mask is not None:
        cords_vips_patch_candidates = [
            cords_vips_patch_candidates[i]
            for i in range(len(cords_vips_patch_candidates))
            if (is_within_mask(cords_vips_patch_candidates[i][0], mask, mask_threshold))
        ]
        n_total_patches = len(cords_vips_patch_candidates)
    ###################################################

    if isinstance(to_yield, float):
        n_patches_to_yield = int(to_yield * n_total_patches)
    elif isinstance(to_yield, int):
        n_patches_to_yield = min(to_yield, n_total_patches)
    else:
        n_patches_to_yield = n_total_patches

    if logger is not None:
        info_n_patches = (
            f"total WSI patches" if mask is None else "eligible patches, using mask"
        )
        logger.info(
            f"Patches to yield: {n_patches_to_yield} out of {n_total_patches} {info_n_patches}"
        )

    if perform_checks:
        tfs = {
            "from_np.uint8_to_torch.uint8": tv2.Compose(
                [
                    tv2.ToImage(),
                    tv2.ToDtype(torch.uint8, scale=True),
                ]
            ),
            "from_np.uint8_to_torch.float": tv2.Compose(
                [
                    tv2.ToImage(),
                    tv2.ToDtype(torch.float32, scale=True),
                ]
            ),
            "from_tensor.float_to_tensor.uint8": tv2.ToDtype(torch.uint8, scale=True),
        }

        normalizer = NormalizerBuilder.build("reinhard", concentration_method="ls").to(
            "cpu"
        )
        ref_image = cv2.resize(
            cv2.cvtColor(
                cv2.imread(
                    str(
                        Path(
                            "src",
                            "patch_class",
                            "patch_class",
                            "img_ref",
                            "1_ref_img.png",
                        )
                    )
                ),
                cv2.COLOR_BGR2RGB,
            ),
            (patch_size, patch_size),
        )
        normalizer.fit(tfs["from_np.uint8_to_torch.float"](ref_image).unsqueeze(0))
    else:
        normalizer = None
        tfs = None

    approved_candidates = []
    all_patches_idx = 0

    while len(approved_candidates) < n_patches_to_yield and all_patches_idx < len(
        cords_vips_patch_candidates
    ):
        candidate_subset = cords_vips_patch_candidates[
            all_patches_idx : all_patches_idx
            + (n_patches_to_yield - len(approved_candidates))
        ]
        all_patches_idx += n_patches_to_yield

        if perform_checks:
            approved_subset_candidates = [
                candidate_subset[i]
                for i in range(len(candidate_subset))
                if (
                    is_hematoxilin_enough_normed_H(
                        candidate_subset[i][1], normalizer, tfs
                    )
                )
            ]
            approved_candidates.extend(approved_subset_candidates)
        else:
            approved_candidates = candidate_subset

    n_patches_to_yield = min(len(approved_candidates), n_patches_to_yield)

    patch_candidates = [x[0] for x in approved_candidates[:n_patches_to_yield]]

    for i in range(n_patches_to_yield):
        yield (patch_candidates[i][0], patch_candidates[i][1])


def visualize_patches(
    slide_info, pd_pts, save_file_path, only_img=False, include_classes=None
):
    slide_path = slide_info["path"]
    available_levels = sorted(list(slide_info["level_dimensions"].keys()))
    extraction_level = 3 if 3 in available_levels else available_levels[-1]

    try:
        slide_np_check = extract_slide(
            slide_path, slide_info, level=extraction_level
        ).numpy()
        slide_vips = extract_slide(slide_path, slide_info, level=extraction_level)
    except:
        slide_vips = extract_slide(slide_path, slide_info, level=extraction_level - 1)

    # create white background
    bg = pyvips.Image.black(slide_vips.width, slide_vips.height, bands=4)

    cmap = {
        "-1": ("unknown", (255, 255, 0, 255)),
        "0": ("necrosis", (0, 0, 0, 255)),
        "1": ("normal_lung", (0, 255, 255, 255)),
        "2": ("stroma_tls", (0, 127, 25, 255)),
        "3": ("tumor", (255, 0, 0, 255)),
    }

    if not only_img:
        for row in pd_pts.iter_rows():
            scaled_pts = scale_coords(
                eval(str(row[pd_pts.columns.index("Points_pair")])),
                slide_info["raw_size"],
                (slide_vips.width, slide_vips.height),
            )
            left = min(scaled_pts[0][0], scaled_pts[1][0])
            top = min(scaled_pts[0][1], scaled_pts[1][1])
            width = abs(scaled_pts[0][0] - scaled_pts[1][0])
            height = abs(scaled_pts[0][1] - scaled_pts[1][1])
            size = max(width, height)
            if include_classes is not None and "class" in pd_pts.columns:
                if row[pd_pts.columns.index(f"class")] in include_classes:
                    color = cmap[str(row[pd_pts.columns.index("class")])][1]
                    bg = bg.draw_rect(color, left, top, size, size, fill=True)
            else:
                color = cmap["-1"][1]
                bg = bg.draw_rect(color, left, top, size, size, fill=True)

    plt.figure(figsize=(10, 10), dpi=200)

    if include_classes is not None and "class" in pd_pts.columns:
        legend_elements = [
            Patch(
                facecolor=[i / 255 for i in cmap[str(cls)][1]],
                edgecolor="w",
                label=cmap[str(cls)][0],
            )
            for cls in include_classes
        ]
        # legend_elements = [Patch(facecolor=cmap[str(cls)][1], edgecolor='w', label=cmap[str(cls)][0]) for cls in include_classes]
        plt.legend(handles=legend_elements, loc="upper right")

    plt.axis("off")
    plt.tight_layout()
    if not only_img:
        plt.imshow(slide_vips, alpha=0.9)
        plt.imshow(bg, alpha=0.8)
    else:
        plt.imshow(slide_vips)
    plt.savefig(save_file_path)
    plt.close()
