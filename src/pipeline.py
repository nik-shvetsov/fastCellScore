import sys
import os
import time
from glob import glob
from datetime import datetime
from functools import partial
import configparser
from collections import defaultdict
from pathlib import Path
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sympy import Point2D, Segment

import cv2
import pyvips
from PIL import Image
import polars as pl
import matplotlib.pyplot as plt

from wsi_utils import (
    extract_slide,
    extract_slide_info,
    remove_small_objects,
    scale_polygons_coords,
    save_poligons_to_itn,
    get_polypons,
    get_points_pair_from_slide_reduced_polys,
    draw_polygon,
    extract_patch,
    reduce_polygons_points,
    is_row_empty_H,
    calculate_cells_score,
    scale_coords,
    calculate_norm_mm2,
    grid_patch_coord_generator,
    fast_grid_patch_coord_generator,
    grid_H_patch_coord_generator,
    visualize_patches,
    visualize_contours,
)

from roi_segment import config as segment_config
from roi_segment import augs as segment_augs

from patch_class.model import ImgAugmentor
from patch_class.dataset import InferDF_PPTS_PatchDataset, AugmentedDataLoader
from patch_class import config as class_config
from patch_class import augs as class_augs

from hover_quant.datasets.dataset import (
    PatchGeneratorDataset,
    pachified_crops_generator,
)
from hover_quant import config as quant_config
from hover_quant import augs as quant_augs
from hover_quant.utils import post_process_batch_hovernet

from joblib import Parallel, delayed, parallel_backend
from loguru import logger

import warnings

warnings.filterwarnings("ignore")


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time < 60:
            logger.info(
                f"Function {func.__name__} executed in {elapsed_time:0.1f} seconds"
            )
        else:
            minutes, seconds = divmod(elapsed_time, 60)
            logger.info(
                f"Function {func.__name__} executed in {int(minutes)} minutes {seconds:0.1f} seconds"
            )
        return result

    return wrapper


@timer
def s1_roi_segmentation(
    slide_info,
    model,
    params,
    output_dir="/tmp/",
):
    """
    Step 1: run roi_segment model on slide
    """
    use_cached_results = params["USE_CACHE"]
    plot_contours = (
        params["DEBUG_PLOTS"] if params["DEBUG_PLOTS"] is not None else False
    )

    if use_cached_results:
        try:
            config_parser = configparser.ConfigParser()
            config_parser.read(Path(output_dir, f"_S1_{slide_info['name']}.itn"))
            target_polygons = get_polypons(config_parser["Polygon"])
            available_levels = sorted(list(slide_info["level_dimensions"].keys()))
            extraction_level = 3 if 3 in available_levels else available_levels[-1]
            try:
                if plot_contours:
                    visualize_contours(
                        slide_info["path"],
                        slide_info,
                        target_polygons,
                        str(Path(output_dir, f"_S1C_{slide_info['name']}.png")),
                        level=extraction_level,
                        fill=False,
                        color=(0, 255, 0),
                        lt=5,
                    )
            except:
                pass
        except:
            logger.info("Cached results not found. Recomputing...")
            use_cached_results = False

    if not use_cached_results:
        slide_path = slide_info["path"]
        available_levels = sorted(list(slide_info["level_dimensions"].keys()))
        extraction_level = 3 if 3 in available_levels else available_levels[-1]
        try:
            l3_slide = extract_slide(
                slide_path, slide_info, level=extraction_level
            ).numpy()
        except:
            extraction_level = extraction_level - 1
            l3_slide = extract_slide(
                slide_path, slide_info, level=extraction_level
            ).numpy()

        with torch.inference_mode():
            preproc_img = {
                "image": segment_augs.ATF["preproc"](l3_slide),
            }
            preproc_img = segment_augs.ATF[f"resize_to_tensor"](
                image=preproc_img["image"]
            )
            img = preproc_img["image"]
            pred_mask = np.array(
                model.predict(img, device=segment_config.ACCELERATOR), np.uint8
            )

        ## Clear cuda memory
        torch.cuda.empty_cache()

        refined_mask = remove_small_objects(pred_mask, min_obj_size=2000)

        # contours, _ = cv2.findContours(refined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # include internal contours
        contours, _ = cv2.findContours(
            refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        reference_polygons = {
            k: [(point[0][0], point[0][1]) for point in obj]
            for k, obj in enumerate(contours)
        }
        reference_size = pred_mask.shape[0:2]  # size of network output
        target_size = slide_info["raw_size"]
        target_polygons = {}
        target_polygons = scale_polygons_coords(
            reference_polygons, reference_size, target_size
        )
        save_poligons_to_itn(
            target_polygons, str(Path(output_dir, f"_S1_{slide_info['name']}.itn"))
        )
        if plot_contours:
            visualize_contours(
                slide_path,
                slide_info,
                target_polygons,
                str(Path(output_dir, f"_S1C_{slide_info['name']}.png")),
                level=extraction_level,
                fill=False,
                color=(0, 255, 0),
                lt=5,
            )

    return target_polygons


@timer
def s2_extract_patches_border(
    slide_info,
    s1_polygons,
    params,
    check_sparse=True,
    output_dir="/tmp/",
):
    """
    Step 2: extract patches from slide
    """
    use_cached_results = params["USE_CACHE"]

    crop_size = params["CROP_SIZE"](int(slide_info["mag_level"]))
    assert crop_size is not None, "CROP_SIZE is equal to None"

    if use_cached_results:
        try:
            s2_ppts_pd = pl.read_csv(Path(output_dir, f"_S2_{slide_info['name']}.csv"))
        except:
            logger.info("Cached results not found. Recomputing...")
            use_cached_results = False

    if not use_cached_results:
        slide_path = slide_info["path"]
        l0_slide = extract_slide(slide_path, slide_info, level=0)

        if isinstance(s1_polygons, str):
            config_parser = configparser.ConfigParser()
            config_parser.read(s1_polygons)
            l0_polys = get_polypons(config_parser["Polygon"])
        elif isinstance(s1_polygons, dict):
            l0_polys = s1_polygons

        l0_reduced_polys = reduce_polygons_points(
            l0_polys, threshold=crop_size, fix_last_missing=True
        )
        slide_with_point_pair_list = get_points_pair_from_slide_reduced_polys(
            l0_slide, l0_reduced_polys, threshold=crop_size
        )

        s2_ppts_pd = pl.DataFrame(
            {
                "Pair_ID": [x[0] for x in slide_with_point_pair_list],
                "Poly_ID": [x[1] for x in slide_with_point_pair_list],
                "Points_pair": [str(tuple(x[2])) for x in slide_with_point_pair_list],
            }
        )
        s2_ppts_pd = s2_ppts_pd.select(["Pair_ID", "Points_pair", "Poly_ID"])

        def process_row(x):
            try:
                is_H_empty = is_row_empty_H(
                    l0_slide, eval(str(x)), crop_size, threshold=0.017
                )
                return is_H_empty
            except Exception as e:
                return None

        if check_sparse:
            s2_ppts_pd = s2_ppts_pd.with_columns(
                pl.col("Points_pair")
                .map_elements(process_row, return_dtype=pl.Boolean)
                .alias("sparse_patch")
            )
            s2_ppts_pd = s2_ppts_pd.filter(pl.col("sparse_patch") == False).drop(
                "sparse_patch"
            )

        s2_ppts_pd.write_csv(Path(output_dir, f"_S2_{slide_info['name']}.csv"))

    return s2_ppts_pd


@timer
def s2_extract_patches_ROI(
    slide_info,
    s1_polygons,
    patch_generator_func,
    params,
    output_dir="/tmp/",
):
    """
    Step 2: extract patches random patches from ROI, controlled by ratio
    """
    use_cached_results = params["USE_CACHE"]

    crop_size = params["CROP_SIZE"](int(slide_info["mag_level"]))
    assert crop_size is not None, "CROP_SIZE is equal to None"

    if use_cached_results:
        try:
            s2_ppts_pd = pl.read_csv(Path(output_dir, f"_S2_{slide_info['name']}.csv"))
        except:
            logger.info("Cached results not found. Recomputing...")
            use_cached_results = False

    if not use_cached_results:
        slide_path = slide_info["path"]
        l0_slide = extract_slide(slide_path, slide_info, level=0)

        if isinstance(s1_polygons, str):
            config_parser = configparser.ConfigParser()
            config_parser.read(s1_polygons)
            l0_polys = get_polypons(config_parser["Polygon"])
        elif isinstance(s1_polygons, dict):
            l0_polys = s1_polygons

        mask = np.zeros((l0_slide.height, l0_slide.width), dtype=np.uint8)
        for poly in l0_polys.values():
            mask = draw_polygon(mask, poly, fill=True, color=1)
        mask = mask.astype(bool)

        patch_coords = list(
            patch_generator_func(
                slide_path,
                patch_size=crop_size,
                roi_mask=None,
            )
        )

        center_coords = [
            [(x, y + crop_size // 2), (x + crop_size, y + crop_size // 2)]
            for x, y in patch_coords
        ]
        s2_ppts_pd = pl.DataFrame(
            {"Pair_ID": list(range(len(center_coords))), "Points_pair": center_coords}
        )

        if s2_ppts_pd.is_empty():
            biggest_poly_id = max(l0_polys, key=lambda k: len(l0_polys[k]))
            biggest_poly = l0_polys[biggest_poly_id]
            biggest_poly = np.array(biggest_poly, dtype=np.int32).reshape(-1, 1, 2)
            biggest_poly_center = np.mean(biggest_poly, axis=0).astype(np.int32)
            center_coords = [
                (
                    Point2D(
                        biggest_poly_center[0][0] - (crop_size // 2),
                        biggest_poly_center[0][1],
                    ),
                    Point2D(
                        biggest_poly_center[0][0] + (crop_size // 2),
                        biggest_poly_center[0][1],
                    ),
                )
            ]
            s2_ppts_pd = pl.DataFrame(
                {
                    "Pair_ID": list(range(len(center_coords))),
                    "Points_pair": center_coords,
                }
            )

        if not s2_ppts_pd.is_empty():
            pts = [eval(str(x)) for x in s2_ppts_pd["Points_pair"]]
            poly_ids = []

            for pts_pair in pts:
                midpoint = Segment(pts_pair[0], pts_pair[1]).midpoint
                min_distance = np.inf
                closest_poly_id = None

                for poly_id, polygon in l0_polys.items():
                    poly = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                    d = cv2.pointPolygonTest(
                        poly, (int(midpoint.x), int(midpoint.y)), True
                    )
                    if d >= 0:
                        closest_poly_id = poly_id
                        break
                    elif d < 0 and abs(d) < min_distance:
                        min_distance = abs(d)
                        closest_poly_id = poly_id

                poly_ids.append(closest_poly_id if closest_poly_id is not None else -1)

            s2_ppts_pd["Poly_ID"] = poly_ids
            s2_ppts_pd["Points_pair"] = s2_ppts_pd["Points_pair"].astype(str)
            s2_ppts_pd.write_csv(Path(output_dir, f"_S2_{slide_info['name']}.csv"))

    return s2_ppts_pd


@timer
def s2_extract_patches_rnd(
    slide_info,
    patch_generator_func,
    params,
    output_dir="/tmp/",
):
    """
    Step 2: extract random patches from slide
    """
    use_cached_results = params["USE_CACHE"]

    crop_size = params["CROP_SIZE"](int(slide_info["mag_level"]))
    assert crop_size is not None, "CROP_SIZE is equal to None"

    plot_selected_patches = (
        params["DEBUG_PLOTS"] if params["DEBUG_PLOTS"] is not None else False
    )

    if use_cached_results:
        try:
            s2_ppts_pd = pl.read_csv(Path(output_dir, f"_S2_{slide_info['name']}.csv"))
            if plot_selected_patches:
                visualize_patches(
                    slide_info,
                    s2_ppts_pd,
                    str(Path(output_dir, f"_S2P_{slide_info['name']}.png")),
                    only_img=False,
                    include_classes=None,
                )
        except:
            logger.info("Cached results not found. Recomputing...")
            use_cached_results = False

    if not use_cached_results:
        slide_path = slide_info["path"]
        l0_slide = extract_slide(slide_path, slide_info, level=0)
        patch_coords = list(
            patch_generator_func(
                slide_path,
                patch_size=crop_size,
                roi_mask=None,
            )
        )

        center_coords = [
            [(x, y + crop_size // 2), (x + crop_size, y + crop_size // 2)]
            for x, y in patch_coords
        ]
        s2_ppts_pd = pl.DataFrame(
            {
                "Pair_ID": list(range(len(center_coords))),
                "Points_pair": [str(coord) for coord in center_coords],
            }
        )

        s2_ppts_pd = s2_ppts_pd.with_columns(pl.lit(-1).alias("Poly_ID"))

        if plot_selected_patches:
            visualize_patches(
                slide_info,
                s2_ppts_pd,
                str(Path(output_dir, f"_S2P_{slide_info['name']}.png")),
                only_img=False,
                include_classes=None,
            )

        s2_ppts_pd.write_csv(str(Path(output_dir, f"_S2_{slide_info['name']}.csv")))

    return s2_ppts_pd


@timer
def s3_filter_patches(
    slide_info,
    model,
    params,
    s2_ppts_pd=None,
    output_dir="/tmp/",
):
    """
    Step 3: filter patches
    """
    use_cached_results = params["USE_CACHE"]

    crop_size = params["CROP_SIZE"](int(slide_info["mag_level"]))
    assert crop_size is not None, "CROP_SIZE is equal to None"

    patch_height = crop_size
    prob_threshold = (
        {k: v[1] for k, v in params["S3_PROB_THRESHOLD"].items()}
        if params["S3_PROB_THRESHOLD"] is not None
        else None
    )
    plot_selected_patches = (
        params["DEBUG_PLOTS"] if params["DEBUG_PLOTS"] is not None else False
    )

    slide_path = slide_info["path"]
    l0_slide = extract_slide(slide_path, slide_info, level=0)

    if use_cached_results:
        try:
            s3_ppts_pd = pl.read_csv(Path(output_dir, f"_S3_{slide_info['name']}.csv"))
            if plot_selected_patches:
                visualize_patches(
                    slide_info,
                    s3_ppts_pd,
                    str(Path(output_dir, f"_S3P_{slide_info['name']}.png")),
                    only_img=False,
                    include_classes=s3_ppts_pd["class"].unique().to_list(),
                )
        except:
            logger.info("Cached results not found. Recomputing...")
            use_cached_results = False

    if not use_cached_results:
        assert (
            s2_ppts_pd is not None
        ), "s2_ppts_pd must be provided if use_cached_results=False"

        patch_dataset = InferDF_PPTS_PatchDataset(
            s2_ppts_pd,
            l0_slide,
            class_augs.ATF,
            extract_patch_height=patch_height,
        )

        dataloader = AugmentedDataLoader(
            DataLoader(
                patch_dataset,
                batch_size=class_config.INF_BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            ),
            ImgAugmentor(
                class_config.ATF,
                p_augment=False,
                preproc=True,
                norm_args={  # class_config.NORM_ARGS,
                    "method": "macenko",  # ['vahadane', 'macenko', 'reinhard']
                    "concentration_method": "ls",  # ['ls', 'cd', 'ista']
                    "ref_img_name": "1_ref_img.png",
                },
                color_augment_args=None,
                use_fast_color_aug=False,
                clamp_values=False,
                train_mode=False,
                proc_device=class_config.ACCELERATOR,
                target_device=class_config.ACCELERATOR,
            ),
        )

        with torch.inference_mode():
            classifications_list = []

            for batch_idx, (imgs, idxs) in enumerate(dataloader):
                if imgs.type != torch.FloatTensor:
                    imgs = imgs.type(torch.FloatTensor)
                if imgs.device != class_config.ACCELERATOR:
                    imgs = imgs.to(device=class_config.ACCELERATOR)
                logits = model(imgs)

                if prob_threshold is None:
                    (_, pred_idxs) = logits.max(1)
                    idxs = idxs.cpu().numpy()
                    pred_idxs = pred_idxs.cpu().numpy()
                    for idx, pred_idx in zip(idxs, pred_idxs):
                        classifications_list.append((idx, pred_idx))

                else:
                    probs = F.softmax(logits, dim=1)
                    (max_probs, pred_idxs) = probs.max(1)
                    idxs = idxs.cpu().numpy()
                    pred_idxs = pred_idxs.cpu().numpy()
                    max_probs = max_probs.cpu().numpy()
                    for idx, pred_idx, max_prob in zip(idxs, pred_idxs, max_probs):
                        if max_prob < prob_threshold[pred_idx]:
                            pred_idx = -1
                        classifications_list.append((idx, pred_idx))

        classifications_pd = pl.DataFrame(
            {
                "Pair_ID": [item[0] for item in classifications_list],
                "class": [item[1] for item in classifications_list],
            }
        )
        s3_ppts_pd = s2_ppts_pd.join(classifications_pd, on="Pair_ID", how="left")
        s3_ppts_pd = s3_ppts_pd.with_columns(
            pl.col("class").fill_nan(-1).cast(pl.Int32)
        )

        if plot_selected_patches:
            visualize_patches(
                slide_info,
                s3_ppts_pd,
                str(Path(output_dir, f"_S3P_{slide_info['name']}.png")),
                only_img=False,
                include_classes=s3_ppts_pd["class"].unique().to_list(),
            )

        s3_ppts_pd.write_csv(Path(output_dir, f"_S3_{slide_info['name']}.csv"))

        ## Clear cuda memory
        torch.cuda.empty_cache()

    return s3_ppts_pd


@timer
def s4_quant_patches(
    slide_info,
    model,
    params,
    s3_ppts_pd=None,
    small_obj_size_thresh=50,
    output_dir="/tmp/",
):
    """
    Step 4: run quantification model on patches
    Note: small_obj_size_thresh=50 (or even 25) is better than default=10 for 256x256 patches

    """
    use_cached_results = params["USE_CACHE"]

    crop_size = params["CROP_SIZE"](int(slide_info["mag_level"]))
    assert crop_size is not None, "CROP_SIZE is equal to None"
    patch_size = crop_size / 3  # quant_config.INPUT_SIZE[0]

    filter_cls = params["FILTER_CLS"]

    slide_path = slide_info["path"]
    l0_slide = extract_slide(slide_path, slide_info, level=0)

    if use_cached_results:
        try:
            s4_ppts_pd = pl.read_csv(Path(output_dir, f"_S4_{slide_info['name']}.csv"))
        except:
            logger.info("Cached results not found. Recomputing...")
            use_cached_results = False

    if not use_cached_results:
        patch_dataset = PatchGeneratorDataset(
            pachified_crops_generator(
                s3_ppts_pd,
                l0_slide,
                quant_augs.ATF,
                crop_size=crop_size,
                patch_size=patch_size,
                preproc=False,
                augment=False,
                filter_cls=filter_cls,
            )
        )
        dataloader = DataLoader(
            patch_dataset,
            batch_size=quant_config.INF_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        with torch.inference_mode():
            mask_pred = None
            data = []

            for batch_idx, (imgs, idxs) in enumerate(dataloader):
                imgs = imgs.type(torch.FloatTensor).to(quant_config.ACCELERATOR)
                outputs = model(imgs)
                # preds_detection not used if n_classes is defined

                preds_detection, preds_classification, _ = post_process_batch_hovernet(
                    outputs,
                    n_classes=quant_config.NUM_CLASSES,
                    small_obj_size_thresh=50,
                )
                mask_pred = (
                    preds_classification
                    if batch_idx == 0
                    else np.concatenate([mask_pred, preds_classification], axis=0)
                )

                parent_patch_ids = list(idxs[0].cpu().numpy())

                batch_indexes = list(zip(idxs[1].cpu().numpy(), idxs[2].cpu().numpy()))

                for img_idx in range(len(batch_indexes)):
                    channel_counts = []
                    # for channel in range(quant_config.NUM_CLASSES):
                    for channel_k in quant_config.LABELS.keys():
                        # if channel is not quant_config.NUM_CLASSES - 1: # remove only on non-background channels
                        if quant_config.LABELS[channel_k] != "Background":
                            channel_uniques = np.delete(
                                np.unique(preds_classification[img_idx][channel_k]),
                                np.where(
                                    np.unique(preds_classification[img_idx][channel_k])
                                    == 0
                                ),
                            )  # exclude 0 as background
                        else:
                            channel_uniques = []
                            # channel_uniques = np.unique(preds_classification[img_idx][channel_k])
                        channel_counts.append(len(channel_uniques))
                    data.append(
                        [parent_patch_ids[img_idx]]
                        + [batch_indexes[img_idx]]
                        + channel_counts
                    )
        column_names = ["Pair_ID", "Subpatch_ID"] + [
            quant_config.LABELS[k] for k in quant_config.LABELS.keys()
        ]
        transposed_data = list(map(list, zip(*data)))
        s4_ppts_pd = pl.DataFrame(
            {name: values for name, values in zip(column_names, transposed_data)}
        )

        if s4_ppts_pd.is_empty():
            return s4_ppts_pd

        s4_ppts_pd = s4_ppts_pd.map_rows(
            lambda row: tuple(
                str(tuple(item)) if isinstance(item, (list, tuple)) else item
                for item in row
            )
        )
        s4_ppts_pd = s4_ppts_pd.rename(
            {old: new for old, new in zip(s4_ppts_pd.columns, column_names)}
        )

        """
        s4_ppts_pd = pl.DataFrame(
            data,
            columns=["Pair_ID"]
                    + ["Subpatch_ID"]
                    + [quant_config.LABELS[k] for k in quant_config.LABELS.keys()],
        )
        """
        s4_ppts_pd.write_csv(Path(output_dir, f"_S4_{slide_info['name']}.csv"))

        ### Clear cuda memory
        torch.cuda.empty_cache()

    return s4_ppts_pd


@timer
def s5_aggregate(slide_info, params, s4_ppts_pd=None, output_dir="/tmp/"):
    """
    Step 5: aggregate results
    """
    cell_types = params["S5_CELL_TYPES"]

    if s4_ppts_pd is None:
        try:
            s4_ppts_pd = pl.read_csv(Path(output_dir, f"_S4_{slide_info['name']}.csv"))
            if s4_ppts_pd.is_empty():
                return None
        except:
            logger.info(
                "s4_ppts_pd was not provided and cached results were not found..."
            )
            return None

    s5_output = calculate_cells_score(slide_info, cell_types, s4_ppts_pd)
    with open(Path(output_dir, f"_S5T_{slide_info['name']}.txt"), "w") as f:
        f.write(str(s5_output))
    return s5_output


@timer
def s5_ROI_visualize(
    slide_info,
    s1_polygons,
    params,
    poly_id=None,
    s3_ppts_pd=None,
    s4_ppts_pd=None,
    output_dir="/tmp/",
):
    """
    Step 5: visualize ROIs on slide, using averaged score of based_on_type cells in analyzed patches
    """
    based_on_type = params["S5_BASED_ON_TYPE"]

    patch_size = params["CROP_SIZE"](int(slide_info["mag_level"]))
    assert patch_size is not None, "CROP_SIZE is equal to None"

    if s3_ppts_pd is None:
        try:
            s3_ppts_pd = pl.read_csv(Path(output_dir, f"_S3_{slide_info['name']}.csv"))
        except:
            logger.info(
                "s3_ppts_pd was not provided and cached results were not found..."
            )
            return None

    if s4_ppts_pd is None:
        try:
            s4_ppts_pd = pl.read_csv(Path(output_dir, f"_S4_{slide_info['name']}.csv"))
        except:
            logger.info(
                "s4_ppts_pd was not provided and cached results were not found..."
            )
            return None

    if based_on_type not in s4_ppts_pd.columns:
        logger.info(f"Column {based_on_type} not found in s4_ppts_pd")
        return None

    if poly_id is None:
        polys = list(s3_ppts_pd["Poly_ID"].unique())
        if set(polys) == {-1}:
            return None
    else:
        if poly_id not in list(s3_ppts_pd["Poly_ID"].unique()):
            logger.info(f"Poly_ID {poly_id} not found in s3_ppts_pd")
            return None
        polys = list(poly_id)

    combined_ppts = s3_ppts_pd.merge(
        s4_ppts_pd.drop("Subpatch_ID", axis=1).groupby("Pair_ID").sum().reset_index(),
        on="Pair_ID",
        how="right",
    )
    combined_ppts[f"{based_on_type}_norm_mm2"] = calculate_norm_mm2(
        slide_info["mpp"], patch_size, combined_ppts[f"{based_on_type}"]
    )
    # normalize between 0 and 1
    combined_ppts[f"{based_on_type}_norm_score"] = (
        combined_ppts[f"{based_on_type}_norm_mm2"]
        / combined_ppts[f"{based_on_type}_norm_mm2"].max()
    )

    combined_ppts = combined_ppts[["Poly_ID", f"{based_on_type}_norm_score"]]
    combined_ppts = combined_ppts.groupby("Poly_ID").mean().reset_index()

    slide_path = slide_info["path"]
    available_levels = sorted(list(slide_info["level_dimensions"].keys()))
    extraction_level = 3 if 3 in available_levels else available_levels[-1]
    try:
        slide_np = extract_slide(
            slide_path, slide_info, add_alpha=True, level=extraction_level
        ).numpy()
    except:
        slide_np = extract_slide(
            slide_path, slide_info, add_alpha=True, level=extraction_level - 1
        ).numpy()

    scaled_target_polygons = scale_polygons_coords(
        s1_polygons, slide_info["raw_size"], (slide_np.shape[1], slide_np.shape[0])
    )

    cmap = plt.get_cmap("RdYlGn")
    combined_ppts["color"] = combined_ppts.apply(
        lambda x: tuple(int(c * 255) for c in cmap(x[f"{based_on_type}_norm_score"])),
        axis=1,
    )

    plt.figure(dpi=300)
    plt.axis("off")
    img = plt.imshow(slide_np, alpha=0.5)

    for index, row in combined_ppts.iterrows():
        if row.Poly_ID in polys:
            mask = np.zeros(slide_np.shape[0:2], dtype=np.uint8)
            npcoords = np.array([scaled_target_polygons[row.Poly_ID]], np.int32)
            cv2.fillPoly(mask, npcoords, 1)
            mask = mask.astype(bool) * 1
            alpha_channel = np.zeros_like(mask)
            alpha_channel[mask == 1] = 255 * 0.7
            mask_rgba = np.dstack(
                (
                    mask * row.color[0],
                    mask * row.color[1],
                    mask * row.color[2],
                    alpha_channel,
                )
            )
            mask_rgba = mask_rgba.astype(np.uint8)
            img = plt.imshow(mask_rgba, cmap=cmap, vmin=0, vmax=100)

    plt.colorbar(img, orientation="vertical", shrink=0.5)  # adjust size of colorbar

    plt.savefig(
        Path(output_dir, f"_S5R_{slide_info['name']}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.cla()
    plt.clf()
    plt.close("all")
    gc.collect()


@timer
def s5_patch_visualize(
    slide_info,
    params,
    s1_polygons=None,
    poly_id=None,
    s3_ppts_pd=None,
    s4_ppts_pd=None,
    min_max_thresholds=(0.0, 10_000.0),
    output_dir="/tmp/",
):
    """
    Step 5: visualize patches on slide, using based_on_type

    Strict:
    max_threshold_patch = 16_000.0
    min_threshold_patch = 0.0

    Softer:
    max_threshold_patch = 10_000.0
    """

    based_on_type = params["S5_CELL_TYPE_PLOT"]

    combined_patch_size = params["CROP_SIZE"](int(slide_info["mag_level"]))
    assert combined_patch_size is not None, "CROP_SIZE is equal to None"

    if s3_ppts_pd is None:
        try:
            s3_ppts_pd = pl.read_csv(Path(output_dir, f"_S3_{slide_info['name']}.csv"))
        except:
            logger.info(
                "s3_ppts_pd was not provided and cached results were not found..."
            )
            return None

    if s4_ppts_pd is None:
        try:
            s4_ppts_pd = pl.read_csv(Path(output_dir, f"_S4_{slide_info['name']}.csv"))
        except:
            logger.info(
                "s4_ppts_pd was not provided and cached results were not found..."
            )
            return None

    if based_on_type not in s4_ppts_pd.columns:
        logger.info(f"Column {based_on_type} not found in s4_ppts_pd")
        return None

    if poly_id is None:
        polys = s3_ppts_pd["Poly_ID"].unique().to_list()
    else:
        if poly_id not in s3_ppts_pd["Poly_ID"].unique().to_list():
            logger.info(f"Poly_ID {poly_id} not found in s3_ppts_pd")
            return None
        polys = list(poly_id)

    s4_ppts_pd = s4_ppts_pd.with_columns(s4_ppts_pd["Pair_ID"].cast(pl.datatypes.Int64))
    combined_ppts = (
        s4_ppts_pd.drop("Subpatch_ID")
        .group_by("Pair_ID")
        .agg(pl.col("*").sum())
        .with_columns(pl.col("Pair_ID"))
        .join(
            s3_ppts_pd,
            on="Pair_ID",
            how="left",
        )
    )

    combined_ppts = combined_ppts.with_columns(
        pl.col(based_on_type)
        .map_elements(
            lambda value: calculate_norm_mm2(
                slide_info["mpp"], combined_patch_size, int(value)
            ),
            return_dtype=pl.Float64,
        )
        .alias(f"{based_on_type}_patch_norm_score")
    )

    combined_ppts = combined_ppts.with_columns(
        (pl.col(f"{based_on_type}_patch_norm_score") / min_max_thresholds[1])
        .clip(0, 1)
        .alias(f"{based_on_type}_patch_norm_score_cliped")
    )

    slide_path = slide_info["path"]
    available_levels = sorted(list(slide_info["level_dimensions"].keys()))
    extraction_level = 3 if 3 in available_levels else available_levels[-1]

    # slide_vips = extract_slide(slide_path, slide_info, level=extraction_level)
    try:
        slide_np_check = extract_slide(
            slide_path, slide_info, level=extraction_level
        ).numpy()
        slide_vips = extract_slide(slide_path, slide_info, level=extraction_level)
    except:
        slide_vips = extract_slide(slide_path, slide_info, level=extraction_level - 1)

    bg = pyvips.Image.black(slide_vips.width, slide_vips.height, bands=4)
    cmap = plt.get_cmap("RdYlGn")

    if s1_polygons is not None:
        bg_np = bg.numpy()
        scaled_target_polygons = scale_polygons_coords(
            s1_polygons, slide_info["raw_size"], (slide_vips.width, slide_vips.height)
        )
        for poly_id, polygon in scaled_target_polygons.items():
            npcoords = np.array([scaled_target_polygons[poly_id]], np.int32)
            cv2.polylines(bg_np, npcoords, True, (0, 255, 0, 255), 10)
        bg = pyvips.Image.new_from_array(bg_np)

    for row in combined_ppts.iter_rows():
        if row[combined_ppts.columns.index("Poly_ID")] in polys:
            scaled_pts = scale_coords(
                eval(str(row[combined_ppts.columns.index("Points_pair")])),
                slide_info["raw_size"],
                (slide_vips.width, slide_vips.height),
            )
            left = min(scaled_pts[0][0], scaled_pts[1][0])
            top = min(scaled_pts[0][1], scaled_pts[1][1])
            width = abs(scaled_pts[0][0] - scaled_pts[1][0])
            height = abs(scaled_pts[0][1] - scaled_pts[1][1])
            size = max(width, height)
            color = tuple(
                int(c * 255)
                for c in cmap(
                    row[
                        combined_ppts.columns.index(
                            f"{based_on_type}_patch_norm_score_cliped"
                        )
                    ]
                )
            )
            bg = bg.draw_rect(color, left, top, size, size, fill=True)

    plt.figure(figsize=(12, 10), dpi=300)
    plt.axis("off")
    plt.imshow(slide_vips, alpha=0.5)
    img = plt.imshow(
        bg, cmap=cmap, vmin=0, vmax=min_max_thresholds[1]
    )  # set data range for colormap
    plt.colorbar(img, orientation="vertical", shrink=0.5)  # adjust size of colorbar
    plt.savefig(
        Path(output_dir, f"_S5P_{slide_info['name']}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.cla()
    plt.clf()
    plt.close("all")
    gc.collect()


def validate_params(params):
    assert (
        params["WSI_GLOB"] is not None and len(list(glob(params["WSI_GLOB"]))) > 0
    ), f"Invalid params['WSI_GLOB']: {params['WSI_GLOB']}"
    assert params["ITN_DIR"] is None or (
        Path(params["ITN_DIR"]).exists()
        and len(list(Path(params["ITN_DIR"]).glob("*.itn")))
    ), f"Invalid params['ITN_DIR']: {params['ITN_DIR']}"
    assert params["PTYPE"] in [
        "rnd",
        "ROI",
        "border",
    ], f"Invalid params['PTYPE']: {params['PTYPE']}"
    assert (
        params["N_RATIO"] is None
        or (isinstance(params["N_RATIO"], int) and params["N_RATIO"] > 0)
        or (isinstance(params["N_RATIO"], float) and 0.0 < params["N_RATIO"] <= 1.0)
    ), f"Invalid params['N_RATIO']: {params['N_RATIO']}"
    assert (
        params["CROP_SIZE_MP"] > 0
    ), f"Invalid params['CROP_SIZE_MP']: {params['CROP_SIZE_MP']}"
    assert params["FILTER_CLS"] is None or (
        isinstance(params["FILTER_CLS"], tuple) and len(params["FILTER_CLS"]) > 0
    ), f"Invalid params['FILTER_CLS']: {params['FILTER_CLS']}"
    assert params["S3_PROB_THRESHOLD"] is None or (
        isinstance(params["S3_PROB_THRESHOLD"], dict)
        and len(params["S3_PROB_THRESHOLD"]) > 0
    ), f"Invalid params['S3_PROB_THRESHOLD']: {params['S3_PROB_THRESHOLD']}"
    assert (
        params["S5_CELL_TYPES"] is not None
        and isinstance(params["S5_CELL_TYPES"], tuple)
        and len(params["S5_CELL_TYPES"]) > 0
    ), f"Invalid params['S5_CELL_TYPES']: {params['S5_CELL_TYPES']}"
    assert (
        params["S5_CELL_TYPE_PLOT"] in params["S5_CELL_TYPES"]
    ), f"Invalid params['S5_CELL_TYPE_PLOT']: {params['S5_CELL_TYPE_PLOT']}"
    assert params["DEBUG_PLOTS"] in [
        True,
        False,
    ], f"Invalid params['DEBUG_PLOTS']: {params['DEBUG_PLOTS']}"
    assert params["USE_CACHE"] in [
        True,
        False,
    ], f"Invalid params['USE_CACHE']: {params['USE_CACHE']}"
    assert params["DETERMINISTIC"] in [
        True,
        False,
    ], f"Invalid params['DETERMINISTIC']: {params['DETERMINISTIC']}"
    assert params["SORT_H_LOW_THRESHOLD"] is None or (
        isinstance(params["SORT_H_LOW_THRESHOLD"], float)
        and 0.0 < params["SORT_H_LOW_THRESHOLD"] < 1.0
    ), f"Invalid params['SORT_H_LOW_THRESHOLD']: {params['SORT_H_LOW_THRESHOLD']}"


def process_slide(
    slide_num,
    slide_path,
    init_models_dict,
    patch_generator_func,
    result_dir,
    proc_params,
):

    assert proc_params["PTYPE"] in [
        "rnd",
        "ROI",
        "border",
    ], f"Invalid proc_params['PTYPE']: {proc_params['PTYPE']}"

    try:
        slide_info = extract_slide_info(slide_path)
    except Exception as e:
        logger.error(f"Unable to process slide: {slide_path}")
        logger.error(e)
        return None

    logger.info(f"==============================================================")
    logger.info(f"-------Processing slide: ({slide_num}) {slide_info['name']}-------")

    # slide_id_name | slide_id | slide_num
    slide_output_dir = Path(
        result_dir, f"{str(slide_info['name'])}_{slide_info['id']}_{slide_num}"
    )

    if not Path(slide_output_dir).exists():
        Path(slide_output_dir).mkdir(parents=True)

    if proc_params["PTYPE"] in ["ROI", "border"]:
        # Step 1 : run roi_segment model on slide
        logger.info(
            f"({slide_num}) Step 1 aka ROI segmentation uses: {segment_config.PROFILE_ID} profile"
        )
        s1_polygons = s1_roi_segmentation(
            slide_info,
            init_models_dict["S1"],
            proc_params,
            output_dir=slide_output_dir,
        )
    else:
        s1_polygons = {}

    # Step 2: extract patches from slide
    if s1_polygons == {} and proc_params["PTYPE"] != "rnd":  # empty slide
        return (slide_num, slide_info["name"])

    logger.info(
        f"({slide_num}) Step 2 aka candidates patch extraction uses <{proc_params['PTYPE']} extraction> method"
    )

    if proc_params["PTYPE"] == "ROI":
        s2_ppts_pd = s2_extract_patches_ROI(
            slide_info,
            s1_polygons,
            patch_generator_func,
            proc_params,
            output_dir=slide_output_dir,
        )
    elif proc_params["PTYPE"] == "rnd":
        s2_ppts_pd = s2_extract_patches_rnd(
            slide_info,
            patch_generator_func,
            proc_params,
            output_dir=slide_output_dir,
        )
    elif proc_params["PTYPE"] == "border":
        s2_ppts_pd = s2_extract_patches_border(
            slide_info,
            s1_polygons,
            proc_params,
            check_sparse=True,
            output_dir=slide_output_dir,
        )

    if s2_ppts_pd.is_empty():
        return (slide_num, slide_info["name"])

    # Step 3: filter patches
    logger.info(
        f"({slide_num}) Step 3 aka Patch classification uses: {class_config.PROFILE_ID} profile"
    )
    s3_ppts_pd = s3_filter_patches(
        slide_info,
        init_models_dict["S3"],
        proc_params,
        s2_ppts_pd=s2_ppts_pd,
        output_dir=slide_output_dir,
    )

    # Step 4: run quantification model on patches
    logger.info(
        f"({slide_num}) Step 4 aka Patch quantification uses: {quant_config.PROFILE_ID} profile"
    )
    s4_ppts_pd = s4_quant_patches(
        slide_info,
        init_models_dict["S4"],
        proc_params,
        s3_ppts_pd=s3_ppts_pd,
        output_dir=slide_output_dir,
    )

    if s4_ppts_pd.is_empty():
        logger.info(f"({slide_num}) No patches found for quantification")
        return (slide_num, slide_info["name"])

    # # Step 5: aggregate results and visualize on slide
    logger.info(f"({slide_num}) Step 5 aka Results for {slide_info['name']}:")

    result_S5 = s5_aggregate(
        slide_info,
        proc_params,
        s4_ppts_pd=None,
        output_dir=slide_output_dir,
    )

    if result_S5 is None:
        return (slide_num, slide_info["name"])

    s5_patch_visualize(
        slide_info,
        proc_params,
        s1_polygons=None,
        poly_id=None,
        s3_ppts_pd=None,
        s4_ppts_pd=None,
        min_max_thresholds=(0.0, 10_000.0),
        output_dir=slide_output_dir,
    )
    return None


if __name__ == "__main__":
    DEBUG_PARAMS = {
        "DEBUG": False,
        "START_IDX": 0,
        "NUM_SLIDES": 1,
    }

    params = defaultdict(lambda: None)
    params.update(
        {
            "WSI_GLOB": "/mnt/data/*.svs",
            "N_RATIO": 0.1,
            "OUT_LABEL": "LABEL",
            "PTYPE": "rnd",
            "CROP_SIZE_MP": 2,
            "FILTER_CLS": (2, 3),
            "S5_CELL_TYPES": ("Inflammatory", "Epithelial"),
            "S5_CELL_TYPE_PLOT": "Inflammatory",
            "DEBUG_PLOTS": True,
            "USE_CACHE": True,
            "DETERMINISTIC": True,
        }
    )

    params.update(
        {
            "CROP_SIZE": lambda x: (
                256 + (256 * params["CROP_SIZE_MP"])
                if x == 40
                else (128 + (128 * params["CROP_SIZE_MP"]) if x == 20 else None)
            ),
            # 'S3_PROB_THRESHOLD': {
            #     0: ("necrosis", 0.7),
            #     1: ("normal_lung", 0.4),
            #     2: ("stroma_tls", 0.6),
            #     3: ("tumor", 0.5)
            # },
            # 'SORT_H_LOW_THRESHOLD': 0.03,
            # 'ITN_DIR': segment_config.ITN_DIR
        }
    )

    validate_params(params)

    out_dir = Path(
        "outputs", f"{params['N_RATIO']}_{params['OUT_LABEL']}_{params['PTYPE']}"
    )
    log_dir = Path(out_dir, "logs")
    result_dir = Path(out_dir, "results")

    ### Best models ###
    models_paths = {
        "S1": max(
            Path("src", "roi_segment", "roi_segment", "saved_models").glob(
                f"{segment_config.PROFILE_ID}_*.pth"
            ),
            key=lambda x: float(x.stem.split("_")[-1]),
        ),
        "S3": max(
            Path("src", "patch_class", "patch_class", "saved_models").glob(
                f"{class_config.PROFILE_ID}_*.pth"
            ),
            key=lambda x: float(x.stem.split("_")[-1]),
        ),
        "S4": max(
            Path("src", "hover_quant", "hover_quant", "saved_models").glob(
                f"{quant_config.PROFILE_ID}_*.pth"
            ),
            key=lambda x: float(x.stem.split("_")[-1]),
        ),
    }

    ### Specific models ###
    # models_paths = {
    #     "S1": Path("src", "roi_segment", "roi_segment", "saved_models", "S1_model.pth"),
    #     "S3": Path("src", "patch_class", "patch_class", "saved_models", "S3_model.pth"),
    #     "S4": Path("src", "hover_quant", "hover_quant", "saved_models", "S4_model.pth"),
    # }

    init_models = {
        "S1": torch.load(models_paths["S1"]).to(segment_config.ACCELERATOR).eval(),
        "S3": torch.load(models_paths["S3"]).to(class_config.ACCELERATOR).eval(),
        "S4": torch.load(models_paths["S4"]).to(quant_config.ACCELERATOR).eval(),
    }

    patch_generator_function = partial(
        fast_grid_patch_coord_generator,
        perform_checks=True,
        ignore_tissue_mask=False,
        mask_threshold=0.4,
        to_yield=params["N_RATIO"],
        benchmark=params["DETERMINISTIC"],
        save_plot=params["DEBUG_PLOTS"],
        logger=logger,
    )

    if not result_dir.exists():
        result_dir.mkdir(parents=True)
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)

    logger.remove(0)
    logger.add(
        Path(
            log_dir,
            f'{result_dir.parent.stem}~{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        ),
        format="{message}",
        mode="a",
    )
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | <green>{level}</green> | {message}",
    )

    assert result_dir.exists(), f"Output directory {result_dir} does not exist"

    logger.info(f"Params: {dict(params)}")

    logger.info(f"Using S1 model: {Path(models_paths['S1']).stem}")
    logger.info(f"Using S3 model: {Path(models_paths['S3']).stem}")
    logger.info(f"Using S4 model: {Path(models_paths['S4']).stem}")

    aperio_slides_fp = sorted(glob(params["WSI_GLOB"]))
    aperio_slides = [os.path.basename(aps) for aps in aperio_slides_fp]

    itn_path = params["ITN_DIR"]
    if itn_path is not None:
        itn_fp = sorted(
            [os.path.join(itn_path, ap.replace(".svs", ".itn")) for ap in aperio_slides]
        )
    else:
        itn_fp = aperio_slides_fp

    result_fp = {
        i: (aperio_slides_fp[i], itn_fp[i]) for i in range(len(aperio_slides_fp))
    }

    if DEBUG_PARAMS["DEBUG"]:
        result_fp = {
            i: (aperio_slides_fp[i], itn_fp[i])
            for i in range(
                DEBUG_PARAMS["START_IDX"],
                DEBUG_PARAMS["START_IDX"] + DEBUG_PARAMS["NUM_SLIDES"],
            )
        }
    else:
        result_fp = {
            i: (aperio_slides_fp[i], itn_fp[i]) for i in range(len(aperio_slides_fp))
        }

    slide_params = [
        (
            slide_num,
            slide_path,
            init_models,
            patch_generator_function,
            result_dir,
            params,
        )
        for slide_num, (slide_path, itn_path) in result_fp.items()
    ]

    with parallel_backend("loky", n_jobs=1):
        results = Parallel()(delayed(process_slide)(*params) for params in slide_params)

    logger.info("--- Finished processing all slides ---")
    logger.info(f"Problematic slides: {[x for x in results if x is not None]}")
