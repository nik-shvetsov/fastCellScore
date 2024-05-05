import numpy as np
from scipy.ndimage import label
import torch
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.detection import PanopticQuality


def dice_score(pred, truth, eps=1e-3):
    """
    Calculate dice score for two tensors of the same shape.
    If tensors are not already binary, they are converted to bool by zero/non-zero.

    Args:
        pred (np.ndarray): Predictions
        truth (np.ndarray): ground truth
        eps (float, optional): Constant used for numerical stability to avoid divide-by-zero errors. Defaults to 1e-3.

    Returns:
        float: Dice score
    """
    assert isinstance(truth, np.ndarray) and isinstance(
        pred, np.ndarray
    ), f"pred is of type {type(pred)} and truth is type {type(truth)}. Both must be np.ndarray"
    assert (
        pred.shape == truth.shape
    ), f"pred shape {pred.shape} does not match truth shape {truth.shape}"
    # turn into binary if not already
    pred = pred != 0
    truth = truth != 0

    num = 2 * np.sum(pred.flatten() * truth.flatten())
    denom = np.sum(pred) + np.sum(truth) + eps
    return float(num / denom)


def jaccard_score(
    preds_classification, masks, num_classes, avg="micro", ignore_index=None
):
    multiclass_jaccard_index = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=ignore_index,
        average=avg,
        validate_args=False,
    )
    return multiclass_jaccard_index(preds_classification, masks)


def pq_score(preds, masks, num_classes, relabel=True):
    panoptic_quality = PanopticQuality(
        # things=set(list(range(num_classes))),
        # stuffs=set([num_classes])
        things=set(list(range(1, num_classes + 1))),
        stuffs=set([0]),
    )
    B, C, W, H = masks.shape

    def configure_maps(maps):
        B, C, H, W = maps.shape
        max_values, argmax_values = maps.max(
            dim=1
        )  # max and argmax over the channel dimension

        # Construct updated maps
        # If there is no instance, the category_id is 0
        category_ids = torch.where(max_values == 0, 0, argmax_values)
        instance_ids = max_values

        upd_maps = torch.stack(
            (category_ids, instance_ids), dim=3
        )  # stack along a new dimension

        if relabel:
            for b in range(B):
                labels, num_features = label(upd_maps[b, :, :, 1].numpy())
                unique_labels = np.unique(labels)
                for i, unique_label in enumerate(unique_labels):
                    upd_maps[b, :, :, 1][labels == unique_label] = i

        return upd_maps

    upd_masks = configure_maps(masks)
    upd_preds = configure_maps(preds)
    return panoptic_quality(upd_preds, upd_masks)


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # B x 1 x H x W => B x H x W

    intersection = (
        (outputs & labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (
        union + SMOOTH
    )  # We smooth our devision to avoid 0/0

    thresholded = (
        torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch
