import torch
from torch import nn
from torch.nn import functional as F
from hover_quant.utils import _convert_multiclass_mask_to_binary, _get_gradient_hv


def dice_loss(true, logits, eps=1e-3):
    """
    Computes the Sørensen-Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return 1 - dice loss.
    From: https://github.com/kevinzakka/pytorch-goodies/blob/c039691f349be9f21527bb38b907a940bfc5e8f3/losses.py#L54

    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen-Dice loss.
    """
    assert (
        true.dtype == torch.long
    ), f"Input 'true' is of type {true.type}. It should be a long."
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1).to(true.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        assert (
            true.max() < num_classes
        ), f"Max target value {true.max()} is not less than num_classes {num_classes}"
        true_1_hot = torch.eye(num_classes).to(true.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    loss = (2.0 * intersection / (cardinality + eps)).mean()
    loss = 1 - loss
    return loss


def _dice_loss_np_head(np_out, true_mask, epsilon=1e-3):
    """
    Dice loss term for nuclear pixel branch.
    This will compute dice loss for the entire batch
    (not the same as computing dice loss for each image and then averaging!)

    Args:
        np_out: logit outputs of np branch. Tensor of shape (B, 2, H, W)
        true_mask: True mask. Tensor of shape (B, n_classes, H, W)
        epsilon (float): Epsilon passed to ``dice_loss()``
    """
    # get logits for only the channel corresponding to prediction of 1
    # unsqueeze to keep the dimensions the same
    preds = np_out[:, 1, :, :].unsqueeze(dim=1)

    true_mask = _convert_multiclass_mask_to_binary(true_mask)
    true_mask = true_mask.type(torch.long)
    loss = dice_loss(logits=preds, true=true_mask, eps=epsilon)
    return loss


def _dice_loss_nc_head(nc_out, true_mask, epsilon=1e-3):
    """
    Dice loss term for nuclear classification branch.
    Computes dice loss for each channel, and sums up.
    This will compute dice loss for the entire batch
    (not the same as computing dice loss for each image and then averaging!)

    Args:
        nc_out: logit outputs of nc branch. Tensor of shape (B, n_classes, H, W)
        true_mask: True mask. Tensor of shape (B, n_classes, H, W)
        epsilon (float): Epsilon passed to ``dice_loss()``
    """
    truth = torch.argmax(true_mask, dim=1, keepdim=True).type(torch.long)
    loss = dice_loss(logits=nc_out, true=truth, eps=epsilon)
    return loss


def _ce_loss_nc_head(nc_out, true_mask):
    """
    Cross-entropy loss term for nc branch.
    Args:
        nc_out: logit outputs of nc branch. Tensor of shape (B, n_classes, H, W)
        true_mask: True mask. Tensor of shape (B, n_classes, H, W)
    """
    truth = torch.argmax(true_mask, dim=1).type(torch.long)
    ce = nn.CrossEntropyLoss()
    loss = ce(nc_out, truth)
    return loss


def _ce_loss_np_head(np_out, true_mask):
    """
    Cross-entropy loss term for np branch.
    Args:
        np_out: logit outputs of np branch. Tensor of shape (B, 2, H, W)
        true_mask: True mask. Tensor of shape (B, n_classes, H, W)
    """
    truth = (
        _convert_multiclass_mask_to_binary(true_mask).type(torch.long).squeeze(dim=1)
    )
    ce = nn.CrossEntropyLoss()
    loss = ce(np_out, truth)
    return loss


def _loss_hv_grad(hv_out, true_hv, nucleus_pixel_mask):
    """
    Equation 3 from HoVer-Net paper for calculating loss for HV predictions.
    Mask is used to compute the hv loss ONLY for nuclear pixels

    Args:
        hv_out: Ouput of hv branch. Tensor of shape (B, 2, H, W)
        true_hv: Ground truth hv maps. Tensor of shape (B, 2, H, W)
        nucleus_pixel_mask: Boolean mask indicating nuclear pixels. Tensor of shape (B, H, W)
    """
    pred_grad_h, pred_grad_v = _get_gradient_hv(hv_out)
    true_grad_h, true_grad_v = _get_gradient_hv(true_hv)

    # pull out only the values from nuclear pixels
    pred_h = torch.masked_select(pred_grad_h, mask=nucleus_pixel_mask)
    true_h = torch.masked_select(true_grad_h, mask=nucleus_pixel_mask)
    pred_v = torch.masked_select(pred_grad_v, mask=nucleus_pixel_mask)
    true_v = torch.masked_select(true_grad_v, mask=nucleus_pixel_mask)

    loss_h = F.mse_loss(pred_h, true_h)
    loss_v = F.mse_loss(pred_v, true_v)

    loss = loss_h + loss_v
    return loss


def _loss_hv_mse(hv_out, true_hv):
    """
    Equation 2 from HoVer-Net paper for calculating loss for HV predictions.

    Args:
        hv_out: Ouput of hv branch. Tensor of shape (B, 2, H, W)
        true_hv: Ground truth hv maps. Tensor of shape (B, 2, H, W)
    """
    loss = F.mse_loss(hv_out, true_hv)
    return loss


def loss_hovernet(outputs, ground_truth, n_classes=None):
    """
    Compute loss for HoVer-Net.
    Equation (1) in Graham et al.

    Args:
        outputs: Output of HoVer-Net. Should be a list of [np, hv] if n_classes is None, or a list of [np, hv, nc] if
            n_classes is not None.
            Shapes of each should be:

                - np: (B, 2, H, W)
                - hv: (B, 2, H, W)
                - nc: (B, n_classes, H, W)

        ground_truth: True labels. Should be a list of [mask, hv], where mask is a Tensor of shape (B, 1, H, W)
            if n_classes is ``None`` or (B, n_classes, H, W) if n_classes is not ``None``.
            hv is a tensor of precomputed horizontal and vertical distances
            of nuclear pixels to their corresponding centers of mass, and is of shape (B, 2, H, W).
        n_classes (int): Number of classes for classification task. If ``None`` then the classification branch is not
            used.

    References:
        Graham, S., Vu, Q.D., Raza, S.E.A., Azam, A., Tsang, Y.W., Kwak, J.T. and Rajpoot, N., 2019.
        Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.
        Medical Image Analysis, 58, p.101563.
    """
    true_mask, true_hv = ground_truth
    # unpack outputs, and also calculate nucleus masks
    if n_classes is None:
        np_out, hv = outputs
        nucleus_mask = true_mask[:, 0, :, :] == 1
    else:
        np_out, hv, nc = outputs
        # in multiclass setting, last channel of masks indicates background, so
        # invert that to get a nucleus mask (Based on convention from PanNuke dataset)
        nucleus_mask = true_mask[:, -1, :, :] == 0

    # from Eq. 1 in HoVer-Net paper, loss function is composed of two terms for each branch.
    np_loss_dice = _dice_loss_np_head(np_out, true_mask)
    np_loss_ce = _ce_loss_np_head(np_out, true_mask)

    hv_loss_grad = _loss_hv_grad(hv, true_hv, nucleus_mask)
    hv_loss_mse = _loss_hv_mse(hv, true_hv)

    # authors suggest using coefficient of 2 for hv gradient loss term
    hv_loss_grad = 2 * hv_loss_grad

    if n_classes is not None:
        nc_loss_dice = _dice_loss_nc_head(nc, true_mask)
        nc_loss_ce = _ce_loss_nc_head(nc, true_mask)
    else:
        nc_loss_dice = 0
        nc_loss_ce = 0

    loss = (
        np_loss_dice
        + np_loss_ce
        + hv_loss_mse
        + hv_loss_grad
        + nc_loss_dice
        + nc_loss_ce
    )
    return loss
