import torch
from torch.nn import functional as F
from torchinfo import summary
import numpy as np
import cv2
import scipy
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import lovely_tensors as lt
from pprint import pprint
import json
from datetime import datetime
from pathlib import Path

lt.monkey_patch()
lt.set_config(sci_mode=False)
torch.set_printoptions(sci_mode=False)


def print_gpu_utilization(device_idx=0):
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Device: {torch.device(f'cuda:{device_idx}')}")

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_idx)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024**2} MB.")


def test_net(model, device="cpu", size=(3, 224, 224), n_batch=32, use_lt=False):
    model = model.to(device)
    model.eval()
    x = torch.randn(n_batch, *size, device=device)
    with torch.no_grad():
        output = model(x)
    if use_lt:
        print(
            "=========================================================================================="
        )
        print(f"Input shape: {x}")
        print(f"Output shape: {output}")
    else:
        print(
            "=========================================================================================="
        )
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    summary(model, input_size=(n_batch, *(size)), device=device)


def save_run_config(cfg, fname):
    consts = {}
    for k in dir(cfg):
        if k.isupper() and not k.startswith("_"):
            consts[k] = str(getattr(cfg, k))
    with open(f"{fname}.conf", "w") as f:
        f.write(json.dumps(obj=consts, indent=4))


def inspect_model(model, output="params"):
    """
    output: 'params' or 'state'
    """
    if output == "state":
        pprint(model.state_dict)
    elif output == "params":
        for idx, (name, param) in enumerate(model.named_parameters()):
            print(f"{idx}: {name} \n{param}")
            print(
                "------------------------------------------------------------------------------------------"
            )
    else:
        raise ValueError("Output must be either 'params' or 'state'")


def segmentation_lines(mask_in):
    """
    Generate coords of points bordering segmentations from a given mask.
    Useful for plotting results of tissue detection or other segmentation.
    """
    assert (
        mask_in.dtype == np.uint8
    ), f"Input mask dtype {mask_in.dtype} must be np.uint8"
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_in, kernel)
    diff = np.logical_xor(dilated.astype(bool), mask_in.astype(bool))
    y, x = np.nonzero(diff)
    return x, y


def plot_segmentation(ax, masks, palette=None, markersize=5):
    """
    Plot segmentation contours. Supports multi-class masks.

    Args:
        ax: matplotlib axis
        masks (np.ndarray): Mask array of shape (n_masks, H, W). Zeroes are background pixels.
        palette: color palette to use. if None, defaults to matplotlib.colors.TABLEAU_COLORS
        markersize (int): Size of markers used on plot. Defaults to 5
    """
    assert masks.ndim == 3
    n_channels = masks.shape[0]

    if palette is None:
        palette = list(TABLEAU_COLORS.values())

    nucleus_labels = list(np.unique(masks))
    if 0 in nucleus_labels:
        nucleus_labels.remove(0)  # background
    # plot each individual nucleus
    for label in nucleus_labels:
        for i in range(n_channels):
            nuclei_mask = masks[i, ...] == label
            x, y = segmentation_lines(nuclei_mask.astype(np.uint8))
            ax.scatter(x, y, color=palette[i], marker=".", s=markersize)


def center_crop_im_batch(batch, dims, batch_order="BCHW"):
    """
    Center crop images in a batch.

    Args:
        batch: The batch of images to be cropped
        dims: Amount to be cropped (tuple for H, W)
    """
    assert (
        batch.ndim == 4
    ), f"ERROR input shape is {batch.shape} - expecting a batch with 4 dimensions total"
    assert (
        len(dims) == 2
    ), f"ERROR input cropping dims is {dims} - expecting a tuple with 2 elements total"
    assert batch_order in {
        "BHCW",
        "BCHW",
    }, f"ERROR input batch order {batch_order} not recognized. Must be one of 'BHCW' or 'BCHW'"

    if dims == (0, 0):
        # no cropping necessary in this case
        batch_cropped = batch
    else:
        crop_t = dims[0] // 2
        crop_b = dims[0] - crop_t
        crop_l = dims[1] // 2
        crop_r = dims[1] - crop_l

        if batch_order == "BHWC":
            batch_cropped = batch[:, crop_t:-crop_b, crop_l:-crop_r, :]
        elif batch_order == "BCHW":
            batch_cropped = batch[:, :, crop_t:-crop_b, crop_l:-crop_r]
        else:
            raise Exception("Input batch order not valid")

    return batch_cropped


def get_sobel_kernels(size, dt=torch.float32):
    """
    Create horizontal and vertical Sobel kernels for approximating gradients
    Returned kernels will be of shape (size, size)
    """
    assert size % 2 == 1, "Size must be odd"

    h_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=dt)
    v_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=dt)
    h, v = torch.meshgrid([h_range, v_range], indexing="ij")
    h, v = h.transpose(0, 1), v.transpose(0, 1)

    kernel_h = h / (h * h + v * v + 1e-5)
    kernel_v = v / (h * h + v * v + 1e-5)

    kernel_h = kernel_h.type(dt)
    kernel_v = kernel_v.type(dt)

    return kernel_h, kernel_v


def _convert_multiclass_mask_to_binary(mask):
    """
    Input mask of shape (B, n_classes, H, W) is converted to a mask of shape (B, 1, H, W).
    The last channel is assumed to be background, so the binary mask is computed by taking its inverse.
    """
    m = torch.tensor(1) - mask[:, -1, :, :]
    m = m.unsqueeze(dim=1)
    return m


def _get_gradient_hv(hv_batch, kernel_size=5):
    """
    Calculate the horizontal partial differentiation for horizontal channel
    and the vertical partial differentiation for vertical channel.
    The partial differentiation is approximated by calculating the central differnce
    which is obtained by using Sobel kernel of size 5x5. The boundary is zero-padded
    when channel is convolved with the Sobel kernel.

    Args:
        hv_batch: tensor of shape (B, 2, H, W). Channel index 0 for horizonal maps and 1 for vertical maps.
            These maps are distance from each nuclear pixel to center of mass of corresponding nucleus.
        kernel_size (int): width of kernel to use for gradient approximation.

    Returns:
        Tuple of (h_grad, v_grad) where each is a Tensor giving horizontal and vertical gradients respectively
    """
    assert (
        hv_batch.shape[1] == 2
    ), f"inputs have shape {hv_batch.shape}. Expecting tensor of shape (B, 2, H, W)"
    h_kernel, v_kernel = get_sobel_kernels(kernel_size, dt=hv_batch.dtype)

    # move kernels to same device as batch
    h_kernel = h_kernel.to(hv_batch.device)
    v_kernel = v_kernel.to(hv_batch.device)

    # add extra dims so we can convolve with a batch
    h_kernel = h_kernel.unsqueeze(0).unsqueeze(0)
    v_kernel = v_kernel.unsqueeze(0).unsqueeze(0)

    # get the inputs for the h and v channels
    h_inputs = hv_batch[:, 0, :, :].unsqueeze(dim=1)
    v_inputs = hv_batch[:, 1, :, :].unsqueeze(dim=1)

    h_grad = F.conv2d(h_inputs, h_kernel, stride=1, padding=2)
    v_grad = F.conv2d(v_inputs, v_kernel, stride=1, padding=2)

    del h_kernel
    del v_kernel

    return h_grad, v_grad


# Post-processing of HoVer-Net outputs
def remove_small_objs(array_in, min_size):
    """
    Removes small foreground regions from binary array, leaving only the contiguous regions which are above
    the size threshold. Pixels in regions below the size threshold are zeroed out.

    Args:
        array_in (np.ndarray): Input array. Must be binary array with dtype=np.uint8.
        min_size (int): Minimum size of each region.

    Returns:
        np.ndarray: Array of labels for regions above the threshold. Each separate contiguous region is labelled with
            a different integer from 1 to n, where n is the number of total distinct contiguous regions
    """
    assert (
        array_in.dtype == np.uint8
    ), f"Input dtype is {array_in.dtype}. Must be np.uint8"
    # remove elements below size threshold
    # each contiguous nucleus region gets a unique id
    n_labels, labels = cv2.connectedComponents(array_in)
    # each integer is a different nucleus, so bincount gives nucleus sizes
    sizes = np.bincount(labels.flatten())
    for nucleus_ix, size_ix in zip(range(n_labels), sizes):
        if size_ix < min_size:
            # below size threshold - set all to zero
            labels[labels == nucleus_ix] = 0
    return labels


def _post_process_single_hovernet(
    np_out, hv_out, small_obj_size_thresh=10, kernel_size=21, h=0.5, k=0.5, amp=True
):
    """
    Combine predictions of np channel and hv channel to create final predictions.
    Works by creating energy landscape from gradients, and the applying watershed segmentation.
    This function works on a single image and is wrapped in ``post_process_batch_hovernet()`` to apply across a batch.
    See: Section B of HoVer-Net article and
    https://github.com/vqdang/hover_net/blob/14c5996fa61ede4691e87905775e8f4243da6a62/models/hovernet/post_proc.py#L27

    Args:
        np_out (torch.Tensor): Output of NP branch. Tensor of shape (2, H, W) of logit predictions for binary classification
        hv_out (torch.Tensor): Output of HV branch. Tensor of shape (2, H, W) of predictions for horizontal/vertical maps
        small_obj_size_thresh (int): Minimum number of pixels in regions. Defaults to 10.
        kernel_size (int): Width of Sobel kernel used to compute horizontal and vertical gradients.
        h (float): hyperparameter for thresholding nucleus probabilities. Defaults to 0.5.
        k (float): hyperparameter for thresholding energy landscape to create markers for watershed
            segmentation. Defaults to 0.5.
    """
    # compute pixel probabilities from logits, apply threshold, and get into np array

    # np_out = np_out.float()
    np_preds = (
        scipy.special.softmax(np_out.numpy(), axis=0)[1, :, :]
        if amp
        else F.softmax(np_out, dim=0)[1, :, :].numpy()
    )

    np_preds[np_preds >= h] = 1
    np_preds[np_preds < h] = 0
    np_preds = np_preds.astype(np.uint8)

    np_preds = remove_small_objs(np_preds, min_size=small_obj_size_thresh)
    # Back to binary. now np_preds corresponds to tau(q, h) from HoVer-Net paper
    np_preds[np_preds > 0] = 1
    tau_q_h = np_preds

    # normalize hv predictions, and compute horizontal and vertical gradients, and normalize again
    hv_out = hv_out.numpy().astype(np.float32)
    h_out = hv_out[0, ...]
    v_out = hv_out[1, ...]
    # https://stackoverflow.com/a/39037135
    h_normed = cv2.normalize(
        h_out, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_normed = cv2.normalize(
        v_out, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    h_grad = cv2.Sobel(h_normed, cv2.CV_64F, dx=1, dy=0, ksize=kernel_size)
    v_grad = cv2.Sobel(v_normed, cv2.CV_64F, dx=0, dy=1, ksize=kernel_size)

    h_grad = cv2.normalize(
        h_grad, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_grad = cv2.normalize(
        v_grad, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # flip the gradient direction so that highest values are steepest gradient
    h_grad = 1 - h_grad
    v_grad = 1 - v_grad

    S_m = np.maximum(h_grad, v_grad)
    S_m[tau_q_h == 0] = 0
    # energy landscape
    # note that the paper says that they use E = (1 - tau(S_m, k)) * tau(q, h)
    # but in the authors' code the actually use: E = (1 - S_m) * tau(q, h)
    # this actually makes more sense because no need to threshold the energy surface
    energy = (1.0 - S_m) * tau_q_h

    # get markers
    # In the paper it says they use M = sigma(tau(q, h) - tau(S_m, k))
    # But it makes more sense to threshold the energy landscape to get the peaks of hills.
    # Also, the fact they used sigma in the paper makes me think that this is what they intended,
    m = np.array(energy >= k, dtype=np.uint8)
    m = binary_fill_holes(m).astype(np.uint8)
    m = remove_small_objs(m, min_size=small_obj_size_thresh)

    # nuclei values form mountains so inverse to get basins for watershed
    energy = -cv2.GaussianBlur(energy, (3, 3), 0)
    out = watershed(image=energy, markers=m, mask=tau_q_h)

    return out


def post_process_batch_hovernet(
    outputs, n_classes, small_obj_size_thresh=10, kernel_size=21, h=0.5, k=0.5, amp=True
):
    """
    Post-process HoVer-Net outputs to get a final predicted mask.
    See: Section B of HoVer-Net article and
    https://github.com/vqdang/hover_net/blob/14c5996fa61ede4691e87905775e8f4243da6a62/models/hovernet/post_proc.py#L27

    Args:
        outputs (list): Outputs of HoVer-Net model. List of [np_out, hv_out], or [np_out, hv_out, nc_out]
            depending on whether model is predicting classification or not.

            - np_out is a Tensor of shape (B, 2, H, W) of logit predictions for binary classification
            - hv_out is a Tensor of shape (B, 2, H, W) of predictions for horizontal/vertical maps
            - nc_out is a Tensor of shape (B, n_classes, H, W) of logits for classification

        n_classes (int): Number of classes for classification task. If ``None`` then only segmentation is performed.
        small_obj_size_thresh (int): Minimum number of pixels in regions. Defaults to 10.
        kernel_size (int): Width of Sobel kernel used to compute horizontal and vertical gradients.
        h (float): hyperparameter for thresholding nucleus probabilities. Defaults to 0.5.
        k (float): hyperparameter for thresholding energy landscape to create markers for watershed
            segmentation. Defaults to 0.5.

    Returns:
        np.ndarray: If n_classes is None, returns det_out. In classification setting, returns (det_out, class_out).

            - det_out is np.ndarray of shape (B, H, W)
            - class_out is np.ndarray of shape (B, n_classes, H, W)

            Each pixel is labelled from 0 to n, where n is the number of individual nuclei detected. 0 pixels indicate
            background. Pixel values i indicate that the pixel belongs to the ith nucleus.
    """

    assert len(outputs) in {2, 3}, (
        f"outputs has size {len(outputs)}. Must have size 2 (for segmentation) or 3 (for "
        f"classification)"
    )
    if n_classes is None:
        np_out, hv_out = outputs
        # send ouputs to cpu
        np_out = np_out.detach().cpu()
        hv_out = hv_out.detach().cpu()
        classification = False
    else:
        assert len(outputs) == 3, (
            f"n_classes={n_classes} but outputs has {len(outputs)} elements. Expecting a list "
            f"of length 3, one for each of np, hv, and nc branches"
        )
        np_out, hv_out, nc_out = outputs
        # send ouputs to cpu
        np_out = np_out.detach().cpu()
        hv_out = hv_out.detach().cpu()
        nc_out = nc_out.detach().cpu()
        classification = True

    batchsize = hv_out.shape[0]
    # first get the nucleus detection preds
    out_detection_list = []
    for i in range(batchsize):
        preds = _post_process_single_hovernet(
            np_out[i, ...],
            hv_out[i, ...],
            small_obj_size_thresh,
            kernel_size,
            h,
            k,
            amp,
        )
        out_detection_list.append(preds)
    out_detection = np.stack(out_detection_list)

    if classification:
        # need to do last step of majority vote
        # get the pixel-level class predictions from the logits

        # nc_out = nc_out.float()
        nc_out_preds = (
            scipy.special.softmax(nc_out.numpy(), axis=1).argmax(axis=1)
            if amp
            else F.softmax(nc_out, dim=1).argmax(dim=1).numpy()
        )
        out_classification = np.zeros_like(nc_out, dtype=np.uint8)

        for batch_ix, nuc_preds in enumerate(out_detection_list):
            # get labels of nuclei from nucleus detection
            nucleus_labels = list(np.unique(nuc_preds))
            if 0 in nucleus_labels:
                nucleus_labels.remove(0)  # 0 is background
            nucleus_class_preds = nc_out_preds[batch_ix, ...]

            out_class_preds_single = out_classification[batch_ix, ...]

            # for each nucleus, get the class predictions for the pixels and take a vote
            for nucleus_ix in nucleus_labels:
                # get mask for the specific nucleus
                ix_mask = nuc_preds == nucleus_ix
                votes = nucleus_class_preds[ix_mask]
                majority_class = np.argmax(np.bincount(votes))
                out_class_preds_single[majority_class][ix_mask] = nucleus_ix

            out_classification[batch_ix, ...] = out_class_preds_single

        return out_detection, out_classification, hv_out
    else:
        return out_detection


# plotting hovernet outputs
def _vis_outputs_single(
    images, preds, n_classes, index=0, ax=None, markersize=5, palette=None
):
    """
    Plot the results of HoVer-Net predictions for a single image, overlayed on the original image.

    Args:
        images: Input RGB image batch. Tensor of shape (B, 3, H, W).
        preds: Postprocessed outputs of HoVer-Net. From post_process_batch_hovernet(). Can be either:
            - Tensor of shape (B, H, W), in the context of nucleus detection.
            - Tensor of shape (B, n_classes, H, W), in the context of nucleus classification.
        n_classes (int): Number of classes for classification setting, or None to indicate detection setting.
        index (int): Index of image to plot.
        ax: Matplotlib axes object to plot on. If None, creates a new plot. Defaults to None.
        markersize: Size of markers used to outline nuclei
        palette (list): list of colors to use for plotting. If None, uses matplotlib.colors.TABLEAU_COLORS.
            Defaults to None
    """
    if palette is None:
        palette = list(TABLEAU_COLORS.values())

    if n_classes is not None:
        classification = True
        n_classes = preds.shape[1]
        assert (
            len(palette) >= n_classes
        ), f"len(palette)={len(palette)} < n_classes={n_classes}."
    else:
        classification = False

    assert len(preds.shape) in [
        3,
        4,
    ], f"Preds shape is {preds.shape}. Must be (B, H, W) or (B, n_classes, H, W)"

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(images[index, ...].permute(1, 2, 0))

    if classification is False:
        nucleus_labels = list(np.unique(preds[index, ...]))
        nucleus_labels.remove(0)  # background
        # plot each individual nucleus
        for label in nucleus_labels:
            nuclei_mask = preds[index, ...] == label
            x, y = segmentation_lines(nuclei_mask.astype(np.uint8))
            ax.scatter(x, y, color=palette[0], marker=".", s=markersize)
    else:
        nucleus_labels = list(np.unique(preds[index, ...]))
        nucleus_labels.remove(0)  # background
        # plot each individual nucleus
        for label in nucleus_labels:
            for i in range(n_classes):
                nuclei_mask = preds[index, i, ...] == label
                x, y = segmentation_lines(nuclei_mask.astype(np.uint8))
                ax.scatter(x, y, color=palette[i], marker=".", s=markersize)
    ax.axis("off")


def wrap_transform_multichannel(transform):
    """
    Wrapper to make albumentations transform compatible with a multichannel mask.
    Channel should be in first dimension, i.e. (n_mask_channels, H, W)

    Args:
        transform: Albumentations transform. Must have 'additional_targets' parameter specified with
            a total of `n_channels` key,value pairs. All values must be 'mask' but the keys don't matter.
            e.g. for a mask with 3 channels, you could use:
            `additional targets = {'mask1' : 'mask', 'mask2' : 'mask', 'pathml' : 'mask'}`

    Returns:
        function that can be called with a multichannel mask argument
    """
    targets = transform.additional_targets
    n_targets = len(targets)

    # make sure that everything is correct so that transform is correctly applied
    assert all(
        [v == "mask" for v in targets.values()]
    ), "error all values in transform.additional_targets must be 'mask'."

    def transform_out(*args, **kwargs):
        mask = kwargs.pop("mask")
        assert mask.ndim == 3, f"input mask shape {mask.shape} must be 3-dimensions ()"
        assert (
            mask.shape[0] == n_targets
        ), f"input mask shape {mask.shape} doesn't match additional_targets {transform.additional_targets}"

        mask_to_dict = {key: mask[i, :, :] for i, key in enumerate(targets.keys())}
        kwargs.update(mask_to_dict)
        out = transform(*args, **kwargs)
        mask_out = np.stack([out.pop(key) for key in targets.keys()], axis=0)
        assert mask_out.shape == mask.shape
        out["mask"] = mask_out
        return out

    return transform_out


def plot_training_curves(
    epoch_train_losses,
    epoch_valid_losses,
    epoch_valid_dice,
    best_epoch,
):
    fix, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    ax[0].plot(epoch_train_losses.keys(), epoch_train_losses.values(), label="Train")
    ax[0].plot(
        epoch_valid_losses.keys(), epoch_valid_losses.values(), label="Validation"
    )
    ax[0].scatter(
        x=best_epoch,
        y=epoch_valid_losses[best_epoch],
        label="Best Model",
        color="green",
        marker="*",
    )
    ax[0].set_title("Training: Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # ax[1].plot(epoch_train_dice.keys(), epoch_train_dice.values(), label="Train")
    ax[1].plot(epoch_valid_dice.keys(), epoch_valid_dice.values(), label="Validation")
    ax[1].scatter(
        x=best_epoch,
        y=epoch_valid_dice[best_epoch],
        label="Best Model",
        color="green",
        marker="*",
    )
    ax[1].set_title("Training: Dice Score")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Dice Score")
    ax[1].legend()
    plt.savefig(Path("assets", f"train_{datetime.now().strftime('%H%M%S_%d%m%Y')}.png"))
