import torch
from torch.utils.data import random_split
import cv2
import numpy as np
import matplotlib.pyplot as plt

##########################################
from roi_segment.model import ROIDetector, ModelTrainer
from roi_segment.dataset import DsWSIDataset

##########################################
import roi_segment.config as config
import roi_segment.augs as augs

lt.monkey_patch()
lt.set_config(sci_mode=False)
torch.set_printoptions(sci_mode=False)

if __name__ == "__main__":
    # Dataset
    full_dataset = DsWSIDataset(
        config.WSI_DIR,
        config.ITN_DIR,
        augs.ATF,
        preproc=True,
        augment=False,
        for_dataloader=False,
    )
    generator = torch.Generator().manual_seed(42)
    _, _, dataset = random_split(full_dataset, [0.8, 0.1, 0.1], generator=generator)

    # Model handler
    mh = ModelTrainer(
        model=ROIDetector(
            smp_model=config.SMP_MODEL,
            encoder_name=config.ENCODER_NAME,
            in_channels=config.IN_CHANNELS,
            out_classes=config.NUM_CLASSES,
            init_weights=config.WEIGHTS,
            model_params=config.MODEL_PARAMS,
        ),
        device=config.ACCELERATOR,
    )

    # Load model
    mh.load(config.INFER_MODEL)

    # Data of conventional format (HxWxC)
    # img = dataset[0][0].numpy().transpose(1, 2, 0)
    # gt_mask = dataset[0][1].numpy().squeeze()

    # Data of torch format (CxHxW)
    idx_random = np.random.randint(0, len(dataset))
    img = dataset[idx_random][0]
    gt_mask = dataset[idx_random][1]
    ref_image = dataset[idx_random][2]
    ref_mask = dataset[idx_random][3]

    # Infer
    pred_mask = mh.predict(img)

    pred_mask = np.array(pred_mask, np.uint8)

    # ############################
    # # Extract polygons from mask
    # # https://stackoverflow.com/questions/58884265/python-convert-binary-mask-to-polygon
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = {}
    for k, object in enumerate(contours):
        coords = []
        for point in object:
            coords.append((point[0][0], point[0][1]))
        polygons[k] = coords

    # # Rescale mask
    rescaled_polygons = DsWSIDataset._scale_coords(
        (pred_mask.shape[0], pred_mask.shape[1]),
        (ref_image.shape[1], ref_image.shape[0]),
        polygons,
    )

    # # Fill polygons
    mask = np.zeros(ref_image.shape[0:2], dtype=np.uint8)
    for polygon_points in rescaled_polygons.values():
        npcoords = np.array([polygon_points], np.int32)
        cv2.fillPoly(mask, npcoords, 1)
    mask = mask.astype(bool) * 255

    plt.subplot(2, 2, 1)
    plt.imshow(ref_image)
    plt.title("Ref image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(ref_mask)
    plt.title("GT mask")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(ref_image)
    plt.imshow(mask, alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(mask)
    plt.title("Prediction")
    plt.axis("off")

    plt.savefig("test.png")
