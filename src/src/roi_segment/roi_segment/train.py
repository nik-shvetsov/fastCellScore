import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp

####################
from roi_segment.model import ROIDetector, ModelTrainer
from roi_segment.dataset import DsWSIDataset

####################
import roi_segment.config as config
import roi_segment.augs as augs

####################
# from accelerate import Accelerator


if __name__ == "__main__":
    # torch.set_float32_matmul_precision(config.MATMUL_PRECISION)

    ### Additionally, some operations on a GPU are implemented stochastic for efficiency
    ### We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.deterministic = config.CUDNN_DETERMINISTIC
    # torch.backends.cudnn.benchmark = config.CUDNN_BENCHMARK

    dataset = DsWSIDataset(
        config.WSI_DIR,
        config.ITN_DIR,
        augs.ATF,
        preproc=config.PREPROC,
        augment=config.AUGMENT,
        for_dataloader=True,
    )
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [0.8, 0.1, 0.1], generator=generator
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    model = ROIDetector(
        smp_model=config.SMP_MODEL,
        encoder_name=config.ENCODER_NAME,
        in_channels=config.IN_CHANNELS,
        out_classes=config.NUM_CLASSES,
        init_weights=config.WEIGHTS,
        model_params=config.MODEL_PARAMS,
    )

    optimizer = config.OPTIMIZER(model.parameters(), **config.OPTIMIZER_PARAMS)
    scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_PARAMS)

    mh = ModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=config.CRITERTION,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        device=config.ACCELERATOR,
    )

    # Train model
    mh.train(accumulate_grad_batches=config.ACCUM_GRAD_BATCHES, save_final=True)

    # Evaluate on test set
    out_eval = mh.evaluate(dataloader=test_dataloader)
    loss = out_eval["avg_loss"]
    dice_score = out_eval["avg_dice"]
    iou_score = out_eval["avg_iou"]
    print(
        f"""
    Model evaluation: \t
    Loss = [{loss:0.5f}] \t
    Dice score = [{(dice_score * 100):0.2f}%] \t
    mIOU score = [{(iou_score * 100):0.2f}%] \t
    """
    )
