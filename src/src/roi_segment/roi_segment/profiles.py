import torch
import segmentation_models_pytorch as smp
import bitsandbytes as bnb

PROFILES = {
    "Unet_xception": {
        "SMP_MODEL": smp.Unet,
        "ENCODER_NAME": "xception",
        "WEIGHTS": "imagenet",
        "MODEL_PARAMS": {
            "decoder_attention_type": "scse",
            "decoder_use_batchnorm": True,
        },
        "IN_CHANNELS": 3,
        "NUM_CLASSES": 1,
        "MAX_NUM_EPOCHS": 100,
        "BATCH_SIZE": 2,
        "ACCUM_GRAD_BATCHES": 8,
        "USE_AMP": True,
        "USE_GRAD_SCALER": True,
        # "EARLY_STOP": True,
        # "PATIENCE": 10,
        "EARLY_STOP_PATIENCE": 30,  # int or None
        # "OPTIMIZER": torch.optim.Adam,
        # "OPTIMIZER_PARAMS": {
        #     "lr": 0.001,
        #     "weight_decay": 5e-4,
        # },
        "OPTIMIZER": bnb.optim.Adam8bit,
        "OPTIMIZER_PARAMS": {
            "lr": 0.001,
            "weight_decay": 5e-4,
        },
        "SCHEDULER": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "SCHEDULER_PARAMS": {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
            "verbose": True,
        },
        "CRITERTION": smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True),
        "INPUT_SIZE": (2048, 2048),  # (width, height)
        "NUM_WORKERS": 8,
        "PREPROC": True,
        "AUGMENT": True,
        "ACCELERATOR": str(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ),
        "MATMUL_PRECISION": "medium",
        "CUDNN_BENCHMARK": False,
        "CUDNN_DETERMINISTIC": True,
    },
}
