import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from focal_loss.focal_loss import FocalLoss
from torch.nn import CrossEntropyLoss


PROFILES = {
    "efficientnetv2_s": {
        "MODEL_NAME": "tf_efficientnetv2_s.in21k_ft_in1k",
        "PRETRAINED": False,
        "NUM_CLASSES": 4,
        "CONF_DROPOUT_CLASSIFIER": {"num_layers": 1, "dropout_p": [0.3]},
        "MAX_NUM_EPOCHS": 400,  # 400
        "BATCH_SIZE": 128,
        "ACCUM_GRAD_BATCHES": 1,
        "USE_AMP": True,
        "USE_GRAD_SCALER": True,
        "EARLY_STOP_PATIENCE": 30,  # int or None
        "OPTIMIZER": AdamW,
        "OPTIMIZER_PARAMS": {
            "lr": 0.001,
            "weight_decay": 0.01,  # Default: 0.01; 5e-4 0.0005
        },
        "SCHEDULER": ReduceLROnPlateau,
        "SCHEDULER_PARAMS": {
            "mode": "min",
            "factor": 0.1,
            "patience": 15,
        },
        "CRITERTION": FocalLoss,  # [CrossEntropyLoss, FocalLoss]
        "CRITERTION_PARAMS": {
            "gamma": 0.7,
            "weights": [2.0, 0.5, 1.5, 1.0],
        },
        "VAL_METRIC": "avg_loss",
        "INPUT_SIZE": (512, 512),  # resize from 768x768 to 512x512
        # "NORM_ARGS": None,
        "NORM_ARGS": {
            "method": "macenko",  # ['vahadane', 'macenko', 'reinhard']
            "concentration_method": "ls",  # ['ls', 'cd', 'ista']
            "ref_img_name": "1_ref_img.png",
        },
        # "COLOR_AUG_ARGS": None,
        "COLOR_AUG_ARGS": {
            "method": "macenko",  # ['vahadane', 'macenko']
            "seed": 42,
            "luminosity_threshold": 0.8,
            "concentration_method": "ls",  # ['ls', 'cd', 'ista']
            "sigma_alpha": 0.7,  # 0.2
            "sigma_beta": 0.2,  # 0.2
        },
        "CLAMP_VALUES": False,
        "NUM_WORKERS": 8,
        "TRAIN_DATA_DIR": "/mnt/data/composed/train_val",
        "TEST_DATA_DIR": "/mnt/data/composed/test",
        "ACCELERATOR": str(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ),
        "MATMUL_PRECISION": "medium",
        "CUDA_DETERMINISTIC": True,
        "CUDA_BENCHMARK": False,
        "INF_BATCH_SIZE": 64,
    },
}
