import torch
from hover_quant.loss import loss_hovernet

PROFILES = {
    "monusac": {
        "INPUT_SIZE": (256, 256),
        "NUM_WORKERS": 8,
        "NUM_CLASSES": 5,
        "BATCH_SIZE": 8,
        "ACCUM_GRAD_BATCHES": 4,
        "MAX_NUM_EPOCHS": 200,
        "EARLY_STOP_PATIENCE": 30,
        "USE_AMP": True,
        "USE_GRAD_SCALER": True,
        "CRITERION": loss_hovernet,
        "OPTIMIZER": torch.optim.AdamW,
        "OPTIMIZER_PARAMS": {
            "lr": 0.0001,
            "fused": True,
        },
        "SCHEDULER": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "SCHEDULER_PARAMS": {
            "mode": "min",
            "factor": 0.1,
            "patience": 15,
        },
        "EVAL_BATCH_SIZE": None,
        "INF_BATCH_SIZE": 32,
        "MULTI_GPU": False,
        "ACCELERATOR": "cuda",
        "LABELS": {
            0: "Epithelial",
            1: "Lymphocyte",
            2: "Macrophage",
            3: "Neutrophil",
            4: "Background",
        },
    },
    "pannuke": {
        "INPUT_SIZE": (256, 256),
        "NUM_WORKERS": 8,
        "NUM_CLASSES": 6,
        "BATCH_SIZE": 8,
        "ACCUM_GRAD_BATCHES": 4,
        "MAX_NUM_EPOCHS": 200,
        "EARLY_STOP_PATIENCE": 30,
        "USE_AMP": True,
        "USE_GRAD_SCALER": True,
        "CRITERION": loss_hovernet,
        "OPTIMIZER": torch.optim.AdamW,
        "OPTIMIZER_PARAMS": {
            "lr": 0.0001,
        },
        "SCHEDULER": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "SCHEDULER_PARAMS": {
            "mode": "min",
            "factor": 0.1,
            "patience": 15,
        },
        "EVAL_BATCH_SIZE": None,
        "INF_BATCH_SIZE": 32,
        "MULTI_GPU": False,
        "ACCELERATOR": "cuda",
        "LABELS": {
            0: "Neoplastic",
            1: "Inflammatory",
            2: "Connective",  # Connective/Soft tissue cells
            3: "Dead cells",
            4: "Epithelial",
            5: "Background",
        },
    },
}
