from pathlib import Path
import torch
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from patch_class.model import PatchClassifier, ImgAugmentor, ModelTrainer
from patch_class.dataset import PatchDataset, AugmentedDataLoader
from patch_class.utils import construct_cm

# import warnings
# warnings.filterwarnings("ignore")


if __name__ == "__main__":
    conf = {
        "mode": "eval",
        "run_params": {
            "single_run": False,
            "kfold": 5,
        },
    }

    assert conf["mode"] in ["train", "eval"], "Invalid mode"
    assert conf["run_params"]["kfold"] > 1, "Invalid kfold"

    # Data info
    train_val_dataset = PatchDataset(
        config.TRAIN_DATA_DIR,
        config.ATF,
        selected_folds=(1),
        kfold=1,
    )

    print("=====================================")
    print(f"Train dataset info: {config.TRAIN_DATA_DIR}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Number of images: {len(train_val_dataset)}")
    print(f"Labels: {train_val_dataset.class_to_idx}")
    print(f"Number of images per class: {train_val_dataset.imgs_per_class}")
    print(f"Using normalization: {config.NORM_ARGS}")
    print(f"Using stain color augmentations: {config.COLOR_AUG_ARGS}")
    print("=====================================")

    ### Test dataset
    test_dataset = PatchDataset(
        config.TEST_DATA_DIR,
        config.ATF,
        selected_folds=(1),
        kfold=1,
        clip_to_min=True,
        # clip_seed=42,
    )
    print("=====================================")
    print(f"Test dataset info: {config.TEST_DATA_DIR}")
    print(f"Test dataset length: {len(test_dataset)}")
    print(f"Number of images per class: {test_dataset.imgs_per_class}")
    print(
        f"Targets for test_dataset: {dict(zip(*np.unique(torch.tensor(test_dataset.targets).numpy(), return_counts=True)))}"
    )
    print("=====================================")

    ### Augmentors
    train_augmentor = ImgAugmentor(
        config.ATF,
        p_augment=True,
        preproc=True,
        norm_args=config.NORM_ARGS,
        color_augment_args=config.COLOR_AUG_ARGS,
        # use_fast_color_aug=False,
        clamp_values=config.CLAMP_VALUES,
        train_mode=True,
        proc_device=config.ACCELERATOR,
        target_device=config.ACCELERATOR,
    )

    val_test_augmentor = ImgAugmentor(
        config.ATF,
        p_augment=False,
        preproc=True,
        norm_args=config.NORM_ARGS,
        # color_augment_args=None,
        # use_fast_color_aug=False,
        clamp_values=config.CLAMP_VALUES,
        train_mode=False,
        proc_device=config.ACCELERATOR,
        target_device=config.ACCELERATOR,
    )

    test_dataloader = AugmentedDataLoader(
        DataLoader(
            test_dataset,
            batch_size=config.INF_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        ),
        val_test_augmentor,
    )

    # Model
    model = PatchClassifier(
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        out_classes=config.NUM_CLASSES,
        conf_dropout_classifier=config.CONF_DROPOUT_CLASSIFIER,
    )

    optimizer = config.OPTIMIZER(model.parameters(), **config.OPTIMIZER_PARAMS)
    scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_PARAMS)

    criterion_params = config.CRITERTION_PARAMS
    for crit_param_key in criterion_params.keys():
        if isinstance(criterion_params[crit_param_key], list) or isinstance(
            criterion_params[crit_param_key], torch.Tensor
        ):
            criterion_params[crit_param_key] = (
                torch.tensor(criterion_params[crit_param_key])
                .float()
                .to(config.ACCELERATOR)
            )

    mh = ModelTrainer(
        model=model,
        loss_fn=config.CRITERTION(**criterion_params),
        optimizer=optimizer,
        scheduler=scheduler,
        # train_dataloader=train_dataloader,
        # valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        val_metric=config.VAL_METRIC,
        device=config.ACCELERATOR,
    )

    if conf["mode"] == "train":
        torch.backends.cudnn.deterministic = config.CUDA_DETERMINISTIC
        torch.set_float32_matmul_precision = config.MATMUL_PRECISION
        torch.backends.cudnn.benchmark = config.CUDA_BENCHMARK

        print(f"Training model: {config.MODEL_NAME} from {config.PROFILE_ID} profile")

        for val_fold in range(1, conf["run_params"]["kfold"] + 1):
            train_folds = [
                x for x in range(1, conf["run_params"]["kfold"] + 1) if x != val_fold
            ]
            val_folds = [val_fold]
            print()
            print(
                f"<<<<<<<< Training on folds: {train_folds} | Validating on folds: {val_folds} >>>>>>>>"
            )

            train_subset = PatchDataset(
                config.TRAIN_DATA_DIR,
                config.ATF,
                selected_folds=train_folds,
                kfold=conf["run_params"]["kfold"],
            )
            valid_subset = PatchDataset(
                config.TRAIN_DATA_DIR,
                config.ATF,
                selected_folds=val_folds,
                kfold=conf["run_params"]["kfold"],
            )

            print("=====================================")
            print(
                f"Train dataset length: {len(train_subset)} | {round(len(train_subset) / len(train_val_dataset), 2)}"
            )
            print(
                f"Val dataset length: {len(valid_subset)} | {round(len(valid_subset) / len(train_val_dataset), 2)}"
            )
            print("=====================================")
            print(
                f"Targets for train_subset: {dict(zip(*np.unique(torch.tensor(train_subset.targets).numpy(), return_counts=True)))}"
            )
            print(
                f"Targets for valid_subset: {dict(zip(*np.unique(torch.tensor(valid_subset.targets).numpy(), return_counts=True)))}"
            )
            print("=====================================")

            ### Dataloaders
            train_dataloader = AugmentedDataLoader(
                DataLoader(
                    train_subset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=config.NUM_WORKERS,
                    pin_memory=True,
                ),
                train_augmentor,
            )

            valid_dataloader = AugmentedDataLoader(
                DataLoader(
                    valid_subset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=config.NUM_WORKERS,
                    pin_memory=True,
                ),
                val_test_augmentor,
            )

            mh.train_dataloader = train_dataloader
            mh.valid_dataloader = valid_dataloader

            mh.train(
                accumulate_grad_batches=config.ACCUM_GRAD_BATCHES,
                save_final=(
                    conf["run_params"]["single_run"]
                    or val_fold == conf["run_params"]["kfold"]
                ),
            )
            if mh.best_trained_model["model_state_dict"] is not None:
                mh.model.load_state_dict(mh.best_trained_model["model_state_dict"])

            if conf["run_params"]["single_run"]:
                break
            else:
                mh.reset_state_for_next_fold()

    elif conf["mode"] == "eval":
        models_paths = sorted(
            [str(x) for x in Path("models").glob(f"{config.PROFILE_ID}*.pt")]
        )
        if len(models_paths) == 0:
            raise FileNotFoundError("No model found to evaluate")
        mh.load_state(models_paths[-1])
        print(f"Loaded model: {models_paths[-1]}")

        summary(
            mh.model, input_size=(config.BATCH_SIZE, *train_val_dataset[0][0].shape)
        )

    ################################################################################
    ################################################################################

    print(f"Evaluating model: {config.MODEL_NAME} from {config.PROFILE_ID} profile")
    # Evaluate on test set
    mh.test_dataloader = test_dataloader
    out_eval = mh.evaluate(dataloader=test_dataloader)
    loss = out_eval["avg_loss"]
    accuracy = out_eval["accuracy"]
    precision = out_eval["precision"]
    recall = out_eval["recall"]
    roc_auc = out_eval["roc_auc"]

    print(
        f"""
    Model evaluation on test dataloader: \t
    Loss = [{loss:0.5f}] \t
    Accuracy = [{(accuracy * 100):0.2f}%] \t
    Precision = [{(precision * 100):0.2f}%] \t
    Recall = [{(recall * 100):0.2f}%] \t
    ROC AUC = [{(roc_auc * 100):0.2f}%] \t
    """
    )

    construct_cm(
        torch.tensor(test_dataset.targets),
        out_eval["preds"],
        test_dataset.class_to_idx.keys(),
        config,
        save_file_path=Path(
            "assets", f"cm_{config.PROFILE_ID}_{(accuracy * 100):0.2f}.png"
        ),
    )
