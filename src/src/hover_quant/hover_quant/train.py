import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchinfo import summary
import numpy as np

######################################################
import hover_quant.config as config

######################################################
from hover_quant.model import HoVerNet, ModelHandler
from hover_quant.datasets.pannuke import PanNukeDataModule
from hover_quant.datasets.monusac import MoNuSACDataModule
from hover_quant.datasets.data_utils import plot_batch_preproc
from hover_quant.metrics import dice_score
from hover_quant.utils import plot_segmentation
from hover_quant.augs import ATF


if __name__ == "__main__":
    DOWNLOAD = False
    USE_COMBINED = False
    model_fname_pt = None

    if model_fname_pt is not None:
        assert (
            config.PROFILE_ID in model_fname_pt
        ), "Model file name does not match with the profile id"

    if "monusac" in config.PROFILE_ID.lower():
        data_module = MoNuSACDataModule(
            data_dir=config.DATA_DIR,
            download=DOWNLOAD,
            train_val_size=0.85,
            nucleus_type_labels=True,
            hovernet_preprocess=True,
            include_ambiguous=False,
            transforms=(
                (ATF["hover_transform"], ATF["hover_transform"], None)
                if USE_COMBINED
                else (ATF["hover_transform"], None, None)
            ),
        )
    elif "pannuke" in config.PROFILE_ID.lower():
        data_module = PanNukeDataModule(
            data_dir=config.DATA_DIR,
            download=DOWNLOAD,
            nucleus_type_labels=True,
            batch_size=config.BATCH_SIZE,
            hovernet_preprocess=True,
            split=1,
            transforms=ATF["hover_transform"],
        )
    else:
        raise ValueError(
            f"No suitable dataset can be assigned for the given profile: {config.PROFILE_ID}"
        )

    train_dataloader = data_module.train_dataloader
    valid_dataloader = data_module.valid_dataloader
    test_dataloader = data_module.test_dataloader

    if isinstance(data_module, MoNuSACDataModule):
        if data_module.modes["test"]["types_mapping"].get("Ambiguous") is not None:
            print(
                "Using valid dataloader, because of <Ambiguous> class is present in test dataset"
            )
            test_dataloader = data_module.valid_dataloader

    if USE_COMBINED:
        ### Combine train and valid dataloaders, if needed
        combined_dataset = ConcatDataset(
            [train_dataloader.dataset, valid_dataloader.dataset]
        )
        train_dataloader = DataLoader(
            combined_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        valid_dataloader = test_dataloader

    assert (
        test_dataloader.dataset[0][1].shape == valid_dataloader.dataset[0][1].shape
    ), "Inconsistent shape of labels, check <Ambiguous> class label"

    hovernet_model = (
        torch.nn.DistributedDataParallel(HoVerNet(n_classes=config.NUM_CLASSES))
        if config.MULTI_GPU
        else HoVerNet(n_classes=config.NUM_CLASSES)
    ).to(config.ACCELERATOR)

    if model_fname_pt is not None:
        hovernet_model.load_state_dict(
            torch.load(Path("models", model_fname_pt), map_location=config.ACCELERATOR)
        )

    optimizer = config.OPTIMIZER(hovernet_model.parameters(), **config.OPTIMIZER_PARAMS)
    scheduler = config.SCHEDULER(optimizer, **config.SCHEDULER_PARAMS)

    mh = ModelHandler(
        model=hovernet_model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=config.CRITERION,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        device=config.ACCELERATOR,
        add_metrics={"jaccard_score", "pq_score"},
    )

    mh.train(
        accumulate_grad_batches=config.ACCUM_GRAD_BATCHES,
        use_best_model=True,
        plot_report=True,
    )

    ### 2-stage training
    # mh.mod_grad_encoder(freeze=False)
    # mh.train(accumulate_grad_batches=config.ACCUM_GRAD_BATCHES, use_best_model=True, plot_report=config.MAX_NUM_EPOCHS > 50)
    # mh.mod_grad_encoder(freeze=True)
    # mh.train(accumulate_grad_batches=config.ACCUM_GRAD_BATCHES, use_best_model=True, plot_report=config.MAX_NUM_EPOCHS > 50)
