import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import cv2
from tqdm import tqdm, trange
from pathlib import Path
from patchify import patchify, unpatchify
from scipy.ndimage import label

#####################################################
import hover_quant.config as config

#####################################################
from hover_quant.layers import _HoVerNetEncoder, _HoverNetDecoder
from hover_quant.utils import (
    post_process_batch_hovernet,
    save_run_config,
    plot_training_curves,
)
from hover_quant.metrics import dice_score, jaccard_score, pq_score

import matplotlib.pyplot as plt


class HoVerNet(nn.Module):
    """
    Model for simultaneous segmentation and classification based on HoVer-Net.
    Can also be used for segmentation only, if class labels are not supplied.
    Each branch returns logits.

    Args:
        n_classes (int): Number of classes for classification task. If ``None`` then the classification branch is not
            used.

    References:
        Graham, S., Vu, Q.D., Raza, S.E.A., Azam, A., Tsang, Y.W., Kwak, J.T. and Rajpoot, N., 2019.
        Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images.
        Medical Image Analysis, 58, p.101563.

        https://github.com/Dana-Farber-AIOS/pathml
    """

    def __init__(self, n_classes=None):
        super().__init__()
        self.n_classes = n_classes
        self.encoder = _HoVerNetEncoder()

        # NP branch (nuclear pixel)
        self.np_branch = _HoverNetDecoder()
        # classification head
        self.np_head = nn.Sequential(
            # two channels in output - background prob and pixel prob
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        )

        # HV branch (horizontal vertical)
        self.hv_branch = _HoverNetDecoder()  # hv = horizontal vertical
        # classification head
        self.hv_head = nn.Sequential(
            # two channels in output - horizontal and vertical
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)
        )

        # NC branch (nuclear classification)
        # If n_classes is none, then we are in nucleus detection, not classification, so we don't use this branch
        if self.n_classes is not None:
            self.nc_branch = _HoverNetDecoder()
            # classification head
            self.nc_head = nn.Sequential(
                # one channel in output for each class
                nn.Conv2d(in_channels=64, out_channels=self.n_classes, kernel_size=1)
            )

    def forward(self, inputs):
        encoded = self.encoder(inputs)

        """for i, block_output in enumerate(encoded):
            print(f"block {i} output shape: {block_output.shape}")"""

        out_np = self.np_branch(encoded)
        out_np = self.np_head(out_np)

        out_hv = self.hv_branch(encoded)
        out_hv = self.hv_head(out_hv)

        outputs = [out_np, out_hv]

        if self.n_classes is not None:
            out_nc = self.nc_branch(encoded)
            out_nc = self.nc_head(out_nc)
            outputs.append(out_nc)

        return outputs


class ModelHandler:
    def __init__(
        self,
        model,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        train_dataloader=None,
        valid_dataloader=None,
        test_dataloader=None,
        device=None,
        add_metrics=[],
    ):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.model = model.to(device=self.device)

        self.criterion = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.patience = config.EARLY_STOP_PATIENCE
        self.best_trained_model = {
            "score": 0,
            "state_dict": None,
            "current_patience": self.patience,
        }
        self.add_metrics = add_metrics  # ["jaccard_score", "pq_score"]

    def mod_grad_encoder(self, freeze=False):
        for param in self.model.encoder.parameters():
            if freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def eval_step(self, imgs, masks, hv):
        self.model.eval()

        if config.USE_AMP:
            with torch.amp.autocast(device_type=config.ACCELERATOR, cache_enabled=True):
                outputs = self.model(imgs)
                loss = self.criterion(
                    outputs=outputs,
                    ground_truth=[masks, hv],
                    n_classes=config.NUM_CLASSES,
                )
        else:
            outputs = self.model(imgs)
            loss = self.criterion(
                outputs=outputs,
                ground_truth=[masks, hv],
                n_classes=config.NUM_CLASSES,
            )

        preds_detection, preds_classification, _ = post_process_batch_hovernet(
            outputs, n_classes=config.NUM_CLASSES, amp=config.USE_AMP
        )
        masks = masks.cpu()
        truth_binary = masks[:, -1, :, :] == 0

        out_metrics = {
            "loss": float(loss.item()),
            "dice_score": dice_score(np_preds_detection, truth_binary.numpy()),
        }

        if len(self.add_metrics) != 0:
            if "jaccard_score" in self.add_metrics:
                out_metrics["jaccard_score"] = jaccard_score(
                    torch.from_numpy(
                        np.argmax(np_preds_classification[:, :-1, :, :], axis=1)
                    ),
                    torch.argmax(masks[:, :-1, :, :], dim=1),
                    num_classes=config.NUM_CLASSES - 1,
                )
            if "pq_score" in self.add_metrics:
                out_metrics["pq_score"] = pq_score(
                    torch.from_numpy(np_preds_classification[:, :-1, :, :]),
                    masks[:, :-1, :, :],
                    num_classes=config.NUM_CLASSES - 1,
                )

        return out_metrics

    def evaluate(self, dataloader, reduced=False):
        # losses = []
        # dice_scores = []
        # jaccard_scores = []
        # pq_scores = []
        accum_list_out = {
            "losses": [],
            "dice_scorees": [],
            "jaccard_scores": [],
            "pq_scores": [],
        }

        reduced_idx = (
            np.random.choice(
                range(len(dataloader)), replace=False, size=config.EVAL_BATCH_SIZE
            )
            if reduced
            else range(len(dataloader))
        )

        with torch.no_grad():
            for batch_idx, (imgs, masks, hv, _) in enumerate(dataloader):
                if batch_idx in reduced_idx:
                    imgs = imgs.type(torch.FloatTensor).to(device=self.device)
                    masks = masks.to(device=self.device)
                    hv = hv.type(torch.FloatTensor).to(device=self.device)

                    out_eval_step = self.eval_step(imgs, masks, hv)

                    accum_list_out["losses"].append(out_eval_step["loss"])
                    accum_list_out["dice_scorees"].append(out_eval_step["dice_score"])
                    if "jaccard_score" in out_eval_step.keys():
                        accum_list_out["jaccard_scores"].append(
                            out_eval_step["jaccard_score"]
                        )
                    if "pq_score" in out_eval_step.keys():
                        accum_list_out["pq_scores"].append(out_eval_step["pq_score"])
        out = {
            "mean_loss": np.mean(accum_list_out["losses"]),
            "mean_dice": np.mean(accum_list_out["dice_scorees"]),
            "mean_jaccard": (
                np.mean(accum_list_out["jaccard_scores"])
                if len(accum_list_out["jaccard_scores"]) > 0
                else None
            ),
            "mean_pq": (
                np.mean(accum_list_out["pq_scores"])
                if len(accum_list_out["pq_scores"]) > 0
                else None
            ),
        }
        # filter from out empty lists in values
        return {k: v for k, v in out.items() if v}

    def train_step(self, imgs, masks, hv, grad_step=True, accumulate_norm_factor=1):
        # Grad accumulation: https://kozodoi.me/blog/20210219/gradient-accumulation

        self.model.train()

        # Forward
        if config.USE_AMP:
            with torch.amp.autocast(device_type=config.ACCELERATOR, cache_enabled=True):
                outputs = self.model(imgs)
                loss = self.criterion(
                    outputs=outputs,
                    ground_truth=[masks, hv],
                    n_classes=config.NUM_CLASSES,
                )
        else:
            outputs = self.model(imgs)
            loss = self.criterion(
                outputs=outputs, ground_truth=[masks, hv], n_classes=config.NUM_CLASSES
            )

        if config.USE_GRAD_SCALER:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if grad_step:
                self.optimizer.step()

        if grad_step:
            self.optimizer.zero_grad()

        return float(loss.item()) / accumulate_norm_factor

    def train(
        self,
        train_dataloader=None,
        valid_dataloader=None,
        accumulate_grad_batches=1,
        use_best_model=False,
        plot_report=False,
    ):
        self.best_trained_model["current_patience"] = self.patience

        if train_dataloader is None:
            train_dataloader = self.train_dataloader
        if valid_dataloader is None:
            valid_dataloader = self.valid_dataloader

        epoch_metrics = {
            "train_losses": {},
            "valid_losses": {},
            "valid_dice": {},
            "valid_jaccard": {},
            "valid_pq": {},
        }

        self.model = self.model.to(self.device)
        for epoch in trange(1, config.MAX_NUM_EPOCHS + 1, desc="Epochs"):
            train_loss = []  # train epoch loss
            for batch_idx, (imgs, masks, hv, _) in enumerate(
                tqdm(train_dataloader, desc="Batches", leave=False)
            ):
                imgs = imgs.type(torch.FloatTensor).to(device=self.device)
                masks = masks.to(device=self.device)
                hv = hv.type(torch.FloatTensor).to(device=self.device)

                # targets = targets.type(torch.LongTensor).to(device=self.device)
                do_grad_step = ((batch_idx + 1) % accumulate_grad_batches == 0) or (
                    batch_idx + 1 == len(train_dataloader)
                )

                # train batch loss
                train_loss.append(
                    self.train_step(
                        imgs,
                        masks,
                        hv,
                        grad_step=do_grad_step,
                        accumulate_norm_factor=accumulate_grad_batches,
                    )
                )
            epoch_train_loss = np.mean(train_loss)

            val_out_eval = self.evaluate(
                valid_dataloader,
                reduced=True if config.EVAL_BATCH_SIZE is not None else False,
            )

            epoch_eval_loss = val_out_eval["mean_loss"]
            epoch_eval_dice = val_out_eval["mean_dice"]
            epoch_eval_jaccard = val_out_eval.get("mean_jaccard", None)
            epoch_eval_pq = val_out_eval.get("mean_pq", None)

            epoch_metrics["train_losses"].update({epoch: epoch_train_loss})
            epoch_metrics["valid_losses"].update({epoch: epoch_eval_loss})
            epoch_metrics["valid_dice"].update({epoch: epoch_eval_dice})
            epoch_metrics["valid_jaccard"].update({epoch: epoch_eval_jaccard})
            epoch_metrics["valid_pq"].update({epoch: epoch_eval_pq})

            log_string = f"""
            Epoch [{epoch}]: \t 
            train loss = [{epoch_train_loss:0.5f}] \t 
            val loss = [{epoch_eval_loss:0.5f}] \t 
            val dice score = [{(epoch_eval_dice * 100):0.2f}%] \t
            """
            if epoch_eval_jaccard is not None:
                log_string += (
                    f"val jaccard score = [{(epoch_eval_jaccard * 100):0.2f}%] \t"
                )
            if epoch_eval_pq is not None:
                log_string += f"val PQ score = [{(epoch_eval_pq * 100):0.2f}%] \t"

            tqdm.write(f"""{log_string}""")

            # Scheduler step
            if self.scheduler is not None:
                if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(epoch_train_loss)  # loss
                elif (
                    type(self.scheduler) == torch.optim.lr_scheduler.MultiStepLR
                    or torch.optim.lr_scheduler.StepLR
                ):
                    self.scheduler.step()

            # Tracking best score model - using: epoch_eval_dice
            if epoch_eval_dice > self.best_trained_model["score"]:
                self.best_trained_model["score"] = epoch_eval_dice
                self.best_trained_model["model_state_dict"] = (
                    self.model.state_dict().copy()
                )
                self.best_trained_model["current_patience"] = self.patience

            # Early stopping
            else:
                if self.patience:
                    self.best_trained_model["current_patience"] -= 1
                    if self.best_trained_model["current_patience"] < 0:
                        tqdm.write(f"Early stopping at epoch {epoch}")
                        break

        if plot_report:
            plot_training_curves(
                epoch_metrics["train_losses"],
                epoch_metrics["valid_losses"],
                epoch_metrics["valid_dice"],
                np.argmax(list(epoch_metrics["valid_dice"].values())) + 1,
            )

        if use_best_model:
            self.model.load_state_dict(self.best_trained_model["model_state_dict"])

        ####### Test set evaluation #######
        test_out_eval = self.evaluate(
            self.test_dataloader
            if self.test_dataloader is not None
            else self.valid_dataloader
        )
        test_dice = test_out_eval["mean_dice"]
        test_jaccard = test_out_eval.get("mean_jaccard", None)
        test_pq = test_out_eval.get("mean_pq", None)

        test_log_string = f"""
        Saving {config.PROFILE_ID} with val dice score {(self.best_trained_model["score"] * 100):0.2f} and \t 
        test set dice score = {(test_dice * 100):0.2f} \t
        """
        if test_jaccard is not None:
            test_log_string += (
                f"test set jaccard score = {(test_jaccard * 100):0.2f} \t"
            )
        if test_pq is not None:
            test_log_string += f"test set PQ score = {(test_pq * 100):0.2f} \t"
        print(f"""{test_log_string}""")

        save_str = f"{config.PROFILE_ID}_{(test_dice * 100):0.2f}.pt"

        print(f"Model file: {save_str}")
        if config.MODEL_SAVE_DIR is not None:
            self.save(f"{Path(config.MODEL_SAVE_DIR, save_str)}")
        else:
            self.save(save_str)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_run_config(config, path)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def run_infer_test_batch(self, dataloader):
        self.model.eval()

        imgs = None
        mask_truth = None
        mask_pred = None
        tissue_types = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                images = data[0].float().to(device)
                masks = data[1].to(device)
                hv = data[2].float().to(device)
                tissue_type = data[3]

                # pass thru network to get predictions
                outputs = self.model(images)
                preds_detection, preds_classification = post_process_batch_hovernet(
                    outputs, n_classes=config.NUM_CLASSES, amp=config.USE_AMP
                )

                if i == 0:
                    imgs = data[0].numpy()
                    mask_truth = data[1].numpy()
                    mask_pred = preds_classification
                    tissue_types.extend(tissue_type)
                else:
                    imgs = np.concatenate([imgs, data[0].numpy()], axis=0)
                    mask_truth = np.concatenate([mask_truth, data[1].numpy()], axis=0)
                    mask_pred = np.concatenate(
                        [mask_pred, preds_classification], axis=0
                    )
                    tissue_types.extend(tissue_type)

        # collapse multi-class preds into binary preds
        preds_detection = np.sum(mask_pred, axis=1)
        dice_scores = np.empty(shape=len(tissue_types))
        for i in range(len(tissue_types)):
            truth_binary = mask_truth[i, -1, :, :] == 0
            preds_binary = preds_detection[i, ...] != 0
            dice = dice_score(preds_binary, truth_binary)
            dice_scores[i] = dice

        if plot_tissue:
            dice_by_tissue = pd.DataFrame(
                {"Tissue Type": tissue_types, "dice": dice_scores}
            )
            dice_by_tissue.groupby("Tissue Type").mean().plot.bar()
            plt.title("Dice Score by Tissue Type")
            plt.ylabel("Averagae Dice Score")
            plt.gca().get_legend().remove()
            plt.figsave(f"_graph_dice_by_tissue.png")

        print(f"Average Dice score in test set: {np.mean(dice_scores)}")

        imgs = np.moveaxis(imgs, 1, 3)
        n = 8
        ix = np.random.choice(np.arange(len(tissue_types)), size=n)
        fig, ax = plt.subplots(nrows=n, ncols=2, figsize=(8, 2.5 * n))

        for i, index in enumerate(ix):
            ax[i, 0].imshow(ims[index, ...])
            ax[i, 1].imshow(ims[index, ...])
            plot_segmentation(ax=ax[i, 0], masks=mask_pred[index, ...])
            plot_segmentation(ax=ax[i, 1], masks=mask_truth[index, ...])
            ax[i, 0].set_ylabel(tissue_types[index])

        for a in ax.ravel():
            a.get_xaxis().set_ticks([])
            a.get_yaxis().set_ticks([])

        ax[0, 0].set_title("Prediction")
        ax[0, 1].set_title("Truth")
        plt.tight_layout()
        plt.figsave(f"_segmentation_GT.png")

    def predict_image(self, img_np):
        """
        Inputs
            - Img: numpy array of shape (256, 256, 3)

        Outputs
            - Updated image with contours
        """

        self.model = self.model.to(self.device)
        self.model.eval()
        img_torch = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device=self.device)
        )

        with torch.no_grad():
            result = self.model(img_torch)
            preds_detection, preds_classification, hv_out = post_process_batch_hovernet(
                result, n_classes=config.NUM_CLASSES
            )
            preds_detection = label(preds_detection.squeeze())[0]
            preds_classification = preds_classification.squeeze()
            labeled_preds_classification = []
            for i in range(
                preds_classification.shape[0] - 1
            ):  # ignore last channel which is background
                # labeled_preds_classification.append(label(preds_classification[i, ...])[0])
                labeled_preds_classification.append(preds_classification[i, ...])
            preds_classification = np.array(labeled_preds_classification)
        return preds_detection, preds_classification, hv_out.numpy()

    @staticmethod
    def get_visualized_cell_map(img_copy, preds_classification, only_label_idx=None):
        """
        only_label_idx: int, index of the label to visualize, from config.LABELS. Example: 0 for 'Neoplastic'
        """

        ### color map
        color_values = [
            (255, 0, 0),
            (0, 255, 0),
            (144, 238, 144),
            (0, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (128, 128, 128),
            (128, 0, 0),
        ]

        # all, except background (which is last in config.LABELS)
        label_names = list(config.LABELS.values())[:-1]
        colors = {}
        for i, value in enumerate(label_names):
            colors[value] = color_values[i]

        if only_label_idx is not None:
            preds_classification = np.expand_dims(
                preds_classification[only_label_idx], axis=0
            )
            color = colors[label_names[only_label_idx]]

        for class_map in preds_classification:
            preds_detection = class_map

            for cell_instance in np.unique(preds_detection):
                if only_label_idx is None:
                    class_indices = np.argmax(preds_classification, axis=0)
                    cell_indices = np.where(preds_detection == cell_instance)
                    cell_classes = class_indices[cell_indices]
                    most_common_class = np.bincount(cell_classes).argmax()
                    color = colors[label_names[most_common_class]]

                elif only_label_idx is not None:
                    if cell_instance not in preds_classification:
                        continue

                cell_detection_map = np.where(
                    preds_detection == cell_instance, cell_instance, 0
                ).astype(np.uint8)
                contours, _ = cv2.findContours(
                    cell_detection_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    cv2.drawContours(img_copy, [contour], -1, color, 2)

        return img_copy
