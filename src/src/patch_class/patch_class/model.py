import math
import copy
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
from focal_loss.focal_loss import FocalLoss
import timm
from pprint import pprint
from tqdm import tqdm, trange
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_precision,
    multiclass_recall,
    multiclass_stat_scores,
)
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import lovely_tensors as lt

import patch_class.config as config
from patch_class.utils import test_net, save_run_config

from torch_staintools.normalizer import NormalizerBuilder
from torch_staintools.augmentor import AugmentorBuilder
from torchvision.transforms.v2.functional import to_dtype

from torchvision import models


lt.monkey_patch()
lt.set_config(sci_mode=False)
torch.set_printoptions(sci_mode=False)


class ImgAugmentor(nn.Module):
    def __init__(
        self,
        tf,
        p_augment=False,
        preproc=True,
        norm_args=None,
        color_augment_args=None,
        use_fast_color_aug=False,
        clamp_values=True,
        train_mode=True,
        proc_device="cuda",
        target_device="cuda",
    ):
        super().__init__()
        self.transforms = tf
        self.clamp = clamp_values
        self.proc_device = proc_device
        self.target_device = target_device

        self.p_augment = p_augment
        self.preproc = preproc
        self.norm_args = norm_args
        self.color_augment_args = color_augment_args
        self.use_fast_color_aug = use_fast_color_aug
        self.train_mode = train_mode

        if self.color_augment_args is not None:
            self.color_augmentor = AugmentorBuilder.build(
                color_augment_args["method"],
                rng=color_augment_args["seed"],
                luminosity_threshold=color_augment_args["luminosity_threshold"],
                concentration_method=color_augment_args["concentration_method"],
                sigma_alpha=color_augment_args["sigma_alpha"],
                sigma_beta=color_augment_args["sigma_beta"],
                target_stain_idx=(0, 1),
                use_cache=True,
                cache_size_limit=-1,
                load_path=None,
            ).to(proc_device)
        else:
            self.color_augmentor = None

        if norm_args is not None:
            self.normalizer = NormalizerBuilder.build(
                norm_args["method"],
                concentration_method=norm_args["concentration_method"],
                use_cache=True,
            ).to(proc_device)

            # Fit to reference image
            ref_img_path = Path(
                Path(__file__).resolve().parent, "img_ref", norm_args["ref_img_name"]
            )
            assert ref_img_path.exists(), "Reference image not found"
            img = cv2.resize(
                cv2.cvtColor(cv2.imread(str(ref_img_path)), cv2.COLOR_BGR2RGB),
                (config.INPUT_SIZE[0], config.INPUT_SIZE[1]),
            )
            ref_image_tensor = (
                self.transforms["from_np.uint8_to_torch.float"](img)
                .unsqueeze(0)
                .to(proc_device)
            )
            self.normalizer.fit(ref_image_tensor)
        else:
            self.normalizer = None

    @torch.no_grad()
    def forward(self, x):
        # x: torch.uint8 [B x C x H x W]

        if x.device != self.proc_device:
            x = x.to(self.proc_device)

        if self.use_fast_color_aug and self.color_augmentor is None:
            x = self.transforms["c_aug"](x)

        if self.color_augmentor is not None:
            try:
                if self.train_mode:
                    x = self.transforms["rand_stain_color_aug"](
                        color_augmentor=self.color_augmentor
                    )(x)
                else:
                    x = self.color_augmentor(x)
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            except Exception as e:
                # x = self.transforms['c_aug'](x)
                pass

        if self.normalizer is not None:
            try:
                if self.train_mode:
                    x = self.transforms["rand_norm"](normalizer=self.normalizer)(x)
                else:
                    x = self.transforms["emphasize_white"]()(x)
                    x = self.normalizer(x)
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            except Exception as e:
                pass

        if self.p_augment:
            x = self.transforms["p_aug"](x)

        if self.preproc:
            # x = to_pil_image(x.squeeze(0))
            x = self.transforms["preproc"](x)
            if self.clamp:
                x = torch.clamp(x, -1.0, 1.0)

            # During self.preproc np_imgs are transormed to np_imgs x ∈ [-1, 1]
        else:
            x = to_dtype(x, torch.float32, scale=True)
            if self.clamp:
                x = torch.clamp(x, 0.0, 1.0)
            #
            # x = self.transforms[f'resize_to_tensor'](image=x['image'])
            # x = alb.pytorch.ToTensorV2()(image=x)['image'] / 255.0) # Convert pixel values to range [0, 1]

        if x.device != self.target_device:
            x = x.to(self.target_device)

        return x


class PatchClassifier(nn.Module):
    def __init__(
        self, model_name, pretrained, out_classes, conf_dropout_classifier=None
    ):
        super().__init__()

        # if 'torch-efficientnetv2_m' in config.PROFILE_ID:
        #     self.model = models.efficientnet_v2_m(weights='DEFAULT' if pretrained else None)
        #     if add_dropout_classifier:
        #         self.model.classifier = nn.Sequential(
        #             nn.Dropout(p=0.3, inplace=True),
        #             nn.Linear(in_features=1280, out_features=out_classes, bias=True)
        #         )

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=out_classes,
            features_only=False,
        )

        if conf_dropout_classifier is not None:
            assert conf_dropout_classifier["num_layers"] == len(
                conf_dropout_classifier["dropout_p"]
            ), "Number of dropout probabilities must match number of layers"

            layers = []
            in_features = self.model.classifier.in_features
            for i in range(conf_dropout_classifier["num_layers"]):
                layers.append(
                    nn.Dropout(p=conf_dropout_classifier["dropout_p"][i], inplace=True)
                )
                layers.append(
                    nn.Linear(
                        in_features=in_features,
                        out_features=(
                            out_classes
                            if i == (conf_dropout_classifier["num_layers"] - 1)
                            else in_features // 2
                        ),
                        bias=True,
                    )
                )
                in_features //= 2
            self.model.classifier = nn.Sequential(*layers)

    def forward(self, images):
        output = self.model(images)
        return output


class ModelTrainer:
    def __init__(
        self,
        model,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        train_dataloader=None,
        valid_dataloader=None,
        test_dataloader=None,
        val_metric="avg_loss",
        device=None,
    ):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.model = model.to(device=self.device)

        if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
            self.criterion = loss_fn
        elif isinstance(loss_fn, FocalLoss):
            softmax = torch.nn.Softmax(dim=-1)
            self.criterion = lambda logits, targets: loss_fn(softmax(logits), targets)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        validation_metric_dict = {
            "avg_loss": "min",
            "accuracy": "max",
            "precision": "max",
            "recall": "max",
            "roc_auc": "max",
        }
        assert (
            val_metric in validation_metric_dict.keys()
        ), f"Invalid validation metric {val_metric}. Choose from {list(validation_metric_dict.keys())}"
        self.val_metric = (val_metric, validation_metric_dict[val_metric])

        self.patience = config.EARLY_STOP_PATIENCE
        self.best_trained_model = {
            "score": float("inf") if self.val_metric[1] == "min" else 0.0,
            "model_state_dict": None,
            "current_patience": self.patience,
            "epoch": 0,
        }

    def reset_state_for_next_fold(self):
        # self.best_trained_model['score'] = float('inf') if self.val_metric[1] == "min" else 0.0
        # self.best_trained_model['model_state_dict'] = self.model.state_dict().copy()
        self.best_trained_model["current_patience"] = self.patience
        self.best_trained_model["epoch"] = 0

    def update_best_model(self, out_eval, epoch):
        self.best_trained_model["score"] = out_eval[self.val_metric[0]]
        # copy.deepcopy(self.model.state_dict())
        self.best_trained_model["model_state_dict"] = self.model.state_dict().copy()
        self.best_trained_model["current_patience"] = self.patience
        self.best_trained_model["epoch"] = epoch

    def handle_early_stopping(self, epoch):
        if self.patience is not None:
            self.best_trained_model["current_patience"] -= 1
            if self.best_trained_model["current_patience"] < 0:
                tqdm.write(
                    f"Early stopping at epoch {epoch}, taking best model at epoch {self.best_trained_model['epoch']}"
                )
                return True
        return False

    def test_step(self, imgs, targets):
        self.model.eval()
        logits = self.model(imgs)

        # probs = logits.sigmoid()
        # preds = (probs > 0.5).float()

        # preds_n_correct = (logits.argmax(1) == targets).type(torch.float).sum().item()
        (pred_logits, pred_idxs) = logits.max(1)

        return {"logits": logits, "preds": pred_idxs}

    def eval_step(self, imgs, targets):
        self.model.eval()

        if config.USE_AMP:
            with torch.amp.autocast(device_type=self.device, cache_enabled=True):
                logits = self.model(imgs)
                loss = self.criterion(logits, targets)
        else:
            logits = self.model(imgs)
            loss = self.criterion(logits, targets)

        # probs = logits.sigmoid()
        # preds = (probs > 0.5).float()

        # preds_n_correct = (logits.argmax(1) == targets).type(torch.float).sum().item()
        (pred_logits, pred_idxs) = logits.max(1)

        return {
            "loss": float(loss.item()),
            "preds": pred_idxs,
            "class_probs": F.softmax(logits, dim=1),
        }

    def evaluate(self, dataloader):
        total_loss = 0
        preds = []
        class_probs = []

        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(dataloader):
                if imgs.type != torch.FloatTensor:
                    imgs = imgs.type(torch.FloatTensor)
                if imgs.device != self.device:
                    imgs = imgs.to(device=self.device)

                if targets.type != torch.LongTensor:
                    targets = targets.type(torch.LongTensor)
                if targets.device != self.device:
                    targets = targets.to(device=self.device)

                out_eval_step = self.eval_step(imgs, targets)

                batch_loss = out_eval_step["loss"]
                total_loss += batch_loss

                preds.extend(out_eval_step["preds"].tolist())

                class_probs.append(out_eval_step["class_probs"])

        accuracy = multiclass_accuracy(
            torch.tensor(preds),
            torch.tensor(dataloader.dataset.targets),
            # torch.tensor(dataloader.dataset.dataset.targets)[dataloader.dataset.indices],
            num_classes=config.NUM_CLASSES,
            average="micro",
        )
        precision = multiclass_precision(
            torch.tensor(preds),
            torch.tensor(dataloader.dataset.targets),
            # torch.tensor(dataloader.dataset.dataset.targets)[dataloader.dataset.indices],
            num_classes=config.NUM_CLASSES,
            average="macro",
        )
        recall = multiclass_recall(
            torch.tensor(preds),
            torch.tensor(dataloader.dataset.targets),
            # torch.tensor(dataloader.dataset.dataset.targets)[dataloader.dataset.indices],
            num_classes=config.NUM_CLASSES,
            average="macro",
        )
        roc_auc = multiclass_auroc(
            torch.tensor(torch.cat(class_probs, dim=0).numpy(force=True)),
            torch.tensor(dataloader.dataset.targets),
            # torch.tensor(dataloader.dataset.dataset.targets)[dataloader.dataset.indices],
            num_classes=config.NUM_CLASSES,
            average="macro",
        )
        # stats = multiclass_stat_scores(torch.tensor(preds), torch.tensor(dataloader.dataset.dataset.targets)[dataloader.dataset.indices], num_classes=config.NUM_CLASSES, average='macro')

        # print (f"Accuracy: {accuracy}")
        # print (f"Precision: {precision}")
        # print (f"Recall: {recall}")
        # print (f"ROC AUC: {roc_auc}")

        # precision, recall, _, _ = precision_recall_fscore_support(torch.tensor(dataloader.dataset.dataset.targets)[dataloader.dataset.indices],, preds, average='macro')
        # roc_auc = roc_auc_score(torch.tensor(dataloader.dataset.dataset.targets)[dataloader.dataset.indices],, class_probs.numpy(force=True), multi_class='ovr')

        return {
            "avg_loss": total_loss / len(dataloader),
            "accuracy": float(accuracy.item()),
            "precision": float(precision.item()),
            "recall": float(recall.item()),
            "roc_auc": float(roc_auc.item()),
            "preds": preds,
        }

    def train_step(self, imgs, targets, grad_step=True, accumulate_norm_factor=1):
        self.model.train()

        # Forward
        if config.USE_AMP:
            with torch.amp.autocast(device_type=self.device, cache_enabled=True):
                logits = self.model(imgs)
                loss = self.criterion(logits, targets)
        else:
            logits = self.model(imgs)
            loss = self.criterion(logits, targets)

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
        save_final=False,
    ):
        if train_dataloader is None:
            train_dataloader = self.train_dataloader
        if valid_dataloader is None:
            valid_dataloader = self.valid_dataloader

        self.model = self.model.to(self.device)
        for epoch in trange(1, config.MAX_NUM_EPOCHS + 1, desc="Epochs"):
            train_loss = 0  # train epoch loss
            for batch_idx, (imgs, targets) in enumerate(
                tqdm(train_dataloader, desc="Batches", leave=False)
            ):
                if imgs.type != torch.FloatTensor:
                    imgs = imgs.type(torch.FloatTensor)
                if imgs.device != self.device:
                    imgs = imgs.to(device=self.device)

                if targets.type != torch.LongTensor:
                    targets = targets.type(torch.LongTensor)
                if targets.device != self.device:
                    targets = targets.to(device=self.device)

                do_grad_step = ((batch_idx + 1) % accumulate_grad_batches == 0) or (
                    batch_idx + 1 == len(train_dataloader)
                )

                # train batch loss
                train_loss += self.train_step(
                    imgs,
                    targets,
                    grad_step=do_grad_step,
                    accumulate_norm_factor=accumulate_grad_batches,
                )

            out_eval = self.evaluate(valid_dataloader)

            loss = out_eval["avg_loss"]
            accuracy = out_eval["accuracy"]
            precision = out_eval["precision"]
            recall = out_eval["recall"]
            roc_auc = out_eval["roc_auc"]

            tqdm.write(
                f"""
            Epoch [{epoch}]: \t 
            train loss = [{train_loss:0.5f}] \t 
            val loss = [{loss:0.5f}] \t 
            val accuracy = [{(accuracy * 100):0.2f}%] \t
            val precision = [{(precision * 100):0.2f}%] \t
            val recall = [{(recall * 100):0.2f}%] \t
            val ROC AUC = [{(roc_auc * 100):0.2f}%] \t
            """
            )

            if math.isnan(float(loss)) or math.isnan(float(train_loss)):
                tqdm.write(
                    f"Early stopping at epoch due to NAN loss {epoch}, best model at epoch {self.best_trained_model['epoch']}"
                )
                break

            # Scheduler step
            if self.scheduler is not None:
                if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(loss)
                elif (
                    type(self.scheduler) == torch.optim.lr_scheduler.MultiStepLR
                    or torch.optim.lr_scheduler.StepLR
                ):
                    self.scheduler.step()

            # Tracking best score model
            if (
                self.val_metric[1] == "min"
                and out_eval[self.val_metric[0]] < self.best_trained_model["score"]
            ) or (
                self.val_metric[1] == "max"
                and out_eval[self.val_metric[0]] > self.best_trained_model["score"]
            ):
                self.update_best_model(out_eval, epoch)
            else:
                if self.handle_early_stopping(epoch):
                    break

        if save_final:
            if self.best_trained_model is not None:
                self.model.load_state_dict(self.best_trained_model["model_state_dict"])

            if self.test_dataloader is not None:
                data_mode = "test"
                print("Evaluating and saving model on test set")
            else:
                data_mode = "valid"
                print("Evaluating and saving model on validation set")

            out_eval = self.evaluate(
                self.test_dataloader
                if self.test_dataloader is not None
                else self.valid_dataloader
            )
            accuracy = out_eval["accuracy"]

            print(
                f"Saving model {config.PROFILE_ID}: \"{config.MODEL_NAME}\" initialized with \"{'pretrained' if config.PRETRAINED else 'random'}\" with {data_mode} set accuracy score = {(accuracy * 100):0.2f}"
            )
            self.save_state_dict(
                f"{'models'}/{config.PROFILE_ID}_{(accuracy * 100):0.2f}.pt"
            )
            self.save(
                f"{'saved_models'}/{config.PROFILE_ID}_{(accuracy * 100):0.2f}.pth"
            )

    def save_state_dict(self, fpath):
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        save_run_config(fpath, config)
        torch.save(self.model.state_dict(), fpath)

    def save(self, fpath):
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, fpath)

    def load_state(self, fpath):
        self.model.load_state_dict(torch.load(fpath))


if __name__ == "__main__":

    batch_augmentor = ImgAugmentor(
        config.ATF,
        p_augment=True,
        preproc=True,
        norm_args=None,
        color_augment_args=None,
        device="cpu",
    )

    model = PatchClassifier(
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        out_classes=config.NUM_CLASSES,
        add_dropout_classifier=True,
        batch_augmentor=batch_augmentor,
    ).eval()

    test_net(model, device="cpu", size=(3, 512, 512), n_batch=8, use_lt=True)
