import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchmetrics.functional import dice
from segmentation_models_pytorch.metrics import iou_score

import matplotlib.pyplot as plt
from pathlib import Path
import lovely_tensors as lt
from pprint import pprint
from tqdm import tqdm, trange

#############################
from roi_segment.utils import test_net, save_run_config, inspect_model

#############################
import roi_segment.config as config


lt.monkey_patch()
lt.set_config(sci_mode=False)
torch.set_printoptions(sci_mode=False)


class ROIDetector(nn.Module):
    def __init__(
        self,
        smp_model,
        encoder_name,
        in_channels,
        out_classes,
        init_weights,
        model_params={},
    ):
        super().__init__()

        self.model = smp_model(
            encoder_name=encoder_name,
            encoder_weights=init_weights,
            in_channels=in_channels,
            classes=out_classes,
            **model_params,
        )
        # self.model.encoder
        # self.model.decoder
        # self.model.segmentation_head
        # self.model.classification_head

    def forward(self, image):
        mask = self.model(image)
        return mask

    def predict(self, img, device, threshold=0.5, fname=None, return_sigmoid=False):
        """
        Img should be already preprocessed and transformed
        """

        self.model = self.model.to(device)
        self.model.eval()

        with torch.no_grad():  # torch.inference_mode():
            # Single image
            img = img.unsqueeze(0).type(torch.FloatTensor).to(device=device)

            logits_masks = self.model(img)

            prob_masks = logits_masks.sigmoid()
            if return_sigmoid:
                return prob_masks.squeeze()

            pred_masks = (prob_masks > threshold).float().squeeze()

            p_img = (
                img.squeeze().cpu().numpy().transpose(1, 2, 0)
                if device == "cuda"
                else img.squeeze().numpy().transpose(1, 2, 0)
            )
            p_mask = (
                pred_masks.squeeze().cpu().numpy()
                if device == "cuda"
                else pred_masks.squeeze().numpy()
            )

            #
            if fname is not None:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(p_img)
                plt.subplot(1, 2, 2)
                plt.imshow(p_mask)
                plt.savefig(fname)

            return p_mask


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
        device=None,
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

        # self.accelerator = Accelerator(fp16=training_args.fp16)
        # model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

        # if config.EARLY_STOP:
        #     self.patience = config.PATIENCE
        #     self.best_trained_model = {
        #         "score": 0,
        #         "state_dict": None,
        #         "current_pateince": self.patience,
        #     }

    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def test_step(self, imgs, masks):
        self.model.eval()
        logits_masks = self.model(imgs)

        prob_masks = logits_masks.sigmoid()
        pred_masks = (prob_masks > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_masks.long(), masks.long(), mode="binary"
        )

        return {
            "logits": logits_masks,
            "dice_score": dice(logits_masks, masks, average="micro"),
            "iou_score": iou_score(tp, fp, fn, tn, reduction="micro"),
        }

    def eval_step(self, imgs, masks):
        self.model.eval()

        if config.USE_AMP:
            with torch.amp.autocast(device_type=config.ACCELERATOR, cache_enabled=True):
                logits_masks = self.model(imgs)
                loss = self.criterion(logits_masks, masks)
        else:
            logits_masks = self.model(imgs)
            loss = self.criterion(logits_masks, masks)

        prob_masks = logits_masks.sigmoid()
        pred_masks = (prob_masks > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_masks.long(), masks.long(), mode="binary"
        )
        # batch_tp: Tensor [batch_size, 1] i64

        return {
            "loss": float(loss.item()),
            "dice_score": dice(logits_masks, masks, average="micro"),
            "iou_score": iou_score(tp, fp, fn, tn, reduction="micro"),
            # "logits": logits_masks.detach().cpu(),
        }

    def evaluate(self, dataloader):
        total_loss = 0
        dice_sum = 0
        iou_sum = 0

        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(dataloader):
                imgs = imgs.type(torch.FloatTensor).to(device=self.device)
                masks = masks.to(device=self.device)

                out_eval_step = self.eval_step(imgs, masks)

                batch_loss = out_eval_step["loss"]
                dice_score = out_eval_step["dice_score"]
                iou_score = out_eval_step["iou_score"]
                # logits_masks = out_eval_step["logits"]

                total_loss += batch_loss
                dice_sum += dice_score
                iou_sum += iou_score

        return {
            "avg_loss": total_loss / len(dataloader),
            "avg_dice": dice_sum / len(dataloader),
            "avg_iou": iou_sum / len(dataloader),
        }

    def train_step(self, imgs, masks, grad_step=True, accumulate_norm_factor=1):
        # Grad accumulation: https://kozodoi.me/blog/20210219/gradient-accumulation

        self.model.train()

        # Forward
        if config.USE_AMP:
            with torch.amp.autocast(device_type=config.ACCELERATOR, cache_enabled=True):
                logits_masks = self.model(imgs)
                loss = self.criterion(logits_masks, masks)
        else:
            logits_masks = self.model(imgs)
            loss = self.criterion(logits_masks, masks)

        # # debug
        # print (logits_masks[0][0])
        # plt.imshow(logits_masks[0][0].cpu().detach().numpy())
        # plt.savefig('act.png')
        # sys.exit(0)

        if config.USE_GRAD_SCALER:
            # accelerator.backward(loss)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # accelerator.backward(loss)
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
            for batch_idx, (imgs, masks) in enumerate(
                tqdm(train_dataloader, desc="Batches", leave=False)
            ):
                imgs = imgs.type(torch.FloatTensor).to(device=self.device)
                masks = masks.to(device=self.device)
                # targets = targets.type(torch.LongTensor).to(device=self.device)
                do_grad_step = ((batch_idx + 1) % accumulate_grad_batches == 0) or (
                    batch_idx + 1 == len(train_dataloader)
                )

                # train batch loss
                train_loss += self.train_step(
                    imgs,
                    masks,
                    grad_step=do_grad_step,
                    accumulate_norm_factor=accumulate_grad_batches,
                )

            out_eval = self.evaluate(valid_dataloader)

            loss = out_eval["avg_loss"]
            dice_score = out_eval["avg_dice"]
            iou_score = out_eval["avg_iou"]

            tqdm.write(
                f"""
            Epoch [{epoch}]: \t 
            train loss = [{train_loss:0.5f}] \t 
            val loss = [{loss:0.5f}] \t 
            val dice score = [{(dice_score * 100):0.2f}%] \t 
            val iou score = [{(iou_score * 100):0.2f}%] \t
            """
            )

            # Scheduler step
            if self.scheduler is not None:
                if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    self.scheduler.step(loss)
                elif type(self.scheduler) == torch.optim.lr_scheduler.MultiStepLR:
                    self.scheduler.step()

            # Early stopping
            if dice_score > self.best_trained_model["score"]:
                self.best_trained_model["score"] = dice_score
                self.best_trained_model["model_state_dict"] = (
                    self.model.state_dict().copy()
                )
                self.best_trained_model["current_patience"] = self.patience
            # Early stopping
            else:
                if self.patience is not None:
                    self.best_trained_model["current_patience"] -= 1
                    if self.best_trained_model["current_patience"] < 0:
                        tqdm.write(f"Early stopping at epoch {epoch}")
                        break

        if save_final:
            if self.best_trained_model is not None:
                self.model.load_state_dict(self.best_trained_model["model_state_dict"])

            out_eval = self.evaluate(
                self.test_dataloader
                if self.test_dataloader is not None
                else self.valid_dataloader
            )
            dice_score = out_eval["avg_dice"]
            iou_score = out_eval["avg_iou"]
            print(
                f'Saving {config.PROFILE_ID}: encoder "{config.ENCODER_NAME}" initialized with "{config.WEIGHTS}" with test set dice score = {(dice_score * 100):0.2f}'
            )

            # if config.WEIGHTS != 'imagenet':
            #     save_str = f'{config.ENCODER_NAME}_{config.WEIGHTS}_{(dice_score * 100):0.2f}.pt'
            # else:
            #     save_str = f'{config.ENCODER_NAME}_{(dice_score * 100):0.2f}.pt'
            save_str_dict = f"{config.PROFILE_ID}_{(dice_score * 100):0.2f}.pt"
            save_str_model = f"{config.PROFILE_ID}_{(dice_score * 100):0.2f}.pth"

            print(f"Model file: {save_str}")
            self.save_state_dict(f"{config.SAVE_DIR}/{save_str_dict}")
            self.save(f"{config.SAVE_DIR}/{save_str_model}")

    def save_state_dict(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_run_config(path, config)
        torch.save(self.model.state_dict(), path)

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def _vis_testloader(self, pt_name=None, threshold=0.5, fname="out_train_vis.png"):
        if self.test_dataloader is None:
            raise Exception("No test dataloader provided")
        if pt_name is not None:
            self.load(pt_name)
        self.model = self.model.to(self.device)
        with torch.no_grad():
            _sum_dice = 0
            for batch_idx, (imgs, masks) in enumerate(self.test_dataloader):
                imgs = imgs.type(torch.FloatTensor).to(device=self.device)
                masks = masks.to(device=self.device)
                out_test_step = self.test_step(imgs, masks)
                logits_masks = out_test_step["logits"]
                # dice_score = out_test_step["dice_score"]
                # iou_score = out_test_step["iou_score"]

                prob_masks = logits_masks.sigmoid()
                pred_masks = (prob_masks > threshold).float()

                for image, gt_mask, pr_mask in zip(imgs, masks, pred_masks):
                    plt.figure(figsize=(10, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
                    plt.title("Image")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(gt_mask.cpu().squeeze().numpy())
                    plt.title("Ground truth")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(pr_mask.cpu().squeeze().numpy())
                    plt.title("Prediction")
                    plt.axis("off")

                    plt.savefig(fname)
                    break
                break


if __name__ == "__main__":
    model = (
        ROIDetector(
            smp_model=config.SMP_MODEL,
            encoder_name=config.ENCODER_NAME,
            in_channels=config.IN_CHANNELS,
            out_classes=config.NUM_CLASSES,
            init_weights=config.WEIGHTS,
            model_params=config.MODEL_PARAMS,
        )
        .model.eval()
        .to(config.ACCELERATOR)
    )

    # inspect_model(model, output='state')
    test_net(model, size=(3, 512, 512), n_batch=8, use_lt=True)
