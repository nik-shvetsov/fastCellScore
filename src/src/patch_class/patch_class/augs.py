import importlib
from pathlib import Path
from dotenv import dotenv_values
import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from torchvision.transforms import v2 as tv2
from torchvision.transforms import InterpolationMode
import timm
from timm.data import resolve_data_config, resolve_model_data_config
from timm.data.transforms_factory import create_transform
from patch_class.profiles import PROFILES
from torchvision import models
import kornia
import kornia.augmentation as K
from kornia.constants import Resample, SamplePadding
from kornia.color import rgb_to_grayscale, rgb_to_lab
from kornia.filters import median_blur
import random
from functools import partial


ENV_CONFIG = dotenv_values(Path(Path(__file__).resolve().parent, ".env"))
PROFILE_ID = ENV_CONFIG["PROFILE_ID"]

# 0: InterpolationMode.NEAREST,
# 2: InterpolationMode.BILINEAR,
# 3: InterpolationMode.BICUBIC,
# 4: InterpolationMode.BOX,
# 5: InterpolationMode.HAMMING,
# 1: InterpolationMode.LANCZOS,

# if 'torch-efficientnetv2_l' in PROFILE_ID:
#     preproc_transforms = transforms.Compose([ # models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms
#         transforms.Resize(480, interpolation=3),
#         transforms.CenterCrop(480),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

ref_transforms = create_transform(
    **resolve_model_data_config(
        timm.create_model(
            PROFILES[PROFILE_ID]["MODEL_NAME"],
            pretrained=PROFILES[PROFILE_ID]["PRETRAINED"],
            num_classes=PROFILES[PROFILE_ID]["NUM_CLASSES"],
        )
    ),
    is_training=False,
)

resize_size = None
for transform in ref_transforms.transforms:
    if isinstance(transform, transforms.Resize):
        # print (f"INFO: Found resize transform - {transform.size} - using for Resize() and CenterCrop()")
        resize_size = ref_transforms.transforms[0].size
        break

if resize_size is None:
    raise ValueError("Resize transform not found in model's transforms")

preproc_transforms = K.AugmentationSequential(
    K.Resize(
        size=(resize_size, resize_size),
        antialias=True,
        resample=Resample.BICUBIC.name,
    ),
    K.CenterCrop(size=(resize_size, resize_size)),
    K.Normalize(mean=torch.tensor([0.5, 0.5, 0.5]), std=torch.tensor([0.5, 0.5, 0.5])),
    same_on_batch=True,
)


class RandomStainColorAug(kornia.augmentation.AugmentationBase2D):
    def __init__(self, color_augmentor=None, same_on_batch=False, p=1.0, keepdim=False):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.color_augmentor = color_augmentor

    def apply_transform(self, input, params, flags, transform=None):
        return self.color_augmentor(input)


class RandomNormalizerPlusEmphasize(kornia.augmentation.AugmentationBase2D):
    def __init__(self, normalizer=None, same_on_batch=False, p=1.0, keepdim=False):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.normalizer = normalizer

        self.threshold = 0.8
        self.value = 1.0
        self.blur_kernel = (5, 5)

    def apply_transform(self, input, params, flags, transform=None):
        binary_masks = (
            rgb_to_grayscale(median_blur(input, self.blur_kernel)) > self.threshold
        ).float()
        return self.normalizer(input * (1.0 - binary_masks) + self.value * binary_masks)


class EmphasizeWhite(kornia.augmentation.AugmentationBase2D):
    def __init__(
        self,
        threshold=0.8,
        value=1.0,
        blur_kernel=(5, 5),
        same_on_batch=True,
        p=1.0,
        keepdim=True,
    ):
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)
        self.threshold = threshold
        self.value = value
        self.blur_kernel = blur_kernel

    def apply_transform(self, input, params, flags, transform=None):
        input_blured = median_blur(input, self.blur_kernel)
        gray_masks = rgb_to_grayscale(input_blured)
        binary_masks = (gray_masks > self.threshold).float()
        return input * (1.0 - binary_masks) + self.value * binary_masks

        # masked_batch = torch.where(masked_batch == 0.0, 1.0, masked_batch)


ATF = {
    "emphasize_white": partial(
        EmphasizeWhite,
        threshold=0.8,
        value=1.0,
        blur_kernel=(5, 5),
        same_on_batch=True,
        p=1.0,
    ),
    "rand_norm": partial(RandomNormalizerPlusEmphasize, same_on_batch=False, p=0.5),
    "rand_stain_color_aug": partial(RandomStainColorAug, same_on_batch=False, p=0.8),
    "preproc": preproc_transforms,
    "p_aug_alb": alb.Compose(
        [
            alb.OneOf(
                [
                    alb.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.2,
                        rotate_limit=45,
                    ),
                    alb.RandomResizedCrop(
                        width=PROFILES[PROFILE_ID]["INPUT_SIZE"][0]
                        - (PROFILES[PROFILE_ID]["INPUT_SIZE"][0] // 4),
                        height=PROFILES[PROFILE_ID]["INPUT_SIZE"][1]
                        - (PROFILES[PROFILE_ID]["INPUT_SIZE"][1] // 4),
                        scale=(0.6, 1.2),
                        ratio=(0.75, 1.3333333333333333),
                    ),
                ],
                p=0.8,
            ),
            # 512 x 512
            alb.Resize(
                height=PROFILES[PROFILE_ID]["INPUT_SIZE"][1],
                width=PROFILES[PROFILE_ID]["INPUT_SIZE"][0],
            ),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(180, p=0.5),
            alb.Transpose(p=0.5),
            # alb.OneOf([
            #     alb.OpticalDistortion(distort_limit=0.5),
            #     alb.GridDistortion(num_steps=1, distort_limit=0.3),
            #     alb.ElasticTransform(alpha=3),
            # ], p=0.7),
            # alb.OneOf([
            #     alb.HueSaturationValue(10, 15, 10),
            #     alb.RandomBrightnessContrast(),
            # ], p=0.2),
            alb.OneOf(
                [
                    alb.GaussNoise(),
                    alb.MotionBlur(),
                    alb.MedianBlur(blur_limit=5),
                    alb.GaussianBlur(),
                ],
                p=0.2,
            ),
            ToTensorV2(),
        ]
    ),
    "p_aug_tv2": tv2.Compose(
        [
            tv2.RandomApply(
                [
                    tv2.RandomChoice(
                        [
                            tv2.RandomAffine(
                                degrees=45, translate=(0.0625, 0.0625), scale=(0.8, 1.2)
                            ),
                            tv2.RandomResizedCrop(
                                size=(
                                    PROFILES[PROFILE_ID]["INPUT_SIZE"][0]
                                    - (PROFILES[PROFILE_ID]["INPUT_SIZE"][0] // 4),
                                    PROFILES[PROFILE_ID]["INPUT_SIZE"][1]
                                    - (PROFILES[PROFILE_ID]["INPUT_SIZE"][1] // 4),
                                ),
                                scale=(0.6, 1.2),
                                ratio=(0.75, 1.3333333333333333),
                                antialias=True,
                            ),
                        ]
                    )
                ],
                p=0.8,
            ),
            # 512 x 512
            tv2.Resize(
                size=(
                    PROFILES[PROFILE_ID]["INPUT_SIZE"][1],
                    PROFILES[PROFILE_ID]["INPUT_SIZE"][0],
                ),
                antialias=True,
            ),
            tv2.RandomHorizontalFlip(p=0.5),
            tv2.RandomVerticalFlip(p=0.5),
            tv2.RandomRotation(degrees=360),
            tv2.RandomApply([tv2.GaussianBlur(kernel_size=5)], p=0.2),
        ]
    ),
    "c_aug": K.AugmentationSequential(
        K.AugmentationSequential(
            random.choice(
                [
                    K.ColorJiggle(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.02,
                    ),
                    K.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.02,
                    ),
                    K.RandomHue(
                        hue=(-0.02, 0.02),
                    ),
                ]
            ),
            random_apply=1,
            random_apply_weights=[0.7],
            same_on_batch=False,
        ),
        K.RandomGrayscale(p=0.1),
        same_on_batch=False,
    ),
    "p_aug": K.AugmentationSequential(
        K.AugmentationSequential(
            random.choice(
                [
                    K.RandomAffine(
                        degrees=45,
                        translate=(0.0625, 0.0625),
                        scale=(0.8, 1.2),
                        resample=Resample.BICUBIC.name,
                        padding_mode=SamplePadding.REFLECTION.name,
                    ),
                    K.RandomResizedCrop(
                        size=(
                            PROFILES[PROFILE_ID]["INPUT_SIZE"][0]
                            - (PROFILES[PROFILE_ID]["INPUT_SIZE"][0] // 4),
                            PROFILES[PROFILE_ID]["INPUT_SIZE"][1]
                            - (PROFILES[PROFILE_ID]["INPUT_SIZE"][1] // 4),
                        ),
                        scale=(0.6, 1.2),
                        ratio=(0.75, 1.3333333333333333),
                        resample=Resample.BICUBIC.name,
                    ),
                ]
            ),
            random_apply=1,
            random_apply_weights=[0.8],
            same_on_batch=False,
        ),
        K.Resize(
            size=(
                PROFILES[PROFILE_ID]["INPUT_SIZE"][1],
                PROFILES[PROFILE_ID]["INPUT_SIZE"][0],
            ),
            antialias=True,
            # resample=Resample.BICUBIC.name,
        ),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=360, p=0.5),
        K.AugmentationSequential(
            random.choice(
                [
                    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
                    K.RandomMedianBlur(kernel_size=(3, 3), p=0.2),
                    K.RandomMotionBlur(
                        kernel_size=(3, 3),
                        angle=(0, 360),
                        direction=0.0,
                        # resample=Resample.BICUBIC.name,
                        p=0.2,
                    ),
                ]
            ),
            random_apply=1,
            random_apply_weights=[0.2],
            same_on_batch=False,
        ),
        # same_on_batch=False,
    ),
    "resize_to_tensor": alb.Compose(
        [
            alb.Resize(
                height=PROFILES[PROFILE_ID]["INPUT_SIZE"][1],
                width=PROFILES[PROFILE_ID]["INPUT_SIZE"][0],
            ),
            ToTensorV2(),
        ]
    ),
    # "resize": K.Resize(
    #     size=(PROFILES[PROFILE_ID]["INPUT_SIZE"][1], PROFILES[PROFILE_ID]["INPUT_SIZE"][0]),
    #     antialias=True,
    #     # resample=Resample.BICUBIC.name,
    # ),
    "resize_for_infer": tv2.Resize(
        size=(
            PROFILES[PROFILE_ID]["INPUT_SIZE"][1],
            PROFILES[PROFILE_ID]["INPUT_SIZE"][0],
        ),
        antialias=True,
        interpolation=3,  # BICUBIC
    ),
    "from_np.uint8_to_torch.uint8": tv2.Compose(
        [
            tv2.ToImage(),
            tv2.ToDtype(torch.uint8, scale=True),
        ]
    ),
    "from_np.uint8_to_torch.float": tv2.Compose(
        [
            tv2.ToImage(),
            tv2.ToDtype(torch.float32, scale=True),
        ]
    ),
    "from_torch.float_to_torch.uint8": tv2.ToDtype(torch.uint8, scale=True),
}

if __name__ == "__main__":

    print(
        f"Preprocessing function for <{PROFILES[PROFILE_ID]['MODEL_NAME']}> encoder and {'pretrained' if PROFILES[PROFILE_ID]['PRETRAINED'] else 'random'} weights: \n{ATF['preproc']}"
    )
