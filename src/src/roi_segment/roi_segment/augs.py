from pathlib import Path
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torchvision.transforms as transforms
from dotenv import dotenv_values
from roi_segment.profiles import PROFILES


ENV_CONFIG = dotenv_values(Path(Path(__file__).resolve().parent, ".env"))
PROFILE_ID = ENV_CONFIG["PROFILE_ID"]


# A.Compose([
#     A.ShiftScaleRotate(p=0.5),
#     A.RandomResizedCrop(height=512, width=512, p=0.5),
#     A.ElasticTransform(p=0.5),
#     A.RandomBrightnessContrast(p=0.5),
#     A.HueSaturationValue(p=0.5),
#     A.RandomGamma(p=0.5),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
#     A.GaussNoise(p=0.5),
#     A.GaussianBlur(p=0.5),
#     A.Cutout(p=0.5),
#     A.CoarseDropout(p=0.5)
# ], p=1)


ATF = {
    "aug": alb.Compose(
        [
            # Tests
            # alb.CoarseDropout(max_height=300, max_width=300, min_holes=None, min_height=100, min_width=100, mask_fill_value=False, p=1.0),
            # alb.Cutout(max_h_size=300, max_w_size=300, p=1.0),
            # alb.HueSaturationValue(p=1),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(180, p=0.5),
            alb.Transpose(p=0.5),
            alb.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.3
            ),  # Randomly apply affine transforms
            alb.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50, p=0.2
            ),  # Apply elastic transformations
            alb.RandomResizedCrop(
                height=PROFILES[PROFILE_ID]["INPUT_SIZE"][0],
                width=PROFILES[PROFILE_ID]["INPUT_SIZE"][1],
                scale=(0.8, 1.0),
                ratio=(1.0, 1.0),
                p=0.2,
            ),  # Randomly crop and resize
            alb.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.3
            ),  # Randomly change brightness and contrast
            alb.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3
            ),  # Randomly jitter color properties
            alb.GaussNoise(var_limit=(10.0, 50.0), p=0.1),  # Add Gaussian noise
        ]
    ),
    "_aug": alb.Compose(
        [
            alb.Resize(
                height=PROFILES[PROFILE_ID]["INPUT_SIZE"][0],
                width=PROFILES[PROFILE_ID]["INPUT_SIZE"][1],
            ),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.Rotate(180, p=0.8),
            alb.Transpose(p=0.5),
            # alb.RandomRotate90(p=1),
            # alb.OneOf([
            #     alb.OpticalDistortion(distort_limit=1.0),
            #     alb.GridDistortion(num_steps=5, distort_limit=1.),
            #     alb.ElasticTransform(alpha=3),
            # ], p=0.2),
            # alb.OneOf([
            #     alb.HueSaturationValue(10, 15, 10),
            #     alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            #     alb.RandomBrightnessContrast(),
            # ], p=0.3),
            # alb.OneOf([
            #     alb.GaussNoise(),
            #     alb.GaussianBlur(),
            #     # alb.MotionBlur(),
            #     # alb.MedianBlur(blur_limit=5),
            # ], p=0.05),
        ]
    ),
    "preproc": transforms.Compose(
        [
            # alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
            get_preprocessing_fn(
                PROFILES[PROFILE_ID]["ENCODER_NAME"],
                pretrained=PROFILES[PROFILE_ID]["WEIGHTS"],
            ),
        ]
    ),
    "resize_to_tensor": alb.Compose(
        [
            alb.Resize(
                height=PROFILES[PROFILE_ID]["INPUT_SIZE"][0],
                width=PROFILES[PROFILE_ID]["INPUT_SIZE"][1],
            ),
            ToTensorV2(),
        ]
    ),
}

if __name__ == "__main__":
    print(
        f'Preprocessing function for {PROFILES[PROFILE_ID]["ENCODER_NAME"]} encoder and initial weights {PROFILES[PROFILE_ID]["WEIGHTS"]}: \n{get_preprocessing_fn(PROFILES[PROFILE_ID]["ENCODER_NAME"], pretrained=PROFILES[PROFILE_ID]["WEIGHTS"])}'
    )
