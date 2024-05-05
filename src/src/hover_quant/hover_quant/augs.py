from pathlib import Path
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from dotenv import dotenv_values

from hover_quant.profiles import PROFILES
from hover_quant.utils import wrap_transform_multichannel


ENV_CONFIG = dotenv_values(Path(Path(__file__).resolve().parent, ".env"))
PROFILE_ID = ENV_CONFIG["PROFILE_ID"]

ATF = {
    "resize_to_tensor": alb.Compose(
        [
            alb.Resize(
                width=PROFILES[PROFILE_ID]["INPUT_SIZE"][0],
                height=PROFILES[PROFILE_ID]["INPUT_SIZE"][1],
            ),
            ToTensorV2(),
        ]
    ),
    "hover_transform": wrap_transform_multichannel(
        alb.Compose(
            [
                alb.VerticalFlip(p=0.5),
                alb.HorizontalFlip(p=0.5),
                alb.Rotate(180, p=0.5),
                alb.Transpose(p=0.5),
                # alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3),
                alb.Compose(
                    [
                        alb.ShiftScaleRotate(
                            shift_limit=0, scale_limit=0.5, rotate_limit=0, p=1
                        ),  # [0.5; 1.5]
                        # alb.Crop(x_min=5, y_min=5, x_max=PROFILES[PROFILE_ID]["INPUT_SIZE"][0]-5, y_max=PROFILES[PROFILE_ID]["INPUT_SIZE"][1]-5),
                        # alb.Resize(height=PROFILES[PROFILE_ID]["INPUT_SIZE"][1], width=PROFILES[PROFILE_ID]["INPUT_SIZE"][0])
                    ]
                ),
                alb.OneOf(
                    [
                        # alb.OpticalDistortion(distort_limit=1.0), # fish-eye effect
                        alb.GridDistortion(num_steps=1, distort_limit=0.5),
                        alb.ElasticTransform(alpha=1),
                    ],
                    p=0.2,
                ),
                alb.OneOf(
                    [
                        alb.HueSaturationValue(10, 15, 10),
                        alb.RandomBrightnessContrast(),
                    ],
                    p=0.4,
                ),
                alb.OneOf(
                    [
                        alb.GaussNoise(),
                        alb.MotionBlur(),
                        alb.MedianBlur(blur_limit=5),
                        alb.GaussianBlur(),
                    ],
                    p=0.3,
                ),
            ],
            additional_targets={
                f"mask{i}": "mask" for i in range(PROFILES[PROFILE_ID]["NUM_CLASSES"])
            },
        )
    ),
}
