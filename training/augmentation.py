"""Train vs eval image transforms for Img2GPS (Group 5).

Policy is documented in docs/DATA.md. Submitted preprocess.py should match eval transforms.
"""

from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Logged in RESULTS.md / report for reproducibility.
AUG_POLICY_ID = "group5_randa_m7"


def build_eval_transforms(
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """Deterministic 224×224 pipeline: resize short side 256, center crop, normalize."""
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_train_transforms(
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 7,
) -> T.Compose:
    """Advanced train-only augmentation: crop, flip, blur, RandAugment, erasing, normalize."""
    return T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.1,
            ),
            T.RandAugment(
                num_ops=randaugment_num_ops,
                magnitude=randaugment_magnitude,
            ),
            T.ToTensor(),
            T.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_train_transforms_trivial_wide(
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """Alternative policy: TrivialAugmentWide instead of RandAugment (see DATA.md ablations)."""
    return T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.1,
            ),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )
