from enum import Enum
from typing import Optional

from loguru import logger
from ssl_library.src.datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader

from src.perturbed_datasets.artefact_augmentation_dataset import (
    ArtefactAugmentationDataset,
)
from src.perturbed_datasets.blur_augment_dataset import BlurAugmentDataset
from src.perturbed_datasets.combined_dataset import CombinedDataset
from src.perturbed_datasets.label_error_dataset import LabelErrorDataset
from src.perturbed_datasets.nd_augment_dataset import NDAugmentDataset


class ErrorName(Enum):
    ARTEFACT_AUGMENT = "artefact_augment"
    AUG_AUGMENT = "aug_augment"
    LABEL_ERROR = "label_error"
    COMBINED_DATASET = "combined_dataset"
    BLUR_DATASET = "blur_dataset"


def get_dataset_errors(
    error_name: str,
    dataset: BaseDataset,
    batch_size: int = 128,
    transform=None,
    frac_error: float = 0.05,
    n_errors: Optional[int] = None,
    **kwargs,
):
    if error_name == ErrorName.ARTEFACT_AUGMENT:
        error_dataset = ArtefactAugmentationDataset(
            dataset=dataset,
            transform=transform,
            frac_error=frac_error,
            n_errors=n_errors,
            **kwargs,
        )

    elif error_name == ErrorName.AUG_AUGMENT:
        error_dataset = NDAugmentDataset(
            dataset=dataset,
            transform=transform,
            frac_error=frac_error,
            n_errors=n_errors,
            **kwargs,
        )

    elif error_name == ErrorName.LABEL_ERROR:
        error_dataset = LabelErrorDataset(
            dataset=dataset,
            transform=transform,
            frac_error=frac_error,
            n_errors=n_errors,
            **kwargs,
        )

    elif error_name == ErrorName.BLUR_DATASET:
        error_dataset = BlurAugmentDataset(
            dataset=dataset,
            transform=transform,
            frac_error=frac_error,
            n_errors=n_errors,
            **kwargs,
        )

    elif error_name == ErrorName.COMBINED_DATASET:
        error_dataset = CombinedDataset(
            dataset=dataset,
            transform=transform,
            frac_error=frac_error,
            n_errors=n_errors,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown error: {str(error_name)}")

    torch_dataset = DataLoader(
        error_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    logger.info(
        f"Loaded `{error_name.value}` errors which contains {len(torch_dataset)} "
        f"batches with a batch size of {batch_size}.\n"
    )
    error_dataset.info()
    return error_dataset, torch_dataset
