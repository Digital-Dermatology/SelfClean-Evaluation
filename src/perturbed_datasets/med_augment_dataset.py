import copy
import math
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from PIL import Image
from ssl_library.src.datasets.base_dataset import BaseDataset
from torchvision import transforms

from ..perturbed_datasets.base_contamination_dataset import BaseContaminationDataset


class NDAugmentDataset(BaseContaminationDataset):
    def __init__(
        self,
        dataset: BaseDataset,
        frac_error: float = 0.1,
        n_errors: Optional[int] = None,
        transform=None,
        name: Optional[str] = None,
        deterministic: bool = True,
        reference_image_size: int = 512,
        **kwargs,
    ):
        super().__init__()
        aug_transform = transforms.Compose(
            [
                transforms.Resize(size=reference_image_size),
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=(0, 180))],
                    p=0.5,
                ),
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 0.9)),
                transforms.RandomApply([transforms.Pad(3)], p=0.5),
                transforms.RandomApply(
                    [transforms.GaussianBlur(5)],
                    p=0.5,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.name = name if name is not None else type(self).__name__
        self.dataset_orig = copy.copy(dataset)
        self.dataset_aug = copy.copy(dataset)
        # set the correct augmentations
        self.dataset_aug.transform = aug_transform
        self.dataset_orig.transform = None
        self.transform = transform
        # random sample to augment
        if n_errors is not None:
            n_samples = n_errors
        else:
            n_samples = math.ceil(frac_error * len(self.dataset_orig))
        idx_range = np.arange(0, len(self.dataset_orig))
        self.list_aug_idx = np.random.choice(idx_range, size=n_samples, replace=False)
        # transform the random sample
        if deterministic:
            self.tmp_aug_path = Path(tempfile.mkdtemp())
            self.aug_samples = []
            for idx in self.list_aug_idx:
                rets = self.dataset_aug[idx]
                image = rets[0]
                img_path = self.tmp_aug_path / f"{idx}.jpg"
                image.save(img_path)
                rets = (img_path,) + rets[1:]
                self.aug_samples.append(rets)
        # global configs
        self.deterministic = deterministic
        self.classes = self.dataset_orig.classes
        self.n_classes = self.dataset_orig.n_classes

    def __len__(self):
        return len(self.dataset_orig) + len(self.list_aug_idx)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < len(self.dataset_orig):
            rets = self.dataset_orig[idx]
            image = rets[0]
        else:
            idx = idx - len(self.dataset_orig)
            if self.deterministic:
                rets = self.aug_samples[idx]
                image = Image.open(rets[0])
                image = image.convert("RGB")
            else:
                idx = self.list_aug_idx[idx]
                rets = self.dataset_aug[idx]
                image = rets[0]
        if self.transform is not None:
            image = self.transform(image)
        rets = (image,) + rets[1:]
        return rets

    def cleanup(self):
        if self.deterministic:
            shutil.rmtree(self.tmp_aug_path)

    def info(self):
        logger.info(f"Name of dataset: {self.name}")
        logger.info(f"Original datset size: {len(self.dataset_orig)}")
        logger.info(f"Augmented datset size: {len(self.list_aug_idx)}")
        logger.info(f"Combined datset size: {len(self)}")

    def get_errors(self):
        l_errors = []
        for idx in range(len(self)):
            if idx >= len(self.dataset_orig):
                dup_idx = idx
                idx = idx - len(self.dataset_orig)
                orig_idx = self.list_aug_idx[idx]
                l_errors.append((orig_idx, dup_idx))
        return set(l_errors), ["original", "duplicate"]
