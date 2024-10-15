import copy
import math
from random import sample
from typing import List, Optional, Union

import torch
from loguru import logger
from ssl_library.src.datasets.base_dataset import BaseDataset
from torch.utils.data import Dataset

from ..perturbed_datasets.base_contamination_dataset import BaseContaminationDataset


class CombinedDataset(BaseContaminationDataset):
    def __init__(
        self,
        dataset: BaseDataset,
        dataset_aug: Union[BaseDataset, List[Dataset]],
        transform=None,
        frac_error: float = 0.1,
        n_errors: Optional[int] = None,
        name: Optional[str] = None,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__()
        self.name = name if name is not None else type(self).__name__
        self.dataset_orig = copy.copy(dataset)
        self.dataset_aug = copy.copy(dataset_aug)
        if n_errors is not None:
            n_samples = n_errors
        else:
            n_samples = math.ceil(frac_error * len(self.dataset_orig))

        if type(dataset_aug) is list:
            self.n_samples_per_dataset = n_samples // len(dataset_aug)
            classes = []
            for aug_data in self.dataset_aug:
                aug_data.transform = None
                aug_data.samples = sample(aug_data.samples, self.n_samples_per_dataset)
                aug_data.samples = [
                    (x[0], x[1] + len(classes)) for x in aug_data.samples
                ]
                classes += aug_data.classes
            self.dataset_aug = torch.utils.data.ConcatDataset(self.dataset_aug)
        else:
            self.dataset_aug.meta_data = self.dataset_aug.meta_data.sample(
                n=n_samples,
                random_state=random_state,
                replace=False,
            )
            self.dataset_aug.transform = None
            classes = self.dataset_aug.classes
        # by setting the transform to `None` we obtain the raw PIL image
        # the dataset handles the augmentation itself
        self.dataset_orig.transform = None
        self.transform = transform
        # global configs
        self.classes = self.dataset_orig.classes + classes
        self.n_classes = len(self.classes)

    def __len__(self):
        return len(self.dataset_orig) + len(self.dataset_aug)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < len(self.dataset_orig):
            rets = self.dataset_orig[idx]
        else:
            idx = idx - len(self.dataset_orig)
            rets = self.dataset_aug[idx]
            lbl = rets[-1]
            lbl += self.dataset_orig.n_classes
            if type(self.dataset_aug) is torch.utils.data.ConcatDataset:
                data_idx = idx // self.n_samples_per_dataset
                data = self.dataset_aug.datasets[data_idx]
                path = data.samples[idx - (data_idx * self.n_samples_per_dataset)][0]
                rets = rets[:-1] + (path, lbl)
            else:
                rets = rets[:-1] + (lbl,)
        if self.transform is not None:
            image = self.transform(rets[0])
            rets = (image,) + rets[1:]
        return rets

    def info(self):
        logger.info(f"Name of dataset: {self.name}")
        logger.info(f"Original datset size: {len(self.dataset_orig)}")
        logger.info(f"Augmented datset size: {len(self.dataset_aug)}")
        logger.info(f"Combined datset size: {len(self)}")

    def get_errors(self):
        l_errors = []
        for idx in range(len(self)):
            if idx >= len(self.dataset_orig):
                l_errors.append(idx)
        return set(l_errors), ["original", "irrelevant sample"]
