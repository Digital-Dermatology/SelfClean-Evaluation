import copy
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from ssl_library.src.datasets.base_dataset import BaseDataset

from ..perturbed_datasets.base_contamination_dataset import BaseContaminationDataset


class LabelErrorDataset(BaseContaminationDataset):
    def __init__(
        self,
        dataset: BaseDataset,
        transform=None,
        frac_error: float = 0.1,
        n_errors: Optional[int] = None,
        change_for_every_label: bool = False,
        label_col: Optional[str] = None,
        meta_data: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.frac_lbl_error = frac_error
        self.change_for_every_label = change_for_every_label
        self.name = name if name is not None else type(self).__name__
        self.dataset_orig = copy.copy(dataset)
        self.dataset_orig.transform = None
        self.transform = transform
        if label_col is None:
            label_col = self.dataset_orig.LBL_COL
        if meta_data is None:
            meta_data = self.dataset_orig.meta_data
        l_idx = list(meta_data.index)
        if change_for_every_label:
            self.lbl_error_idx = []
            for lbl in meta_data[label_col].unique():
                df_lbl = meta_data[meta_data[label_col] == lbl]
                if n_errors is not None:
                    n_changes = n_errors
                else:
                    n_changes = math.ceil(frac_error * len(df_lbl))
                rdn_idx = np.random.choice(
                    list(df_lbl.index),
                    size=n_changes,
                    replace=False,
                )
                self.lbl_error_idx += list(rdn_idx)
        else:
            if n_errors is not None:
                n_changes = n_errors
            else:
                n_changes = math.ceil(frac_error * len(l_idx))
            self.lbl_error_idx = np.random.choice(
                list(meta_data.index), size=n_changes, replace=False
            )
        # global configs
        self.classes = self.dataset_orig.classes
        self.n_classes = self.dataset_orig.n_classes
        # transform the random sample
        self.aug_samples = []
        for idx in range(len(self.dataset_orig)):
            rets = self.dataset_orig[idx]
            if idx in self.lbl_error_idx:
                # alter the label
                lbl = rets[-1]
                poss_lbls = list(range(self.n_classes))
                if lbl in poss_lbls:
                    poss_lbls.remove(lbl)
                new_lbl = np.random.choice(poss_lbls)
                rets = rets[:-1] + (new_lbl,)  # comes from original dataset
            self.aug_samples.append(rets)

    def __len__(self):
        return len(self.dataset_orig)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rets = self.aug_samples[idx]
        if self.transform is not None:
            image = self.transform(rets[0])
            rets = (image,) + rets[1:]
        return rets

    def info(self):
        logger.info(f"Name of dataset: {self.name}")
        logger.info(f"Datset size: {len(self)}")
        logger.info(f"Fraction of lbl. errors: {self.frac_lbl_error}")
        logger.info(f"Change for every label separately: {self.change_for_every_label}")
        logger.info(f"N. of label changes: {len(self.lbl_error_idx)}")

    def get_errors(self) -> Tuple[list, list]:
        l_errors = list(range(len(self)))
        # TODO: harmonize with the other perturbed datasets!
        l_errors = [1 if x in self.lbl_error_idx else 0 for x in l_errors]
        return l_errors, ["original", "label error"]
