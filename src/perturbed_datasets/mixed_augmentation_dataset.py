from typing import Callable, Optional

from loguru import logger
from ssl_library.src.datasets.base_dataset import BaseDataset

from ..perturbed_datasets.base_contamination_dataset import BaseContaminationDataset


class MixedAugmentationDataset(BaseContaminationDataset):
    def __init__(
        self,
        dataset: BaseDataset,
        duplicate_dataset_cls: Callable,
        irrelevant_dataset_cls: Callable,
        label_error_dataset_cls: Callable,
        transform=None,
        frac_error: float = 0.1,
        name: Optional[str] = None,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        super().__init__()
        self.name = name if name is not None else type(self).__name__
        # create the correct datasets
        self.frac_error = frac_error
        self.frac_error = ((1 + self.frac_error) ** (1 / 3)) - 1
        # NOTE: they have a dependency on each other!
        self.irrelevant_dataset = irrelevant_dataset_cls(
            dataset=dataset,
            transform=transform,
            frac_error=self.frac_error,
            random_state=random_state,
            name=f"{self.name}-IrrelevantSamples",
        )
        self.duplicate_dataset = duplicate_dataset_cls(
            dataset=self.irrelevant_dataset,
            transform=transform,
            frac_error=self.frac_error,
            random_state=random_state,
            name=f"{self.name}-NearDuplicates",
        )
        self.label_error_dataset = label_error_dataset_cls(
            dataset=self.duplicate_dataset,
            transform=transform,
            frac_error=self.frac_error,
            random_state=random_state,
            label_col=dataset.LBL_COL,
            meta_data=dataset.meta_data,
            name=f"{self.name}-LabelErrors",
        )

    @property
    def classes(self):
        # as the label errors are the last dataset perturbation
        # they include all classes and can be used
        return self.label_error_dataset.classes

    @property
    def n_classes(self):
        # as the label errors are the last dataset perturbation
        # they include all classes and can be used
        return self.label_error_dataset.n_classes

    @property
    def irrelevant_classes(self):
        return self.irrelevant_dataset.classes

    @property
    def duplicate_classes(self):
        return self.duplicate_dataset.classes

    @property
    def label_error_classes(self):
        return self.label_error_dataset.classes

    @property
    def irrelevant_n_classes(self):
        return self.irrelevant_dataset.n_classes

    @property
    def duplicate_n_classes(self):
        return self.duplicate_dataset.n_classes

    @property
    def label_error_n_classes(self):
        return self.label_error_dataset.n_classes

    def __len__(self):
        # since all datasets are chained together we're using the last one
        return len(self.label_error_dataset)

    def __getitem__(self, idx: int):
        # since all datasets are chained together we're using the last one
        return self.label_error_dataset[idx]

    def info(self):
        logger.info(f"Name of dataset: {self.name}")
        self.irrelevant_dataset.info()
        self.duplicate_dataset.info()
        self.label_error_dataset.info()

    def get_errors(self):
        raise NotImplementedError(
            "Overall `get_errors` not implemented, "
            "use the error specific implementation, e.g. `get_duplicate_errors`"
        )

    def get_irrelevant_errors(self):
        return self.irrelevant_dataset.get_errors()

    def get_duplicate_errors(self):
        return self.duplicate_dataset.get_errors()

    def get_label_errors(self):
        return self.label_error_dataset.get_errors()
