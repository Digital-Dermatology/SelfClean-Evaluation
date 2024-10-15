from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple, Union

from torch.utils.data import Dataset


class BaseContaminationDataset(Dataset, ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def get_errors(self) -> Tuple[Set[Union[int, tuple]], List[str]]:
        pass

    def cleanup(self):
        return
