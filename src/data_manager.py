from typing import cast, Any
from pathlib import Path

from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset as TorchDataset

from src.config import Config


class DataManager(DataLoader):
    def __init__(self, config: Config):
        dataset_path = Path(config.data_path) / 'train.csv'
        dataset = HFDataset.from_csv(dataset_path.as_posix())
        super().__init__(
            cast(TorchDataset[Any], dataset.with_format('torch')),
            batch_size=config.batch_size,
            shuffle=True
        )
