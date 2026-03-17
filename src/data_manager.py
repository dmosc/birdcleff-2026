from typing import cast, Any
from pathlib import Path

from datasets import Audio
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import ASTFeatureExtractor
from src.config import Config


class DataManager(DataLoader):
    def __init__(self, config: Config):
        self.config = config
        dataset = HFDataset.from_csv(self.config.dataset_file_path)
        dataset = dataset.select(
            range(int(len(dataset) * self.config.dataset_sample_size))
        )
        print(f'Processing {len(dataset)} rows.')
        dataset = self._prepare_dataset(dataset)
        super().__init__(
            cast(TorchDataset[Any], dataset.with_format('torch')),
            batch_size=self.config.batch_size,
            shuffle=True
        )

    def _prepare_dataset(self, dataset: HFDataset) -> HFDataset:
        dataset = dataset.map(self._add_audio_path)
        dataset = dataset.cast_column(
            'audio',
            Audio(sampling_rate=self.config.audio_sampling_rate)
        )
        ast_feature_extractor = ASTFeatureExtractor.from_pretrained(
            self.config.ast_feature_extractor_id
        )
        dataset = dataset.map(
            self._parse_audio_as_mel_spectogram,
            batched=True,
            fn_kwargs={'ast_feature_extractor': ast_feature_extractor},
            remove_columns=['audio']
        )
        return dataset

    def _add_audio_path(self, example):
        example['audio'] = (
            f'{self.config.audio_data_folder}/{example["filename"]}'
        )
        return example

    def _parse_audio_as_mel_spectogram(self, examples, ast_feature_extractor):
        return ast_feature_extractor(
            [example['array'] for example in examples['audio']],
            sampling_rate=self.config.audio_sampling_rate,
            return_tensors='pt'
        )
