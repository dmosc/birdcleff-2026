import random
import numpy as np

from typing import cast, Any

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
        self.label_to_id, self.id_to_label = self._get_label_maps(dataset)
        dataset = self._prepare_dataset(dataset)
        super().__init__(
            cast(TorchDataset[Any], dataset.with_format('torch')),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.hf_dataset = dataset

    def get_dataset_splits(self):
        train_test_split = self.hf_dataset.train_test_split(
            test_size=self.config.test_size,
            seed=self.config.seed
        )
        return train_test_split['train'], train_test_split['test']

    def get_num_unique_labels(self):
        return len(self.label_to_id.keys())

    def _get_label_maps(self, dataset):
        all_labels = []

        for label in dataset['primary_label']:
            all_labels.extend(label.split(';'))

        unique_labels = sorted(list(set(all_labels)))
        label_to_id = {label: id for id, label in enumerate(unique_labels)}
        id_to_label = {id: label for id, label in enumerate(unique_labels)}
        return label_to_id, id_to_label

    def _prepare_dataset(self, dataset: HFDataset) -> HFDataset:
        dataset = dataset.map(self._rename_label_column)
        dataset = dataset.map(self._add_audio_path)
        dataset = dataset.cast_column(
            'audio',
            Audio(sampling_rate=self.config.audio_sampling_rate)
        )
        ast_feature_extractor = ASTFeatureExtractor.from_pretrained(
            self.config.ast_feature_extractor_id
        )
        columns_to_remove = [
            column for column in dataset.column_names
            if column not in self.config.columns_to_keep
        ]
        dataset = dataset.map(
            self._parse_audio_as_mel_spectrogram,
            batched=True,
            batch_size=self.config.batch_size,
            fn_kwargs={'ast_feature_extractor': ast_feature_extractor},
            remove_columns=columns_to_remove,
        )
        return dataset

    def _rename_label_column(self, example):
        num_classes = self.get_num_unique_labels()
        multi_hot_encoding = np.zeros(num_classes, dtype=np.float32)

        for label in example['primary_label'].split(';'):
            if label in self.label_to_id:
                multi_hot_encoding[self.label_to_id[label]] = 1.0

        example['labels'] = multi_hot_encoding
        return example

    def _add_audio_path(self, example):
        example['audio'] = (
            f'{self.config.audio_data_folder}/{example["filename"]}'
        )
        return example

    def _parse_audio_as_mel_spectrogram(self, examples, ast_feature_extractor):
        target_samples = self.config.audio_sampling_rate * \
            self.config.audio_seconds_to_sample
        audios_to_process = []

        for audio_data in examples['audio']:
            audio_array = audio_data['array']
            if len(audio_array) > target_samples:
                start = random.randint(0, len(audio_array) - target_samples)
                audio_array = audio_array[start: start + target_samples]
            audios_to_process.append(audio_array)

        inputs = ast_feature_extractor(
            audios_to_process,
            sampling_rate=self.config.audio_sampling_rate,
            return_tensors='pt',
            padding='max_length',
            max_length=self.config.max_time_frames_in_spectrogram,
        )
        return {
            "input_values": inputs['input_values'].numpy(),
            "labels": examples['labels']
        }
