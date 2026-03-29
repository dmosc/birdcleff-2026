import random
import torch
import numpy as np

from typing import cast, Any

from datasets import Audio
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchcodec.decoders import AudioDecoder
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

    @staticmethod
    def get_inference_input(
        config: Config,
        audio_path: str,
        seconds_per_sample: int
    ) -> torch.Tensor:
        audio_decoder = AudioDecoder(
            audio_path,
            sample_rate=config.audio_sampling_rate
        )
        ast_feature_extractor = ASTFeatureExtractor.from_pretrained(
            config.ast_feature_extractor_id,
            max_length=config.max_timeframes_in_spectrogram,
        )
        inputs = []
        assert audio_decoder.metadata.duration_seconds
        duration_seconds = int(audio_decoder.metadata.duration_seconds)

        for to_seconds in range(seconds_per_sample, duration_seconds + 1,
                                seconds_per_sample):
            from_seconds = to_seconds - seconds_per_sample
            audio_frames = audio_decoder.get_samples_played_in_range(
                from_seconds,
                to_seconds
            )
            audio_samples = audio_frames.data.float()
            sample_input = ast_feature_extractor(
                audio_samples.numpy(),
                return_tensors='pt',
                sampling_rate=config.audio_sampling_rate
            )
            inputs.append(sample_input['input_values'])

        return torch.cat(inputs, dim=0)

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
            self.config.ast_feature_extractor_id,
            max_length=self.config.max_timeframes_in_spectrogram,
        )
        remove_columns = [
            column for column in dataset.column_names
            if column not in self.config.columns_to_keep
        ]
        dataset = dataset.map(
            self._parse_audio_as_mel_spectrogram,
            batched=True,
            batch_size=self.config.batch_size,
            fn_kwargs={'ast_feature_extractor': ast_feature_extractor},
            remove_columns=remove_columns,
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

        for idx, audio_data in enumerate(examples['audio']):
            audio_array = audio_data['array']

            def _to_secs(timemstap: str):
                hours, minutes, seconds = map(int, timemstap.split(':'))
                return hours * 3600 + minutes * 60 + seconds

            segment_start_idx = int(
                _to_secs(examples['start'][idx]) *
                self.config.audio_sampling_rate
            )
            segment_end_idx = int(
                _to_secs(examples['end'][idx]) *
                self.config.audio_sampling_rate
            )
            audio_segment_array = audio_array[segment_start_idx:segment_end_idx]

            if len(audio_segment_array) > target_samples:
                start_offset = random.randint(
                    0, len(audio_segment_array) - target_samples
                )
                audio_segment_array = audio_segment_array[
                    start_offset:start_offset + target_samples
                ]

            audios_to_process.append(audio_segment_array)

        inputs = ast_feature_extractor(
            audios_to_process,
            return_tensors='pt',
            sampling_rate=self.config.audio_sampling_rate
        )
        return {
            "input_values": inputs['input_values'].numpy(),
            "labels": examples['labels']
        }
