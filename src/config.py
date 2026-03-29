class Config:
    seed = 11
    test_size = 0.1
    dataset_sample_size = 1
    epochs = 10
    batch_size = 16
    warmup_steps = 0.1
    checkpoint_folder = 'data/checkpoints'
    dataset_file_path = 'data/train_soundscapes_labels.csv'
    taxonomy_file_path = 'data/taxonomy.csv'
    audio_data_folder = 'data/train_soundscapes'
    ast_feature_extractor_id = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    audio_sampling_rate = 16_000
    audio_seconds_to_sample = 5
    max_timeframes_in_spectrogram = 512
    columns_to_keep = ['input_values', 'labels']

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
