class Config:
    seed = 11
    test_size = 0.1
    dataset_sample_size = 0.001
    epochs = 10
    batch_size = 16
    warmup_ratio = 0.1
    checkpoint_folder = 'data/checkpoints'
    dataset_file_path = 'data/train.csv'
    audio_data_folder = 'data/train_audio'
    ast_feature_extractor_id = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    audio_sampling_rate = 16_000
    audio_seconds_to_sample = 10
    max_time_frames_in_spectrogram = 1024
    columns_to_keep = ['input_values', 'labels']
