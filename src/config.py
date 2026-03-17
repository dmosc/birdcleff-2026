class Config:
    batch_size = 16
    dataset_sample_size = 0.001
    dataset_file_path = 'data/train.csv'
    audio_data_folder = 'data/train_audio'
    ast_feature_extractor_id = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    audio_sampling_rate = 16_000
    audio_seconds_to_sample = 10
    max_time_frames_in_spectogram = 1024
