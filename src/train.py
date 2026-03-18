from transformers import (
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from src.data_manager import DataManager
from src.config import Config


def main():
    config = Config()
    data_manager = DataManager(config)
    train_dataset, test_dataset = data_manager.get_dataset_splits()
    model_config = AutoConfig.from_pretrained(
        config.ast_feature_extractor_id,
        num_labels=data_manager.get_num_unique_labels(),
        label2id=data_manager.label_to_id,
        id2label=data_manager.id_to_label
    )
    model = ASTForAudioClassification.from_pretrained(
        config.ast_feature_extractor_id,
        config=model_config,
        ignore_mismatched_sizes=True
    )
    training_args = TrainingArguments(
        output_dir=config.checkpoint_folder,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=10,
        num_train_epochs=config.epochs,
        warmup_ratio=config.warmup_ratio,
        push_to_hub=False,
        report_to='none',
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()


if __name__ == '__main__':
    main()
