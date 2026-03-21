import numpy as np

from transformers import (
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    EvalPrediction
)
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    label_ranking_average_precision_score,
    f1_score
)

from src.data_manager import DataManager
from src.config import Config


def compute_metrics(eval_predictions: EvalPrediction):
    print(eval_predictions)
    predictions = np.array(eval_predictions.predictions)
    labels = np.array(eval_predictions.label_ids).astype(np.float32)
    probabilities = expit(predictions)
    # Measures global ranking quality.
    mean_average_precision = average_precision_score(
        labels,
        probabilities,
        average='macro'
    )
    # Measures per-sample ranking quality.
    label_ranking_average_precision = label_ranking_average_precision_score(
        labels,
        probabilities
    )
    # F1-score derived from an explicit 20% confidence threshold in predictions.
    predictions = (probabilities > 0.2).astype(float)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    # Measures if the "most" confident prediction is right.
    top_idxs = np.argmax(probabilities, axis=1)
    top1_matches = labels[np.arange(len(labels)), top_idxs]
    top1_accuracy = np.mean(top1_matches)
    # Measures how confident on average is the top predictions.
    mean_max_confidence = np.mean(np.max(probabilities, axis=1))
    return {
        'mean_average_precision': mean_average_precision,
        'label_ranking_average_precision': label_ranking_average_precision,
        'f1': f1,
        'top1_accuracy': top1_accuracy,
        'mean_max_confidence': mean_max_confidence
    }


def main():
    print('Setting up training...')
    config = Config()
    data_manager = DataManager(config)
    train_dataset, test_dataset = data_manager.get_dataset_splits()
    model_config = AutoConfig.from_pretrained(
        config.ast_feature_extractor_id,
        num_labels=data_manager.get_num_unique_labels(),
        label2id=data_manager.label_to_id,
        id2label=data_manager.id_to_label,
        # Use BCEWithLogitsLoss instead of standard Cross Entropy.
        problem_type='multi_label_classification'
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
        warmup_steps=config.warmup_steps,
        push_to_hub=False,
        report_to='none',
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == '__main__':
    main()
