import numpy as np
import evaluate

# Initialize metrics
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
bleu_metric = evaluate.load("bleu")


# Load ROUGE-L, METEOR, and BLEU metrics
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
bleu_metric = evaluate.load("bleu")
def compute_hyphen_position_accuracy(pred_str, label_str):
    """Calculate accuracy of hyphens being in the correct positions"""
    # Get hyphen positions for both strings
    label_hyphens = set(i for i, c in enumerate(label_str) if c == '-')
    pred_hyphens = set(i for i, c in enumerate(pred_str) if c == '-')

    # Case 1: No hyphens expected
    if not label_hyphens:
        return 1.0 if not pred_hyphens else 0.0

    # Case 2: Check position matches
    correct_positions = label_hyphens & pred_hyphens
    position_accuracy = len(correct_positions) / len(label_hyphens)

    # Penalize extra hyphens
    extra_hyphens = len(pred_hyphens - label_hyphens)
    penalty = 0.5 ** extra_hyphens  # Reduce score by 50% per extra hyphen

    return position_accuracy * penalty


# Define the compute_metrics function
def compute_metrics(pred, tokenizer):
    """Compute evaluation metrics for the model."""
    pred_ids, label_ids = pred

    # Replace -100 in labels
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute metrics
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "hyphen_accuracy": np.mean([
            compute_hyphen_position_accuracy(p, l)
            for p, l in zip(decoded_preds, decoded_labels)
        ]),
        "exact_match": np.mean([
            int(p.strip() == l.strip())
            for p, l in zip(decoded_preds, decoded_labels)
        ]),
        "rouge_l": rouge_result["rougeL"],
        "bleu": bleu_result["bleu"],
        "meteor": meteor_result["meteor"]
    }

