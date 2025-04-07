from torch.utils.data import DataLoader
from collections import defaultdict
import evaluate

def compute_metrics_on_the_fly(eval_dataloader, model, tokenizer):
    """Compute metrics incrementally for large datasets."""
    # Initialize metrics
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    
    # Initialize accumulators
    rouge_scores = defaultdict(float)
    meteor_scores = []
    bleu_scores = []
    total_samples = 0

    model.eval()  # Ensure model is in eval mode
    
    for batch in eval_dataloader:
        # Move batch to same device as model
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # Get input_ids and labels
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Generate predictions
        predictions = model.generate(input_ids, max_length=160)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute metrics
        rouge_result = rouge_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels
        )
        bleu_result = bleu_metric.compute(
            predictions=decoded_preds, 
            references=[[label] for label in decoded_labels]
        )
        meteor_result = meteor_metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels
        )

        # Accumulate metrics
        rouge_scores["rougeL"] += rouge_result["rougeL"] * len(decoded_preds)
        bleu_scores.append(bleu_result["bleu"] * len(decoded_preds))
        meteor_scores.append(meteor_result["meteor"] * len(decoded_preds))
        total_samples += len(decoded_preds)

    # Normalize scores
    rouge_scores = {key: score / total_samples for key, score in rouge_scores.items()}
    bleu_score = sum(bleu_scores) / total_samples
    meteor_score = sum(meteor_scores) / total_samples

    return {
        "bleu": bleu_score,
        "rouge_l": rouge_scores["rougeL"],
        "meteor": meteor_score,
    }

def evaluate_model(model, tokenizer, val_dataset, batch_size=1):
    """Convenience function to run evaluation."""
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return compute_metrics_on_the_fly(eval_dataloader, model, tokenizer)