import torch
import os
from config import *
from data_loading import load_data, prepare_datasets, preprocess_function
from model import setup_model, CustomTrainer, get_training_args
from metrics import compute_metrics
from evaluation import evaluate_model

def main():
    # Set up environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare data
    data = load_data(DATA_PATH)
    train_dataset, val_dataset = prepare_datasets(data)
    
    # Setup model
    model, tokenizer = setup_model(MODEL_NAME, device)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    # Training
    training_args = get_training_args(OUTPUT_PATH)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
    )
    
    # Train the model
    trainer.train()
    
    # Save model
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    # Evaluate the model
    metrics = evaluate_model(model, tokenizer, val_dataset)
    print("Final Evaluation Metrics:", metrics)

if __name__ == "__main__":
    main()