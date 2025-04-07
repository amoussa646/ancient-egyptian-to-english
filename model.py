import torch
import gc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, Seq2SeqTrainingArguments, Trainer
from config import training_args

def get_training_args(output_path, **overrides):
    """Get training arguments with optional overrides."""
    kwargs = {**training_args, **overrides, "output_dir": output_path}
    return Seq2SeqTrainingArguments(**kwargs)

def setup_model(model_name, device):
    """Set up the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = T5Config.from_pretrained(model_name, dropout_rate=0.1, attention_dropout_rate=0.1)
    config.decoder_start_token_id = config.pad_token_id
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config).to(device)
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

class CustomTrainer(Trainer):
    """Custom trainer with GPU memory management."""
    def evaluate(self, *args, **kwargs):
        # Clear GPU cache before evaluation
        torch.cuda.empty_cache()
        gc.collect()
        return super().evaluate(*args, **kwargs)    