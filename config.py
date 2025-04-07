import pandas as pd
from transformers import Seq2SeqTrainingArguments

# Data configuration
DATA_PATH = "swap4origin.csv"
TEST_SIZE = 0.1
RANDOM_STATE = 42

# Model configuration
MODEL_NAME = '/content/drive/MyDrive/egy_nlp/checkpoints/checkpoint-1194'
OUTPUT_PATH = "/content/drive/My Drive/egy_nlp/checkpoints"

# Training configuration
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_PATH,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=16,
    eval_steps=500,
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=1,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=100,
    fp16=False,
    load_best_model_at_end=False,
    metric_for_best_model="loss",
    overwrite_output_dir=True,
    label_smoothing_factor=0.1,
    warmup_steps=500,
    max_grad_norm=1.0,
)