import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from config import TEST_SIZE, RANDOM_STATE

def load_data(data_path):
    """Load and shuffle the dataset."""
    data = pd.read_csv(data_path)
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    data.columns = ["input_text", "target_text"]
    return data

def prepare_datasets(data):
    """Split data into training and validation sets."""
    input_texts = data.iloc[:, 0].tolist()
    target_texts = data.iloc[:, 1].tolist()
    
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        input_texts, target_texts, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    train_dataset = Dataset.from_dict({"input_text": train_inputs, "target_text": train_targets})
    val_dataset = Dataset.from_dict({"input_text": val_inputs, "target_text": val_targets})
    
    return train_dataset, val_dataset

# Tokenization function
# data_loading.py
def preprocess_function(examples, tokenizer):
    """Tokenize the input and target texts."""
    inputs = tokenizer(
        examples["input_text"], 
        max_length=512, 
        truncation=True, 
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"], 
            max_length=512, 
            truncation=True, 
            padding="max_length"
        )
    inputs["decoder_input_ids"] = [
        [tokenizer.pad_token_id] + seq[:-1] for seq in labels["input_ids"]
    ]
    inputs["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in seq]
        for seq in labels["input_ids"]
    ]
    return inputs