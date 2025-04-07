# Egypt-English Translation Model

This project implements a sequence-to-sequence model for translation tasks using the T5 architecture.

## Project Structure

- `main.py`: Main training script
- `config.py`: Configuration parameters
- `data_loading.py`: Data loading and preprocessing
- `model.py`: Model setup and training utilities
- `metrics.py`: Evaluation metrics
- `evaluation.py`: Model evaluation utilities
- `inference.py`: Model inference utilities

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare your data:

- Place your training data in a CSV file with columns "input_text" and "target_text"
- Update the `DATA_PATH` in `config.py` to point to your data file

## Usage

1. Training:

```bash
python main.py
```

2. Inference:

```bash
python inference.py
```

## Configuration

You can modify the following parameters in `config.py`:

- `DATA_PATH`: Path to your training data
- `MODEL_NAME`: Pretrained model to use
- `OUTPUT_PATH`: Where to save the trained model
- `TEST_SIZE`: Validation set size
- Training parameters in `training_args`

## Model Architecture

The model uses the T5 architecture with the following features:

- Gradient checkpointing for memory efficiency
- Custom evaluation with GPU memory management
- Multiple evaluation metrics (BLEU, ROUGE, METEOR)
- Hyphen position accuracy tracking
