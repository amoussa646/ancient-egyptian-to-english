import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, pipeline

def load_model(model_path: str):
    """Load the trained model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_text(model, tokenizer, input_text: str, max_length: int = 512):
    """Generate text using the trained model."""
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():

        outputs = model.generate(input_ids, attention_mask=attention_mask,
                         max_length=max_length, do_sample=False, top_k=4, temperature=0.5)


    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    model_path = "/content/drive/My Drive/egy_nlp/checkpoints/checkpoint-597"
    model, tokenizer = load_model(model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

# G5-M17X1V31-Q3G43-Q3M22M22-Q3D4-X8N35
    input_text = "G5M17X1V31Q3G43Q3M22M22Q3D4X8N35"

    generated_text = generate_text(model, tokenizer, input_text)
    print("Generated Output:", generated_text)
main()



