import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of text")
    parser.add_argument("--model_dir", default="./model", help="Path to fine-tuned model")
    parser.add_argument("--text", required=True, help="Text to classify")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = logits.argmax(dim=-1).item()
    label = "positive" if predicted_class == 1 else "negative"
    print(label)


if __name__ == "__main__":
    main()
