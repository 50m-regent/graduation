import transformers


def main() -> None:
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased"
    )

    print(model)
