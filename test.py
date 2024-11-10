import transformers
from transformers.models.bert.modeling_bert import BertSdpaSelfAttention


def main() -> None:
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased"
    )

    print(model)
    print("======================================")

    for _, module in model.named_modules():
        if not isinstance(module, BertSdpaSelfAttention):
            continue

        module.position_embedding_type = None

    print(model)


if __name__ == "__main__":
    main()
