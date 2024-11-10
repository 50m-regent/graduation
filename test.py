from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark import imdb_benchmark
from utils import get_logger


logger = get_logger(__name__)


def main() -> None:
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased"
    )
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    """
    for _, module in model.named_modules():
        if not isinstance(module, BertSdpaSelfAttention):
            continue

        module.position_embedding_type = None
    """

    logger.info(model)

    imdb_benchmark(model, tokenizer)


if __name__ == "__main__":
    main()
