from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark import imdb_benchmark
from utils import get_logger
from customize_model import remove_positional_embeddings


logger = get_logger(__name__)


def main() -> None:
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased"
    )
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    model = remove_positional_embeddings(model)

    logger.info(model)

    imdb_benchmark(model, tokenizer)


if __name__ == "__main__":
    main()
