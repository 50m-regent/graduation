from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark import imdb_benchmark, imdb_feature_comparison
from utils import get_logger


logger = get_logger(__name__)


def main() -> None:
    model1 = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased"
    )
    model2 = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased", position_embedding_type="none"
    )
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    imdb_feature_comparison(model1, model2, tokenizer)

    imdb_benchmark(model1, tokenizer)


if __name__ == "__main__":
    main()
