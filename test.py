import transformers
from transformers.models.bert.modeling_bert import BertSdpaSelfAttention

from benchmark import imdb_benchmark
from utils import get_logger


logger = get_logger(__name__)


def main() -> None:
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-uncased"
    )

    for _, module in model.named_modules():
        if not isinstance(module, BertSdpaSelfAttention):
            continue

        module.position_embedding_type = None

    logger.info(model)

    imdb_benchmark(model)


if __name__ == "__main__":
    main()
