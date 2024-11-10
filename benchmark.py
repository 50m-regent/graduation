import datasets
from transformers import AutoModel

from utils import get_logger


logger = get_logger(__name__)


def imdb_benchmark(model: AutoModel) -> None:
    dataset = datasets.load_dataset("imdb")["test"]

    logger.info(dataset)
