import datasets
import torch
from sklearn import metrics
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoModel, AutoTokenizer

from utils import get_logger


logger = get_logger(__name__)


def imdb_benchmark(model: AutoModel, tokenizer: AutoTokenizer) -> None:
    dataset = datasets.load_dataset("imdb")["test"]

    logger.info(dataset)

    labels = dataset["label"]

    dataset = tokenizer(
        dataset["text"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    predictions = []

    with torch.no_grad(), logging_redirect_tqdm(loggers=[logger]):
        for input_ids, token_type_ids, attention_mask in tqdm(
            zip(
                dataset["input_ids"],
                dataset["token_type_ids"],
                dataset["attention_mask"],
            ),
            total=len(dataset["input_ids"]),
        ):
            prediction = (
                model(
                    input_ids.unsqueeze(dim=0),
                    token_type_ids.unsqueeze(dim=0),
                    attention_mask.unsqueeze(dim=0),
                )
                .logits.argmax()
                .item()
            )

            predictions.append(prediction)

    accuracy = metrics.accuracy_score(labels, predictions)
    f1 = metrics.f1_score(labels, predictions).item()
    precision = metrics.precision_score(labels, predictions).item()
    recall = metrics.recall_score(labels, predictions).item()

    logger.info(f"{accuracy=}")
    logger.info(f"{f1=}")
    logger.info(f"{precision=}")
    logger.info(f"{recall=}")
