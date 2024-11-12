import datasets
import torch
from sklearn import metrics
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import AutoModel, AutoTokenizer

from utils import get_logger


logger = get_logger(__name__)


def __get_dataset(
    tokenizer: AutoTokenizer
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    dataset = datasets.load_dataset("imdb")["test"][:10]
    labels = dataset["label"]

    logger.info(dataset)

    dataset = tokenizer(
        dataset["text"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return dataset, labels


def imdb_feature_comparison(
    model1: AutoModel,
    model2: AutoModel,
    tokenizer: AutoTokenizer,
) -> None:
    dataset, _ = __get_dataset(tokenizer)

    similarities = []

    with torch.no_grad(), logging_redirect_tqdm(loggers=[logger]):
        for input_ids, token_type_ids, attention_mask in tqdm(
            zip(
                dataset["input_ids"],
                dataset["token_type_ids"],
                dataset["attention_mask"],
            ),
            total=len(dataset["input_ids"]),
        ):
            feature1 = model1.bert(
                input_ids.unsqueeze(dim=0),
                token_type_ids.unsqueeze(dim=0),
                attention_mask.unsqueeze(dim=0),
            ).last_hidden_state
            feature2 = model2.bert(
                input_ids.unsqueeze(dim=0),
                token_type_ids.unsqueeze(dim=0),
                attention_mask.unsqueeze(dim=0),
            ).last_hidden_state

            similarity = (
                torch.cosine_similarity(feature1, feature2, dim=2).mean(dim=1).item()
            )

            similarities.append(similarity)

    similarity = torch.asarray(similarities).mean()

    logger.info(f"{similarity=}")


def imdb_benchmark(model: AutoModel, tokenizer: AutoTokenizer) -> None:
    dataset, labels = __get_dataset(tokenizer)

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
