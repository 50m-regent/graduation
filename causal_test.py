import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import qa_benchmark
from utils import get_logger
from zero_pe import ZeroPE

logger = get_logger(__name__)


def change_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.arange(0.1, 1.001, 0.9 / (mask.shape[1] - 1)).unsqueeze(dim=0)


def no_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(mask)


def main() -> None:
    model1 = AutoModelForCausalLM.from_pretrained("gpt2")
    model2 = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model2.transformer.wpe = ZeroPE(768)

    logger.info(model1)
    logger.info(model2)

    qa_benchmark(model=model1, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
