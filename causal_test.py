import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import qa_benchmark
from utils import get_logger

logger = get_logger(__name__)


class ZeroPE(torch.nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        self.embed_dim = embed_dim

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (position_ids.shape[0], position_ids.shape[1], self.embed_dim)
        )


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model.transformer.wpe = ZeroPE(768)

    logger.info(model)

    qa_benchmark(model, tokenizer)


if __name__ == "__main__":
    main()
