import torch
import seaborn
from matplotlib import pyplot
from pathlib import Path
from transformers import AutoModelForCausalLM

from utils import get_logger

logger = get_logger(__name__)

MAX_ID = 1024


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    with torch.no_grad():
        token_ids = torch.arange(0, 50257, dtype=torch.long).unsqueeze(0)
        token_embeddings = model.transformer.wte(token_ids)

    logger.info(token_embeddings.shape)

    pyplot.figure()
    seaborn.heatmap(token_embeddings[0].transpose(0, 1))

    pyplot.xlabel("token")
    pyplot.ylabel("embedding")

    """
    pyplot.xticks(
        ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
        labels=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
    )
    """
    pyplot.yticks(
        ticks=[0, 128, 256, 384, 512, 640, 768],
        labels=[0, 128, 256, 384, 512, 640, 768],
    )

    save_path = Path("outputs/embedding")
    save_path.mkdir(parents=True, exist_ok=True)

    pyplot.savefig(save_path / "gpt2_token.png", bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    main()
