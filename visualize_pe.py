import torch
import seaborn
from matplotlib import pyplot
from pathlib import Path
from transformers import AutoModelForCausalLM

from utils import get_logger

logger = get_logger(__name__)

MAX_LENGTH = 1024


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    with torch.no_grad():
        position_ids = torch.arange(0, MAX_LENGTH, dtype=torch.long).unsqueeze(0)
        position_embeddings = model.transformer.wpe(position_ids)

    logger.info(position_embeddings.shape)

    pyplot.figure()
    seaborn.heatmap(position_embeddings[0].transpose(0, 1))

    pyplot.xlabel("position")
    pyplot.ylabel("embedding")

    pyplot.xticks(
        ticks=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
        labels=[0, 128, 256, 384, 512, 640, 768, 896, 1024],
    )
    pyplot.yticks(
        ticks=[0, 128, 256, 384, 512, 640, 768],
        labels=[0, 128, 256, 384, 512, 640, 768],
    )

    save_path = Path("outputs/embedding")
    save_path.mkdir(parents=True, exist_ok=True)

    pyplot.savefig(save_path / "gpt2_position.png", bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    main()
