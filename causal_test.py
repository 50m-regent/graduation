from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import qa_benchmark
from utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    logger.info(model)

    qa_benchmark(model, tokenizer)


if __name__ == "__main__":
    main()
