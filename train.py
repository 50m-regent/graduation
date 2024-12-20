import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from matplotlib import pyplot

from utils import get_logger
from zero_pe import ZeroPE

logger = get_logger(__name__)


def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2").to("mps")
    model.transformer.wpe = ZeroPE(768, device=model.device)
    logger.info(model)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset("tatsu-lab/alpaca")["train"]

    def tokenize(examples: dict, max_length: int = 512) -> dict:
        return tokenizer(
            examples["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    test_dataset = dataset.select(range(1024)).map(
        tokenize, batched=True, batch_size=1024
    )
    valid_dataset = dataset.select(range(1024, 2048)).map(
        tokenize, batched=True, batch_size=1024
    )
    train_dataset = dataset.select(range(2048, len(dataset))).map(
        tokenize, batched=True, batch_size=1024
    )

    token_lengths = [len(data["input_ids"]) for data in train_dataset]
    pyplot.hist(token_lengths)
    pyplot.savefig("outputs/training/token_lengths.png")

    logger.info(test_dataset)
    logger.info(valid_dataset)
    logger.info(train_dataset)

    logger.info(test_dataset[0])

    trainer = Trainer(
        model,
        TrainingArguments(
            output_dir="outputs/training",
            num_train_epochs=10,
            do_train=True,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=16,
            logging_strategy="steps",
            logging_steps=16,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            load_best_model_at_end=True,
            save_steps=64,
        ),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
        ),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[EarlyStoppingCallback(3)],
    )
    trainer.train()


if __name__ == "__main__":
    main()
