from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from datasets import Dataset, load_from_disk
import pandas as pd
import logging
import transformers
from itertools import chain
import evaluate

logger = logging.getLogger(__name__)


class WikiDataset():
    def __init__(self, tokenizer: AutoTokenizer, model_args, data_args, training_args) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args

        if training_args.model_max_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({training_args.model_max_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(training_args.model_max_length, tokenizer.model_max_length)
        import os

        path = data_args.data_path
        if not os.path.isdir(path):
            lm_datasets = load_from_disk(path)
        else:
            raw_datasets = self.load_data(path, training_args.cache_dir)
            print(f"RAW DATASETS: {raw_datasets}")
            if training_args.do_train:
                column_names = raw_datasets.column_names
                print(f"COLUMN NAMES: {column_names}")
            else:
                column_names = raw_datasets["test"].column_names
            self.text_column_name = "question" if "question" in column_names else column_names[0]
            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = raw_datasets.map(
                    self.tokenize_function,
                    batched=True,
                    batch_size=300,
                    num_proc=80,
                    remove_columns=["id", "system_prompt"],
                )
            tokenized_datasets.save_to_disk(path + "-tokenized")
            print(f"TOKENIZED DATASETS: {tokenized_datasets}")
            with training_args.main_process_first(desc="grouping texts together"):
                lm_datasets = tokenized_datasets.map(
                    self.group_texts,
                    batched=True,
                    batch_size=300,
                    num_proc=80,
                )
            lm_datasets.save_to_disk(path)

        if training_args.do_train:
            if not lm_datasets['response']:
                raise ValueError("--do_train requires a train dataset")
            self.train_dataset = lm_datasets['response']

        if training_args.do_predict:
            self.predict_dataset = self.train_dataset.train_test_split(test_size=0.1)['response']

        if training_args.do_eval:
            # if "validation" not in tokenized_datasets:
            #     raise ValueError("--do_eval requires a validation dataset")
            self.eval_dataset = self.predict_dataset

        # self.metric = evaluate.load(r"accuracy.py")
        self.metric = evaluate.load("accuracy")

        self.data_collator = default_data_collator
        if training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def load_data(self, data_args, cache_dir):

        import os
        csv_data = None
        json_files = [f for f in os.listdir(data_args) if f.endswith('.csv')]
        for file_path in json_files:
            with open(os.path.join(data_args, file_path), 'r') as f:
                csv_data = pd.read_csv(f)
        print(csv_data)
        raw_datasets = Dataset.from_pandas(pd.DataFrame(data=csv_data))
        # raw_datasets = raw_datasets.train_test_split(test_size=0.01)
        return raw_datasets

    def tokenize_function(self, examples):
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
        from transformers.testing_utils import CaptureLogger
        with CaptureLogger(tok_logger) as cl:
            output = self.tokenizer(examples[self.text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return self.metric.compute(predictions=preds, references=labels)

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.training_args.model_max_length:
            total_length = (total_length // self.training_args.model_max_length) * self.training_args.model_max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.training_args.model_max_length] for i in
                range(0, total_length, self.training_args.model_max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
