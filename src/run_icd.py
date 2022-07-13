# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
import torch
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from modeling_bert import BertForMultilabelClassification
from modeling_roberta import RobertaForMultilabelClassification
from modeling_longformer import LongformerForMultilabelClassification
from evaluation import all_metrics


logger = logging.getLogger(__name__)


MODELS_CLASSES = {
    'bert': BertForMultilabelClassification,
    'roberta': RobertaForMultilabelClassification,
    'longformer': LongformerForMultilabelClassification
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--code_file", type=str, default=None, help="A txt file containing all codes."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help=(
            "The size of chunks that we'll split the inputs into"
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="The type of model",
        required=True,
        choices=["bert", "roberta", "longformer"]
    )
    parser.add_argument(
        "--model_mode",
        type=str,
        help="Specify how to aggregate output in the model",
        required=True,
        choices=["cls-sum", "cls-max", "laat", "laat-split"]
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--cased",
        action="store_true",
        help="equivalent to do_lower_case=False",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--code_50", action='store_true', help="use only top-50 codes"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    labels = set()
    all_codes_file = "../data/mimic3/ALL_CODES.txt" if not args.code_50 else "../data/mimic3/ALL_CODES_50.txt"
    if args.code_file is not None:
        all_codes_file = args.code_file

    with open(all_codes_file, "r") as f:
        for line in f:
            if line.strip() != "":
                labels.add(line.strip())
    label_list = sorted(list(labels))
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    if args.model_type == "longformer":
        config.attention_window = args.chunk_size
    elif args.model_type in ["bert", "roberta"]:
        config.model_mode = args.model_mode
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        do_lower_case=not args.cased)
    model_class = MODELS_CLASSES[args.model_type]
    if args.num_train_epochs > 0:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        model = model_class.from_pretrained(
            args.output_dir,
            config=config,
        )

    sentence1_key, sentence2_key = "text", None

    label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True, add_special_tokens="cls" not in args.model_mode)
        if "label" in examples:
            result["labels"] = examples["label"]
            result["label_ids"] = [[label_to_id[label.strip()] for label in labels.strip().split(';') if label.strip() != ""] if labels is not None else [] for labels in examples["label"]]
        return result

    remove_columns = raw_datasets["train"].column_names if args.train_file is not None else raw_datasets["validation"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )

    eval_dataset = processed_datasets["validation"]

    if args.num_train_epochs > 0:
        train_dataset = processed_datasets["train"]
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            logger.info(f"Original tokens: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    def data_collator(features):
        batch = dict()

        if "cls" in args.model_mode:
            for f in features:
                new_input_ids = []
                for i in range(0, len(f["input_ids"]), args.chunk_size - 2):
                    new_input_ids.extend([tokenizer.cls_token_id] + f["input_ids"][i:i+(args.chunk_size)-2] + [tokenizer.sep_token_id])
                f["input_ids"] = new_input_ids
                f["attention_mask"] = [1] * len(f["input_ids"])
                f["token_type_ids"] = [0] * len(f["input_ids"])

        max_length = max([len(f["input_ids"]) for f in features])
        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size

        batch["input_ids"] = torch.tensor([
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]).contiguous().view((len(features), -1, args.chunk_size))
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.tensor([
                f["attention_mask"] + [0] * (max_length - len(f["attention_mask"]))
                for f in features
            ]).contiguous().view((len(features), -1, args.chunk_size))
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor([
                f["token_type_ids"] + [0] * (max_length - len(f["token_type_ids"]))
                for f in features
            ]).contiguous().view((len(features), -1, args.chunk_size))
        label_ids = torch.zeros((len(features), len(label_list)))
        for i, f in enumerate(features):
            for label in f["label_ids"]:
                label_ids[i, label] = 1
        batch["labels"] = label_ids
        return batch

    if args.num_train_epochs > 0:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, eval_dataloader = accelerator.prepare(
        model, optimizer, eval_dataloader
    )
    if args.num_train_epochs > 0:
        train_dataloader = accelerator.prepare(train_dataloader)

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)

    if args.num_train_epochs > 0:
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in tqdm(range(args.num_train_epochs)):
            model.train()
            epoch_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                epoch_loss += loss.item()
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_postfix(loss=epoch_loss / completed_steps)

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            all_preds = []
            all_preds_raw = []
            all_labels = []
            for step, batch in tqdm(enumerate(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                preds_raw = outputs.logits.sigmoid().cpu()
                preds = (preds_raw > 0.5).int()
                all_preds_raw.extend(list(preds_raw))
                all_preds.extend(list(preds))
                all_labels.extend(list(batch["labels"].cpu().numpy()))
            
            all_preds_raw = np.stack(all_preds_raw)
            all_preds = np.stack(all_preds)
            all_labels = np.stack(all_labels)
            metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
            logger.info(f"epoch {epoch} finished")
            logger.info(f"metrics: {metrics}")
    
    if args.num_train_epochs == 0 and accelerator.is_local_main_process:
        model.eval()
        all_preds = []
        all_preds_raw = []
        all_labels = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            preds_raw = outputs.logits.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["labels"].cpu().numpy()))
        
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        logger.info(f"evaluation finished")
        logger.info(f"metrics: {metrics}")
        for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            all_preds = (all_preds_raw > t).astype(int)
            metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5,8,15])
            logger.info(f"metrics for threshold {t}: {metrics}")

    if args.output_dir is not None and args.num_train_epochs > 0:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
