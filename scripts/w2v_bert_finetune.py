import torch
import argparse
from sconf import Config
import os
from os.path import basename
from pathlib import Path
import datetime
from datasets import load_dataset, load_metric, Audio, concatenate_datasets
from accelerate import Accelerator
import re
import json
import numpy as np
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor,\
    Wav2Vec2CTCTokenizer, Wav2Vec2BertForCTC, TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader

def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")

def save_vocab_json(vocab_dict, path):
    save_path = Path(path) / "vocab.json"
    with open(save_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)


def preprocess_dataset(dataset, config):
    # Precompile the regex pattern if 'remove_special_characters' is enabled
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]'

    def preprocess_fn(batch):
        try:
            if config.preprocessing.text.remove_special_characters:
                batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"])
            if config.preprocessing.text.lowercase:
                batch["sentence"] = batch["sentence"].lower()
            if config.preprocessing.text.remove_latin_characters:
                batch["sentence"] = re.sub(r'[a-z]+', '', batch["sentence"])
        except Exception as e:
            print(f"An error occurred in preprocessing: {e}")

        return batch

    # Apply the preprocessing function to the dataset
    return dataset.map(preprocess_fn)  # Ensure batched=True if the function expects batched inputs


def create_vocabulary(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def prepare_dataset(dynamic_datasets):
    combined_dataset = []

    for dataset_config in dynamic_datasets:
        try:
            dataset_name = dataset_config.name
            split = dataset_config.split
            audio_field = dataset_config.input_fields[0]
            text_field = dataset_config.input_fields[1]
            sampling_rate = config.sampling_rate

            # Load the dataset
            dataset = load_dataset(dataset_name, split=f"{split}[:100]")

            # Ensure the audio field exists before casting
            if audio_field in dataset.column_names:
                dataset = dataset.cast_column(audio_field, Audio(sampling_rate))
            else:
                raise ValueError(f"The audio field {audio_field} does not exist in the dataset {dataset_name}.")

            # Rename the text field if necessary
            if text_field in dataset.column_names and text_field != "sentence":
                dataset = dataset.rename_column(text_field, "sentence")
            elif text_field not in dataset.column_names:
                raise ValueError(f"The text field {text_field} does not exist in the dataset {dataset_name}.")

            # Remove unwanted columns
            required_columns = ["audio", "sentence"]
            columns_to_remove = set(dataset.column_names) - set(required_columns)
            dataset = dataset.remove_columns(columns_to_remove)

            combined_dataset.append(dataset)
        except Exception as e:
            # Instead of printing the error, you can raise it to exit the function
            raise Exception(f"An error occurred while preparing the dataset {dataset_name}: {e}")

    # Concatenate and shuffle datasets
    ds_to_return = concatenate_datasets(combined_dataset)
    ds_to_return = ds_to_return.shuffle(seed=22)

    ds_to_return = preprocess_dataset(ds_to_return, config)
    vocab = ds_to_return.map(create_vocabulary, batched=True, batch_size=-1, keep_in_memory=True,
                                         remove_columns=ds_to_return.column_names)

    return vocab, ds_to_return


@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch



wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    parser.add_argument("--debug", action="store_true")
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)
    config.debug = args.debug

    if not config.get("exp_name", False):
        config.exp_name = basename(args.config).split(".")[0]

    config.exp_version = (
     datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.exp_version
        else args.exp_version
    )
    save_path = Path(config.result_path) / config.exp_name / config.exp_version

    save_config_file(config, save_path)

    vocab_train, train_set = prepare_dataset(config.train_datasets)
    vocab_test, val_set = prepare_dataset(config.eval_datasets)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    save_vocab_json(vocab_dict, save_path)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(save_path, unk_token="[UNK]", pad_token="[PAD]",
                                                     word_delimiter_token="|")

    feature_extractor = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=16000,
                                                    padding_value=0.0)

    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


    def preprocess_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_features"])

        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch


    train_set = train_set.map(preprocess_dataset, remove_columns=train_set.column_names)



    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2BertForCTC.from_pretrained(
        "facebook/w2v-bert-2.0",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    training_args = TrainingArguments(
        output_dir=save_path,
        group_by_length=True,
        per_device_train_batch_size=config.train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        evaluation_strategy="steps",
        num_train_epochs=config.train_epochs,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=True,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        save_total_limit=config.save_total_limit,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()









