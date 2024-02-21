import argparse
from sconf import Config
import os
from pathlib import Path
from datasets import load_dataset,Audio, concatenate_datasets

import re
import json
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor,\
    Wav2Vec2CTCTokenizer
import shutil
from functools import partial


def save_vocab_json(vocab_dict, path):
    save_path = Path(path) / "vocab.json"
    with open(save_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)

def clean_up_data(batch, config):
    # Precompile the regex pattern if 'remove_special_characters' is enabled
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]'
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


def normalize_dataset(dataset, config):

    cleanup_fn = partial(clean_up_data, config=config)
    # Apply the preprocessing function to the dataset
    return dataset.map(cleanup_fn, num_proc=8)  # Ensure batched=True if the function expects batched inputs


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
            dataset = load_dataset(dataset_name, split=split)
            # dataset = load_dataset(dataset_name, split=f"{split}[:1000]")


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

    ds_to_return = normalize_dataset(ds_to_return, config)
    vocab = ds_to_return.map(create_vocabulary, batched=True, batch_size=-1, keep_in_memory=False,
                                         remove_columns=ds_to_return.column_names, num_proc=8)

    return vocab, ds_to_return


def preprocess_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])

    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

def recreate_directory(dir_path):
    # Check if the directory exists
    if os.path.exists(dir_path):
        # Remove the existing directory
        shutil.rmtree(dir_path)
        print(f"Directory {dir_path} has been removed.")

    # Create a new directory
    os.makedirs(dir_path)
    print(f"Directory {dir_path} has been created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--preprocessed_dataset", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)

    recreate_directory(args.preprocessed_dataset)


    vocab_train, train_set = prepare_dataset(config.train_datasets)
    vocab_test, val_set = prepare_dataset(config.eval_datasets)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    save_vocab_json(vocab_dict, args.preprocessed_dataset)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.preprocessed_dataset, unk_token="[UNK]", pad_token="[PAD]",
                                                     word_delimiter_token="|")

    feature_extractor = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=16000,
                                                    padding_value=0.0)

    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)



    preprocess_fn = partial(preprocess_dataset, processor=processor)


    columns_to_remove = ["audio", "sentence"]

    train_set = train_set.map(preprocess_fn, remove_columns=columns_to_remove, num_proc=16, load_from_cache_file=False)
    val_set = val_set.map(preprocess_fn, remove_columns=columns_to_remove,  num_proc=16, load_from_cache_file=False)

    train_set.save_to_disk(Path(args.preprocessed_dataset) / "train")
    val_set.save_to_disk(Path(args.preprocessed_dataset) / "val")

    train_set.cleanup_cache_files()
    val_set.cleanup_cache_files()











