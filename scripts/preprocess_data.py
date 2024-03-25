from datasets import Dataset
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


MIN_DURATION_IN_SECONDS = 1.0
def is_audio_length_in_range(input_length):
    return input_length > MIN_DURATION_IN_SECONDS
def save_vocab_json(vocab_dict, path):
    save_path = Path(path) / "vocab.json"
    with open(save_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)


def clean_up_data(batch, config):
    # Precompile the regex pattern if 'remove_special_characters' is enabled
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«\]\[\_]'

    try:

        sentence = batch["sentence"]

        # Remove leading and trailing whitespace
        sentence = sentence.strip()

        # Replace newlines and carriage returns with a space (or you could use '')
        sentence = sentence.replace('\n', ' ')

        if config.preprocessing.text.remove_special_characters:
            sentence = re.sub(chars_to_remove_regex, '', sentence)
        if config.preprocessing.text.lowercase:
            sentence = sentence.lower()
        if config.preprocessing.text.remove_latin_characters:
            sentence = re.sub(r'[a-z]+', '', sentence)

        # Update the batch with the cleaned sentence
        batch["sentence"] = sentence

    except Exception as e:
        print(f"An error occurred in preprocessing: {e}")

    return batch

def normalize_dataset(dataset, config):

    cleanup_fn = partial(clean_up_data, config=config)
    # Apply the preprocessing function to the dataset
    return dataset.map(cleanup_fn, num_proc=config.num_workers)  # Ensure batched=True if the function expects batched inputs


def create_vocabulary(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def prepare_dataset(dynamic_datasets, num_workers):
    combined_dataset = []

    for dataset_config in dynamic_datasets:
        try:
            dataset = dataset_config.dataset
            split = dataset_config.split
            audio_field = dataset_config.input_fields[0]
            text_field = dataset_config.input_fields[1]
            sampling_rate = config.sampling_rate
            if "custom" in dataset_config and dataset_config.custom:
                dataset = create_custom_dataset(dataset, split)
            else:
                # Load the dataset
                if "name" in dataset_config:
                    dataset = load_dataset(dataset, dataset_config.name, split=split)
                else:
                    dataset = load_dataset(dataset, split=split)
                    # dataset = load_dataset(dataset_name, split=f"{split}[:1000]")


            # Ensure the audio field exists before casting
            if audio_field in dataset.column_names:
                dataset = dataset.cast_column(audio_field, Audio(sampling_rate))
            else:
                raise ValueError(f"The audio field {audio_field} does not exist in the dataset {dataset}.")

            # Rename the text field if necessary
            if text_field in dataset.column_names and text_field != "sentence":
                dataset = dataset.rename_column(text_field, "sentence")
            elif text_field not in dataset.column_names:
                raise ValueError(f"The text field {text_field} does not exist in the dataset {dataset}.")

            # Remove unwanted columns
            required_columns = ["audio", "sentence"]
            columns_to_remove = set(dataset.column_names) - set(required_columns)
            dataset = dataset.remove_columns(columns_to_remove)

            combined_dataset.append(dataset)
        except Exception as e:
            # Instead of printing the error, you can raise it to exit the function
            raise Exception(f"An error occurred while preparing the dataset {dataset}: {e}")

    # Concatenate and shuffle datasets
    ds_to_return = concatenate_datasets(combined_dataset)
    ds_to_return = ds_to_return.shuffle(seed=22)

    ds_to_return = normalize_dataset(ds_to_return, config)
    vocab = ds_to_return.map(create_vocabulary, batched=True, batch_size=-1, keep_in_memory=False,
                                         remove_columns=ds_to_return.column_names, num_proc=num_workers)

    return vocab, ds_to_return


def preprocess_dataset(batch, processor):
    audio = batch["audio"]
    audio_length_seconds = len(audio["array"]) / audio["sampling_rate"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])
    batch["length_in_seconds"] = audio_length_seconds
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


def create_custom_dataset(input_directory, split):
    audio_dict = []
    sentence_dict = []

    # Walk through the input directory
    for root, dirs, files in os.walk(f"{input_directory}/{split}"):
        for file in files:
            # Check if the current file is a WAV file
            if file.endswith('.wav'):
                # Get the base name of the file (without extension)
                base_name = os.path.splitext(file)[0]
                # Construct the path for the corresponding text file
                text_file_path = os.path.join(root, f"{base_name}.txt")

                # Check if the text file exists
                if os.path.exists(text_file_path):
                    # Read and print the content of the text file
                    with open(text_file_path, 'r', encoding='utf-8') as text_file:
                        content = text_file.read()
                        audio_dict.append(os.path.join(root, f"{base_name}.wav"))
                        sentence_dict.append(content)
                else:
                    print(f"Text file for {file} not found.")


    return Dataset.from_dict({
        "audio": audio_dict,
        "sentence": sentence_dict
    }).cast_column("audio", Audio())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--preprocessed_dataset", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)

    recreate_directory(args.preprocessed_dataset)


    vocab_train, train_set = prepare_dataset(config.train_datasets, config.num_workers)
    vocab_test, val_set = prepare_dataset(config.eval_datasets, config.num_workers)
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

    train_set = train_set.map(preprocess_fn, remove_columns=columns_to_remove, num_proc=config.num_workers, load_from_cache_file=False)
    val_set = val_set.map(preprocess_fn, remove_columns=columns_to_remove,  num_proc=config.num_workers, load_from_cache_file=False)

    train_set.save_to_disk(Path(args.preprocessed_dataset) / "train")
    val_set.save_to_disk(Path(args.preprocessed_dataset) / "val")

    train_set.cleanup_cache_files()
    val_set.cleanup_cache_files()
