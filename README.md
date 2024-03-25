# W2V2-BERT 2.0 Fine-tuning for ASR 

This repository contains code to fine-tune the W2V-BERT-2.0 model for Automatic Speech Recognition (ASR).

## Prerequisites

Before running the code, ensure you have the following prerequisites installed:

- CUDA 11.8
- torch >= 2.0
- torchaudio

## Installation

1. Clone the repository:

```bash
git clone https://github.com/pashanitw/W2V2-BERT-ASR-Training.git
cd W2V2-BERT-ASR-Training
```

2. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## Training Process Overview

The training process is divided into four main steps, each crucial for the successful training of your model. Follow these steps sequentially:

### Step 1: Authenticate with Hugging Face

you can log in to your Hugging Face account by opening a terminal and running the following command:
```bash
huggingface-cli login
````
### Step 2: Configure Your Training


- **Configure Your Training**: Prepare your training configuration by creating or updating a YAML file (`config.yaml`). This file should include your datasets' paths, training parameters, and any model-specific settings. Ensure the configuration aligns with your project needs and the datasets you plan to use.

### Step 3: Preprocess the Dataset

To preprocess the dataset, run the following command:

```bash
python scripts/preprocess_data.py --config configs/w2v_bert_ka_hf.yaml --preprocessed_dataset preprocessed
```


### Step 4: Initiate Training
To train the model, execute the following command:

```bash
python scripts/w2v_bert_finetune.py --config configs/w2v_bert_ka_hf.yaml --preprocessed_dataset preprocessed
```


## Configuration Parameters

The configuration of the model training and evaluation is defined by the following parameters:

### `train_datasets` and `eval_datasets`
Specifies the datasets used for training and evaluating the model, respectively. Each entry in these lists consists of the following fields:

- `dataset`: The path to the dataset.
- `custom`: Indicates whether the dataset is a custom dataset (`True`) or not.
- `split`: The dataset split to use, e.g., `train`, `test`.
- `input_fields`: The fields to be used as input from the dataset. For training datasets, these are typically `"audio"` and `"sentence"`.

#### Example of `train_datasets` combining multiple datasets:

```yaml
train_datasets:
  - dataset: "mozilla-foundation/common_voice_16_0"
    name: "ka"
    split: "train+validation"
    input_fields:
      - "audio"
      - "sentence"
  - dataset: "mozilla-foundation/common_voice_16_0"
    name: "ka"
    split: "test"
    input_fields:
      - "audio"
      - "sentence"

```

#### Example of `eval_datasets`:
```yaml
eval_datasets:
  - dataset: "google/fleurs"
    name: "ta_in"
    split: "test"
    input_fields: ["audio", "transcription"]
```


Each dataset directory contains the respective splits as indicated.

### preprocessing
Defines the text preprocessing parameters:
- `remove_special_characters`: Whether to remove special characters from the text.
- `lowercase`: Whether to convert all characters to lowercase.
- `remove_punctuation`: Whether to remove punctuation marks from the text.
- `remove_latin_characters`: Whether to remove Latin characters from the text.

### Other Parameters
- `pretrained_model`: The path or identifier of the pretrained model to use, e.g., `"facebook/w2v-bert-2.0"`.
- `train_batch_size`: The batch size to use during training.
- `eval_batch_size`: The batch size to use during evaluation.
- `num_workers`: The number of worker threads for loading data.
- `result_path`: The path where results should be saved.
- `exp_name`: The name of the experiment.
- `sampling_rate`: The sampling rate for audio data.
- `warmup_steps`: The number of warmup steps for learning rate scheduling.
- `learning_rate`: The learning rate for training.
- `save_steps`, `eval_steps`, `logging_steps`: The frequency of saving checkpoints, evaluating the model, and logging training information, respectively.
- `save_total_limit`: The maximum number of checkpoints to save.
- `push_to_hub`: Whether to push checkpoints to the Hugging Face Hub.
- `gradient_checkpointing`: Enable gradient checkpointing to reduce memory usage.
- `train_epochs`: The number of training epochs.
- `gradient_accumulation_steps`: The number of steps over which gradients are accumulated.
- `resume_from_checkpoint`: Whether to resume training from a checkpoint.
- `ckpt_dir_path`: The directory path for saving or loading checkpoints.

This configuration allows for flexible and detailed setup of model training and evaluation, tailored to specific needs and datasets.

## Training on Custom Datasets

To train the model on custom datasets, you need to organize your data into specific directories and ensure each audio file has a corresponding text file. Here’s how to set it up:

### Custom Dataset Structure

For the model to correctly process your custom datasets, each dataset should reside in its designated directory. Inside these directories, every unique audio file (in WAV format) must have a corresponding text file. The text file should have the same name as the audio file, ensuring they can be correctly paired during training and testing.

### Configuring Custom Datasets

In the configuration, specify your custom datasets as follows:

```yaml
train_datasets:
  - dataset: "./custom_data_dir_1"
    custom: True
    split: "train"
    input_fields:
      - "audio"
      - "sentence"

  - dataset: "./custom_data_dir_2"
    custom: True
    split: "test"
    input_fields:
      - "audio"
      - "sentence"
```

### Directory and File Creation

1. **Create the custom data directories**: Make directories named `custom_data_dir_1` and `custom_data_dir_2` in your project folder.
2. **Add audio and text files**: In each directory, place your audio files (`.wav`) along with their corresponding text files. The text file should contain the transcription of the audio and share the same name as the audio file, differing only in the file extension (`.txt` for text files).

### Example Hierarchical Structure

To give you a clear picture, here’s how your dataset structure should look:

```
.
├── custom_data_dir_1
│   ├── audiofile1.wav
│   ├── audiofile1.txt
│   ├── audiofile2.wav
│   ├── audiofile2.txt
│   └── ...
└── custom_data_dir_2
    ├── audiofile1.wav
    ├── audiofile1.txt
    ├── audiofile2.wav
    ├── audiofile2.txt
    └── ...
```

### Reference for custom data

For a sample of how your custom data should be structured, refer to the `custom_data` folder. This folder contains example audio and text files arranged according to the specifications mentioned above. It serves as a guide for preparing your datasets for training and testing.
## Evaluation
This part explains how to check the model's performance using Hugging Face datasets.
### Example command
``` bash
python scripts/eval_on_hf_dataset.py \
  --ckpt_dir path/to/your/model_checkpoint_directory \
  --dataset google/fleurs \
  --name ka_ge \
  --split test
```
- `--ckpt_dir`: Required. The directory containing the pytorch_model.bin file of the model you wish to evaluate.
- `--dataset`: Required. The dataset identifier on the Hugging Face Hub to evaluate the model on. Default is mozilla-foundation/common_voice_11_0.
- `--name`: Required. The configuration of the dataset. For example, use 'hi' for the Hindi split of the Common Voice dataset.
- `--split`: The specific split of the dataset to evaluate on. Default is test.

## Resuming Training from a Checkpoint
To resume the training process from a previously saved checkpoint, you need to modify the YAML configuration file. Update the following parameters:

- `resume_from_checkpoint`: Set this parameter to `True` to enable resuming from a checkpoint.
- `ckpt_dir_path`: Specify the path to the directory where the checkpoint files are stored.

### Example Configuration

Here's an example of how you can update the YAML configuration to resume training from a checkpoint:

```yaml
resume_from_checkpoint: True
ckpt_dir_path: "/path/to/results/folder"
```
## Push To Hub
If everything appears to be in order and you wish to push to the hub, execute the following command:

``` bash
python push_to_hub.py \
--ckpt_dir path/to/your/model_checkpoint_directory \
--name model_name
```