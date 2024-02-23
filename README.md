![Work in Progress](https://img.shields.io/badge/Work%20in%20Progress-Yes-yellow)

## This Repository is a Work in Progress (WIP)

This repository is currently under active development. Please note that some features may be incomplete or subject to change.

# W2V2-BERT Fine-tuning 

This repository contains code to fine-tune the W2V2-BERT model for Automatic Speech Recognition (ASR).

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

### Preprocessing the Dataset

To preprocess the dataset, run the following command:

```bash
python scripts/preprocess_hf_dataset.py --config configs/w2v_bert_hf.yaml --preprocessed_dataset preprocessed
```

### Training

To train the model, execute the following command:

```bash
python scripts/w2v_bert_finetune.py --config configs/w2v_bert_hf.yaml --preprocessed_dataset preprocessed
```

## Configuration

The repository includes configuration files (`configs/w2v_bert_hf.yaml`) with the following parameters:

- `train_datasets`: A list of datasets used for training the model.
- `eval_datasets`: A list of datasets used for evaluating the model.
- `preprocessing`: Parameters for text preprocessing.
- `train_batch_size`: Batch size for training.
- `num_workers`: Number of workers for data loading.
- `debug`: Debug mode flag.
- `result_path`: Path to save results.
- `exp_name`: Experiment name.
- `sampling_rate`: Sampling rate for audio data.
- `warmup_steps`: Number of warmup steps for learning rate scheduling.
- `learning_rate`: Learning rate for training.
- `save_steps`: Frequency of saving checkpoints.
- `eval_steps`: Frequency of evaluating the model.
- `logging_steps`: Frequency of logging training information.
- `save_total_limit`: Maximum number of checkpoints to save.
- `push_to_hub`: Flag for pushing checkpoints to the Hugging Face Hub.
- `gradient_checkpointing`: Flag for using gradient checkpointing.
- `train_epochs`: Number of training epochs.
- `gradient_accumulation_steps`: Number of gradient accumulation steps.

You can modify these parameters in the configuration file according to your requirements.

## Evaluation
- `--ckpt_dir`: Required. The directory containing the pytorch_model.bin file of the model you wish to evaluate.
- `--dataset`: Required. The dataset identifier on the Hugging Face Hub to evaluate the model on. Default is mozilla-foundation/common_voice_11_0.
- `--name`: Required. The configuration of the dataset. For example, use 'hi' for the Hindi split of the Common Voice dataset.
- `--split`: The specific split of the dataset to evaluate on. Default is test.

### Example command 

``` bash
python scripts/eval_on_hf_dataset.py \
  --ckpt_dir path/to/your/model_checkpoint_directory \
  --dataset google/fleurs \
  --name ka_ge \
  --split test
```
## Push To Hub
If everything appears to be in order and you wish to push to the hub, execute the following command:

``` bash
python push_to_hub.py \
--ckpt_dir path/to/your/model_checkpoint_directory \
--name model_name
```