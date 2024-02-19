import torch
import argparse
from sconf import Config
import os
from os.path import basename
from pathlib import Path
import datetime
from datasets import load_dataset, load_metric, load_from_disk, DatasetDict, Audio
import numpy as np
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor,\
    Wav2Vec2CTCTokenizer, Wav2Vec2BertForCTC, TrainingArguments, Trainer, HfArgumentParser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Required

def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


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


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    config: Optional[str] = field(metadata={"help": "training config file"})
    preprocessed_dataset: Optional[str] = field( metadata={"help": "preprocessed hugginface dataset"})
    exp_version: Optional[str] = field(default="", metadata={"help": "experiment version"})
    config_file: Optional[str] = field(default="", metadata={"help": "please provide if there is a config file !"})



parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


config = Config(script_args.config)

if not config.get("exp_name", False):
    config.exp_name = basename(script_args.config).split(".")[0]

config.exp_version = (
 datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not script_args.exp_version
    else script_args.exp_version
)
save_path = Path(config.result_path) / config.exp_name / config.exp_version

save_config_file(config, save_path)


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(script_args.preprocessed_dataset, unk_token="[UNK]", pad_token="[PAD]",
                                                 word_delimiter_token="|")

feature_extractor = SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=16000,
                                                padding_value=0.0)

processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


print('DATASET PREPARATION IN PROGRESS...')
raw_dataset = DatasetDict()
raw_dataset["train"] = load_from_disk(f"{script_args.preprocessed_dataset}/train")
raw_dataset["eval"] = load_from_disk(f"{script_args.preprocessed_dataset}/val")
print(raw_dataset["train"].column_names)
raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))


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
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    tokenizer=processor.feature_extractor,
)

trainer.train()






