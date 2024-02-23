import torch
import argparse
import os
from pathlib import Path
import evaluate
from tqdm import tqdm
import shutil
from datasets import load_dataset, Audio
from transformers import  Wav2Vec2BertProcessor,Wav2Vec2BertForCTC

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""

def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )

def get_text_column_names(column_names):
    if "text" in column_names:
        return "text"
    elif "sentence" in column_names:
        return "sentence"
    elif "normalized_text" in column_names:
        return "normalized_text"
    elif "transcript" in column_names:
        return "transcript"
    elif "transcription" in column_names:
        return "transcription"

def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": get_text(item)}


def main(args):
    ckpt_dir_parent = str(Path(args.ckpt_dir).parent)
    if not os.path.exists(f"{args.ckpt_dir}/vocab.json"):
        shutil.copy2(f"{ckpt_dir_parent}/vocab.json", f"{args.ckpt_dir}/vocab.json")
    else:
        print(f"Loading vocab.json from {args.ckpt_dir}")

    model_id = args.ckpt_dir
    model = Wav2Vec2BertForCTC.from_pretrained(model_id).to("cuda")
    processor = Wav2Vec2BertProcessor.from_pretrained(model_id,  unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        use_auth_token=True,
    )
    text_column_name = get_text_column_names(dataset.column_names)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.filter(is_target_text_in_range, input_columns=[text_column_name], num_proc=2)
    predictions = []
    references = []
    with torch.no_grad():
        for item in tqdm(data(dataset), total=len(dataset), desc='Decode Progress'):
            input_features = processor(item["array"], sampling_rate=item["sampling_rate"], return_tensors="pt").input_features[0]
            input_features = input_features.to("cuda").unsqueeze(0)
            logits = model(input_features).logits
            pred_ids = torch.argmax(logits, dim=-1)[0]
            pred_text = processor.decode(pred_ids)

            predictions.append(pred_text)
            references.append(item["reference"])

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)
    cer = cer_metric.compute(references=references, predictions=predictions)
    cer = round(100 * cer, 2)
    print("\nWER : ", wer)
    print("CER : ", cer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Folder with the pytorch_model.bin file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset from huggingface to evaluate the model on. Example: mozilla-foundation/common_voice_11_0",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Config of the dataset. Eg. 'hi' for the Hindi split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
        help="Split of the dataset. Eg. 'test'",
    )

    args = parser.parse_args()
    main(args)