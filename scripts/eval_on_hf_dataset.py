import torch
import os
import argparse
import evaluate
from tqdm import tqdm
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
    model_id = args.hf_model
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
    # dataset.set_format(type='torch', columns=['audio'])
    predictions = []
    references = []
    with torch.no_grad():
        for item in tqdm(data(dataset), total=len(dataset), desc='Decode Progress'):
            input_features = processor(item["array"], sampling_rate=item["sampling_rate"], return_tensors="pt").input_features[0]
            input_features = input_features.to("cuda").unsqueeze(0)
            print(input_features)
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
        "--hf_model",
        type=str,
        required=True,
        help="Huggingface model name or trained checkpoint path.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=False,
        default=".",
        help="Folder with the pytorch_model.bin file",
    )
    parser.add_argument(
        "--temp_ckpt_folder",
        type=str,
        required=False,
        default="temp_dir",
        help="Path to create a temporary folder containing the model and related files needed for inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=False,
        default="hi",
        help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi. This helps initialize the tokenizer.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset from huggingface to evaluate the model on. Example: mozilla-foundation/common_voice_11_0",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="hi",
        help="Config of the dataset. Eg. 'hi' for the Hindi split of Common Voice",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
        help="Split of the dataset. Eg. 'test'",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="The device to run the pipeline on. -1 for CPU, 0 for the first GPU (default) and so on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=16,
        help="Number of samples to go through each streamed batch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="predictions_dir",
        help="Output directory for the predictions and hypotheses generated.",
    )

    args = parser.parse_args()
    main(args)