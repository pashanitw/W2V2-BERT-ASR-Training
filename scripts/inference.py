import torch
import argparse
import os
from pathlib import Path
import shutil
from transformers import  Wav2Vec2BertProcessor,Wav2Vec2BertForCTC
import librosa

def inference(args):
    ckpt_dir_parent = str(Path(args.ckpt_dir).parent)
    if not os.path.exists(f"{args.ckpt_dir}/vocab.json"):
        shutil.copy2(f"{ckpt_dir_parent}/vocab.json", f"{args.ckpt_dir}/vocab.json")
    else:
        print(f"Loading vocab.json from {args.ckpt_dir}")

    model_id = args.ckpt_dir
    model = Wav2Vec2BertForCTC.from_pretrained(model_id).to("cuda")
    processor = Wav2Vec2BertProcessor.from_pretrained(model_id, unk_token="[UNK]", pad_token="[PAD]",
                                                      word_delimiter_token="|")

    with torch.no_grad():
        wav, sr = librosa.load(args.audio_file)
        input_features = processor(wav, sampling_rate=sr, return_tensors="pt").input_features[0]
        input_features = input_features.to("cuda").unsqueeze(0)
        logits = model(input_features).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        pred_text = processor.decode(pred_ids)
        print(pred_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Folder with the pytorch_model.bin file",
    )

    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        default="mozilla-foundation/common_voice_11_0",
        help="Audio File",
    )


    args = parser.parse_args()

    inference(args)