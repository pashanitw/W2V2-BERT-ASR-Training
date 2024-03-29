import argparse
import os
from pathlib import Path
import shutil
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor


def main(args):
    ckpt_dir_parent = str(Path(args.ckpt_dir).parent)
    if not os.path.exists(f"{args.ckpt_dir}/vocab.json"):
        shutil.copy2(f"{ckpt_dir_parent}/vocab.json", f"{args.ckpt_dir}/vocab.json")
    else:
        print(f"Loading vocab.json from {args.ckpt_dir}")

    model_id = args.ckpt_dir
    model = Wav2Vec2BertForCTC.from_pretrained(model_id)
    processor = Wav2Vec2BertProcessor.from_pretrained(model_id,  unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

    model.push_to_hub(args.name)
    processor.push_to_hub(args.name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Folder with the pytorch_model.bin file",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Model name that you want to save",
    )

    args = parser.parse_args()
    main(args)