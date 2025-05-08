# put in src/f5_tts/train/datasets/prepare_emilia_v2.py
# prepares Emilia dataset with the new format w/ Emilia-YODAS

import json
import os
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_pinyin, repetition_found


# Define filters for exclusion
out_en = set()
zh_filters = ["い", "て"]
en_filters = ["ا", "い", "て"]


def process_audio_directory(audio_dir):
    sub_result, durations, vocab_set = [], [], set()
    bad_case_zh = 0
    bad_case_en = 0

    for file in audio_dir.iterdir():
        if file.suffix == ".json":
            with open(file, "r") as f:
                obj = json.load(f)

                text = obj["text"]
                if obj["language"] == "en":
                    if any(f in text for f in en_filters) or repetition_found(text, length=4):
                        bad_case_en += 1
                        continue
                elif obj["language"] == "zh":
                    if any(f in text for f in zh_filters) or repetition_found(text):
                        bad_case_zh += 1
                        continue
                    else:
                        text = text.translate(str.maketrans({",": "，", "!": "！", "?": "？"}))
                if tokenizer == "pinyin":
                    text = convert_char_to_pinyin([text], polyphone=polyphone)[0]

                duration = obj["duration"]
                audio_file = file.with_suffix(".mp3")
                if audio_file.exists():
                    sub_result.append({"audio_path": str(audio_file), "text": text, "duration": duration})
                    durations.append(duration)
                    vocab_set.update(list(text))

    return sub_result, durations, vocab_set, bad_case_zh, bad_case_en


def main():
    assert tokenizer in ["pinyin", "char"]
    result, duration_list, text_vocab_set = [], [], set()
    total_bad_case_zh = 0
    total_bad_case_en = 0

    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    for split in splits:
        for lang in langs:
            dataset_path = Path(os.path.join(dataset_dir, split, lang))
            for sub_dir in dataset_path.iterdir():
                if sub_dir.is_dir():
                    futures.append(executor.submit(process_audio_directory, sub_dir))

    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set, bad_case_zh, bad_case_en = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
        total_bad_case_zh += bad_case_zh
        total_bad_case_en += bad_case_en

    executor.shutdown()

    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"For {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")
    if "ZH" in langs:
        print(f"Bad zh transcription case: {total_bad_case_zh}")
    if "EN" in langs:
        print(f"Bad en transcription case: {total_bad_case_en}\n")


if __name__ == "__main__":
    max_workers = 32
    tokenizer = "pinyin"
    polyphone = True
    dataset_dir = "/apdcephfs_cq10/share_1297902/user/nenali/project/chukewang/data/Emilia-Dataset-test"
    splits = ["Emilia", "Emilia-YODAS"]
    langs = ["EN", "ZH"]
    dataset_name = f"Emilia_{'_'.join(langs)}_{tokenizer}"
    # save_dir = os.path.expanduser(f"~/F5-TTS/data/{dataset_name}")
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"

    print(f"Prepare for {dataset_name}, will save to {save_dir}\n")
    main()
