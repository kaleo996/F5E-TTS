from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import jieba
import torch
from pypinyin import Style, lazy_pinyin
from torch.nn.utils.rnn import pad_sequence


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    # for t in text:
    #     for c in t:
    #         if c not in vocab_char_map.keys():
    #             if c not in "[]/—\{\}":
    #                 print(f"character '{c}' not found in vocab, skipping")
    #                 continue
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


def get_g2p_mix_vocab():
    _pad = '_'

    # unstressed phoneme set
    english_phone_set = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F',
                        'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S',
                        'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
    mandarin_phone_set = ['a', 'b', 'c', 'ch', 'd', 'e', 'er', 'f',
                        'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'ng', 'o', 'p', 'q',
                        'r', 's', 'sh', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'zh']
    punc_set = [',', '.', '?', '!', ' ', '(', ')', ';', ':', '-', '\'', '\"', '，', '。', '、', '？', '！', '：', '；', '（', '）', '“', '”', '‘', '’', '—']
    unstressed_phone_set = [_pad] + mandarin_phone_set + english_phone_set + punc_set

    # phoneme with tones
    mandarin_finals = ['a', 'e', 'er', 'i', 'o', 'u', 'v', 'ng', 'n', 'm']
    mandarin_tones = ['0', '1', '2', '3', '4', '5']
    mandarin_tone_phones = [p + t for p in mandarin_finals for t in mandarin_tones] # Mandarin finals + tone
    english_finals = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
    english_tones = ['0', '1', '2']
    english_tone_phones = [p + t for p in english_finals for t in english_tones] # English vowels + accent
    
    # numbers
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # final vocab for g2p-mix tokenizer
    phone_set_with_tones = unstressed_phone_set + mandarin_tone_phones + english_tone_phones + numbers
    tone_phone_to_id = {p : i for i, p in enumerate(phone_set_with_tones)} # convert phone to id
    return tone_phone_to_id


# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char", "char-level-pinyin", "phone-level-pinyin"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    elif tokenizer == "g2p-mix":
        vocab_char_map = get_g2p_mix_vocab()
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# split a pinyin into 4 parts according to Chinese phonology
def split_pinyin(pinyin):
    onset = None # 声母
    medial = None # 介音
    rime = None # 韵腹
    coda = None # 韵尾
    
    valid_onsets = [
        "b", "p", "m", "f",
        "d", "t", "n", "l",
        "g", "k", "h",
        "j", "q", "x",
        "zh", "ch", "sh", "r",
        "z", "c", "s",
        "y", "w"
    ]
    for o in valid_onsets:
        if pinyin.startswith(o):
            onset = o
            pinyin = pinyin[len(o):]
            break

    valid_codas = ["n", "ng"]
    for c in valid_codas:
        if pinyin.endswith(c):
            coda = c
            pinyin = pinyin[:-len(c)]
            break

    valid_medials = ['i', 'u', 'ü']
    for m in valid_medials:
        if pinyin.startswith(m):
            medial = m
            pinyin = pinyin[len(m):]
            break

    rime = pinyin

    return [x for x in [onset, medial, rime, coda] if x is not None]


# convert char to finer pinyin
# `convert_char_to_pinyin` treat the pronunciation of a chinese character as one token, e.g. ['chuan1']
# but in this func, we split a Chinese syllable into at most 4 parts using `split_pinyin`, e.g. ['ch', 'u', 'ā', 'n']
def convert_char_to_finer_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        if char_list and char_list[-1] not in " :'\"":
                            char_list.append(" ")
                        char_list.extend([char + "_zh" for char in split_pinyin(seg_[i])])
                    else:
                        char_list.extend([char for char in seg_[i]])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        if char_list and char_list[-1] not in " :'\"":
                            char_list.append(" ")
                        syllable = lazy_pinyin(c, style=Style.TONE, tone_sandhi=True)[0]
                        pinyin_list = [char + "_zh" for char in split_pinyin(syllable)]
                        char_list.extend(pinyin_list)
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False
