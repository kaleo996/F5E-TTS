import json
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default
from f5_tts.ppg.wenet.dataset.feats import kaldiFbank


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        tokenizer=None,
        use_ppg=False
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        self.tokenizer = tokenizer
        self.use_ppg = use_ppg

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

        if self.tokenizer == "g2p-mix":
            from g2p_mix import G2pMix
            self.g2p = G2pMix()
        
        if use_ppg:
            self.featCal = kaldiFbank().eval()

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)
    
    # 分割汉语韵母
    def split_rime(self, rime):
        # 最后一个字符应该是阿拉伯数字表示的声调
        if not rime[-1].isdigit():
            raise ValueError("The last character of rime should be a digit in {}".format(rime))
        # 除了声调外，末尾两个字符如果是 er 或 ng，则和声调分一组
        if len(rime) >= 3 and rime[-3:-1] in ["er", "ng"]:
            last_group = rime[-3:]
            rime = rime[:-3]
        else: # 否则，末尾一个字符和声调分一组
            last_group = rime[-2:]
            rime = rime[:-2]
        # 前面的一个字符一组
        return [char for char in rime] + [last_group]

    # 处理一个 token
    def process_token(self, token):
        phone_list = token.phones
        # 如果是中文，需要把韵母分割到和 tokenizer 一样的粒度
        if token.lang == "ZH":
            phone_list = phone_list[:-1] + self.split_rime(phone_list[-1])
        # 如果是数字，则拆成单个阿拉伯数字
        if token.lang == "NUM":
            phone_list = [char for char in phone_list[0]]
        return phone_list

    def __getitem__(self, index):
        # import ipdb; ipdb.set_trace()
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            
            # filter if g2p-mix fail to generate phoneme
            if self.tokenizer == "g2p-mix":
                try:
                    text = text.replace(" n't", "n't") # LibriTTS puts spaces before n't, but g2p-mix does n't
                    g2p_list = self.g2p.g2p(text)
                    # no extra space before 1st word, add a space before every following word's phoneme list unless it's a punctuation
                    text = [phone for phone in self.process_token(g2p_list[0])] + [phone for token in g2p_list[1:] for phone in (self.process_token(token) if token.lang=="SYM" else [" "] + self.process_token(token))]
                except Exception as e:
                    print(f"Error occurred while g2p-mix processing text: {text}")
                    index = (index + 1) % len(self.data)
                    continue
            
            duration = row["duration"]

            # filter by given length
            if duration < 0.3 or duration > 30:
                print(f"Duration {duration} is out of range [0.3, 30]")
                index = (index + 1) % len(self.data)
                continue
            
            break  # valid
        
        data_item = dict(text=text)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
            data_item["mel_spec"] = mel_spec
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler_mel = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio_mel_target_sr = resampler_mel(audio)
            else:
                audio_mel_target_sr = audio

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio_mel_target_sr)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
            data_item["mel_spec"] = mel_spec
        
            if self.use_ppg:
                if source_sample_rate != 16000:
                    resampler_ppg = torchaudio.transforms.Resample(source_sample_rate, 16000)
                    audio_ppg_target_sr = resampler_ppg(audio)
                else:
                    audio_ppg_target_sr = audio # TODO 这里的 audio 是什么形状？需要 unsqueeze(0) 吗？
                feats, feats_len = self.featCal(audio_ppg_target_sr)
                data_item["mel_spec_for_ppg"] = feats[0].transpose(0,1)
        
        return data_item


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    use_ppg=False,
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    # TODO currently only support ppg in CustomDataset
    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            tokenizer=tokenizer,
            use_ppg=use_ppg,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])
    
    if len(batch)>0 and "mel_spec_for_ppg" in batch[0]:
        mel_specs_for_ppg = [item["mel_spec_for_ppg"].squeeze(0) for item in batch]
        mel_lengths_for_ppg = torch.LongTensor([spec.shape[-1] for spec in mel_specs_for_ppg])
        max_mel_length_for_ppg = mel_lengths_for_ppg.amax()
        
        padded_mel_specs_for_ppg = []
        for spec in mel_specs_for_ppg:
            padding = (0, max_mel_length_for_ppg - spec.size(-1))
            padded_spec = F.pad(spec, padding, value=0)
            padded_mel_specs_for_ppg.append(padded_spec)
            
        mel_specs_for_ppg = torch.stack(padded_mel_specs_for_ppg)
    else:
        mel_specs_for_ppg = None
        mel_lengths_for_ppg = None

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
        mel_for_ppg=mel_specs_for_ppg,
        mel_lengths_for_ppg=mel_lengths_for_ppg
    )
