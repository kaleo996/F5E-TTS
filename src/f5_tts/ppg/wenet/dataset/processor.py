# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
import random
import re
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
            sample: {'src':
            '/apdcephfs/share_1149801/speech_user/nenali/private_nenali/corpus/AIshell/shards/train/shards/shards_000000119.tar',
            'rank': 0, 'world_size': 1, 'worker_id': 0, 'num_workers': 1,
            'stream': <_io.BufferedReader
            name='/apdcephfs/share_1149801/speech_user/nenali/private_nenali/corpus/AIshell/shards/train/shards/shards_000000119.tar'>}
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')#url: tar file path
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, spk_id, wav, txt, sample_rate, (spks, segs)}]
            example.keys(): dict_keys(['txt', 'wav', 'sample_rate',
            'key','spk_id']), in
            fetch order
    """
    for sample in data:#sample is a dict
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}  # dictionary to store data
        valid = True
        for tarinfo in stream:#files in a tar file
            name = tarinfo.name
            # print(name)
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix == 'txt':
                        example['txt'] = file_obj.read().decode('utf8').strip()
                    elif postfix == 'spk_id':
                        example['spk_id'] = int(file_obj.read().decode('utf8').strip())
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    # extract spk seg information 
                    elif postfix in 'spk_seg':
                        strings = file_obj.read().decode('utf8')
                        strings = strings.split('\n')[:-1]
                        spks = []
                        segs = []
                        for seg in strings:
                            seg = seg.split()
                            spks.append(seg[0])     # spkid
                            segs.append([float(seg[1]), float(seg[2])])  # start time
                        segs = torch.tensor(segs)
                        start_time = segs.min()
                        segs = segs - start_time # cal relative time
                        example['spks'] = spks
                        example['segs'] = segs
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
                    exit
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']
        try:
            if 'start' in obj:
                assert 'end' in obj
                sample_rate = torchaudio.backend.sox_io_backend.info(
                    wav_file).sample_rate
                start_frame = int(obj['start'] * sample_rate)
                end_frame = int(obj['end'] * sample_rate)
                waveform, _ = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_file,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(key=key,
                           txt=txt,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def filter(data,
           max_length=10240, # 102second
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate, (spks, segs)}]
            max_length: drop utterance which is greater than max_length(10ms per frame)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate, (spks, segs)}]
    """
    # import ipdb; ipdb.set_trace()
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            print(sample['key'], 'filter:', int(num_frames) )
            continue
        if len(sample['label']) < token_min_length:
            continue
        if len(sample['label']) > token_max_length:
            continue
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['label']) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate, (spks, segs)}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            # import ipdb; ipdb.set_trace()
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav
            if 'segs' in sample:
                # speed = 0.9, wav len become longer
                ratio = waveform.shape[-1]/float(wav.shape[-1])
                assert (ratio - speed) < 0.01
                sample['segs'] /= ratio
        # TODO: need to test
        if 'spks' in sample:
            sp = f"{(round(speed*10)):02d}"
            # since the sp in emb is reversed
            if sp == '11':
                sp = '09'
            elif sp == '09':
                sp = '11'
            sample['spks'] = [spk+'sp'+sp for spk in sample['spks']]

        yield sample

def wav_aug(data,
            augmentor):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            augmentor: augmentor

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'wav' in sample
        waveform = sample['wav']
        waveform = waveform.detach().numpy()#1xtlen
        wav_len = waveform.shape[1]
        augtype = random.randint(1,10)
        if augtype == 1:
            waveform = augmentor.reverberate(waveform)
        elif augtype == 2:
            waveform = augmentor.reverberate(waveform)
        elif augtype == 3:
            waveform = augmentor.additive_noise('music',waveform)
        elif augtype == 4:
            waveform = augmentor.additive_noise('speech',waveform)
        elif augtype == 5:
            waveform = augmentor.additive_noise('noise',waveform)
        assert wav_len == waveform.shape[1]    
        sample['wav'] = torch.from_numpy(waveform)
        yield sample

def compute_fbank(data,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate, (spks, segs)}]

        Returns:
            Iterable[{key, feat, label, (spks, frame_segs)}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)#1 << 15 == 32768
        # Only keep key, feat, label
        # flen, mel 
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        new_sample = dict(key=sample['key'], label=sample['label'], feat=mat)
        # yield dict(key=sample['key'], label=sample['label'], feat=mat)
        # import ipdb; ipdb.set_trace()
        if 'spk_id' in sample:
            new_sample['spk_id'] = sample['spk_id']
        if 'spks' in sample:
            new_sample['spks'] = sample['spks']
        if 'segs' in sample:
            # new_sample['segs'] = sample['segs']
            # frame_segs to give more specific information
            frame_segs = (sample['segs']*sample['sample_rate']).int()
            tlen = sample['wav'].shape[-1]
            flen = mat.shape[0]
            frame_segs = (flen*(frame_segs/float(tlen))).round().int()
            # limits its maximum
            frame_segs = torch.minimum(frame_segs, torch.tensor(flen))
            # assert frame_segs.max() == flen
            new_sample['frame_segs'] = frame_segs
        yield new_sample


def re_seg(data, max_length=10000):
    """ repeat the overlapped fbank according to frame_segs
        Args:
            Iterable[{key, feat, label, (spks, frame_segs)}]
            max_length: in frame * 0.01s
        Returns:
            Iterable[{key, new_feat, label, (spks, frame_segs)}]
    """
    for sample in data:
        assert 'frame_segs' in sample
        assert 'feat' in sample
        flen = sample['feat'].shape[0]  # flen, mel
        frame_segs = sample['frame_segs']
        if frame_segs.max() != flen:
            # import ipdb; ipdb.set_trace()
            flen = min(flen, frame_segs.max())
        assert frame_segs.max() <= flen
        if len(frame_segs) == 1:
            yield sample
        else:
            # import ipdb; ipdb.set_trace()
            feat = sample['feat']
            new_feat = []
            # compute the total flen
            for seg in sample['frame_segs']:
                new_feat.append(feat[seg[0]:seg[1]])
            new_feat = torch.cat(new_feat, 0)

            # avoid padding too long frames
            if len(new_feat) > max_length: # 102second
                # print(new_feat.shape)
                print(sample['key'], 'reseg', len(new_feat))
                continue

            sample['feat'] = new_feat
            yield sample

def add_emb(data, spk_emb_dict, use_seg=True):
    """ repeat the overlapped fbank according to frame_segs
        Args:
            Iterable[{key, feat, label, (spks, frame_segs)}]
            use_seg: if is true, concatenate the seg
                    else, add the ivectors for the overlapped regions
        Returns:
            Iterable[{key, new_feat, label, (emb, frame_segs)}]
    """
    for sample in data:
        # import ipdb; ipdb.set_trace()
        assert 'spks' in sample
        flen = sample['feat'].shape[0]  # flen, mel
        elen = sum([fseg[1]-fseg[0] for fseg in sample['frame_segs']])
        emb_input = []                  # flen, emb_dim
        for spk, fseg in zip(sample['spks'], sample['frame_segs']):
            if spk in spk_emb_dict:
                emb = spk_emb_dict[spk] # emb_dim
                emb_input.append(emb.repeat(fseg[1]-fseg[0], 1))
            else:
                # import ipdb; ipdb.set_trace()
                print(f"Error {spk}, {fseg} not in emb dict")
                exit(0)
        if use_seg:
            emb_input = torch.cat(emb_input, 0)  # flen, emb_dim
        else:
            # if len(emb_input) > 1:
            #    import ipdb; ipdb.set_trace()
            # no re seg is used, add emb for overlap regions
            emb_overlap = torch.zeros(flen, emb.shape[0])
            for emb, fseg in zip(emb_input, sample['frame_segs']):
                emb_overlap[fseg[0]:fseg[1]] += emb
            emb_input = emb_overlap

        assert not torch.any(torch.isnan(emb_input)), ("emb has nan", emb_input)
        sample['emb'] = emb_input
        yield sample

def __tokenize_by_bpe_model(sp, txt):
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r'([\u4e00-\u9fff])')
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper())
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            for p in sp.encode_as_pieces(ch_or_w):
                tokens.append(p)

    return tokens


def tokenize(data, symbol_table, bpe_model=None, non_lang_syms=None,
             split_with_space=False):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            # data: Iterable[{key, wav, txt, sample_rate}]
            data: Iterable[{key, wav, txt, sample_rate, (spks, segs)}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate, (spks, segs)}]
    """
    if non_lang_syms is not None:
        non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
    else:
        non_lang_syms = {}
        non_lang_syms_pattern = None

    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    else:
        sp = None

    # import ipdb; ipdb.set_trace()
    for sample in data:
        assert 'txt' in sample
        txt = sample['txt'].strip()
        if non_lang_syms_pattern is not None:
            parts = non_lang_syms_pattern.split(txt.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [txt]

        label = []
        tokens = []
        for part in parts:
            if part in non_lang_syms:
                tokens.append(part)
            else:
                if bpe_model is not None:
                    tokens.extend(__tokenize_by_bpe_model(sp, part))
                else:
                    if split_with_space:
                        part = part.split(" ")
                    for ch in part:
                        if ch == ' ':
                            ch = "▁"
                        tokens.append(ch)

        # label store the indeces
        for ch in tokens:
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table['<unk>'])

        sample['tokens'] = tokens
        sample['label'] = label
        yield sample


def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            Iterable[{key, feat, label, (spks, frame_segs)}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label, (spks, frame_segs)}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label, (spks, frame_segs)}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label, (spks, frame_segs)}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label, (spks, frame_segs)}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label, (spks, frame_segs)}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':#all frames (after padding) less than  max_frames_in_batch
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data, use_emb=False):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label, (embs, frame_segs)}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths,
            (embs, frame_segs))]
    """
    for sample in data:#sample is a list, the num of elements is batch_size
        assert isinstance(sample, list)
        feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(feats_length, descending=True)
        feats_lengths = torch.tensor(
            [sample[i]['feat'].size(0) for i in order], dtype=torch.int32)
        sorted_feats = [sample[i]['feat'] for i in order]
        sorted_keys = [sample[i]['key'] for i in order]
        if 'spk_id' in sample[0]:
            sorted_spk_ids = [sample[i]['spk_id'] for i in order]
            sorted_spk_ids = torch.tensor(sorted_spk_ids, dtype=torch.int64)
            use_spk = True
        else:
            use_spk = False
        sorted_labels = [
            torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
        ]
        # store the org length of labels for loss computation 
        label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                     dtype=torch.int32)

        # B, T, featdim, pad on the Time axis
        padded_feats = pad_sequence(sorted_feats,
                                    batch_first=True,
                                    padding_value=0)
        padding_labels = pad_sequence(sorted_labels,
                                      batch_first=True,
                                      padding_value=-1)
        #assert not torch.any(torch.isnan(padded_feats)), ("padded_feats have nan", padded_feats )
        #assert not torch.any(torch.isnan(padding_labels)), ("padding_labels have nan", padding_labels )
        #import ipdb; ipdb.set_trace()
        #print(f"feats: {padded_feats.size()}\n")
        #print(f"labels: {padding_labels.size()}\n")
        if torch.any(torch.isnan(padded_feats)) or torch.any(torch.isnan(padding_labels)) or padded_feats.shape[1] != padding_labels.shape[1]:
            continue
        #import ipdb; ipdb.set_trace()
        if use_emb:
            sorted_embs = [sample[i]['emb'] for i in order]
            padded_embs = pad_sequence(sorted_embs, batch_first=True,
                                        padding_value=0)
            sorted_frame_segs = [sample[i]['frame_segs'].tolist() for i in order]
            assert not torch.any(torch.isnan(padded_embs)), ("padded_embs have nan", padded_embs )
            yield (sorted_keys, padded_feats, padding_labels, feats_lengths,
                   label_lengths, padded_embs, sorted_frame_segs)
        elif use_spk:
            yield (sorted_keys, sorted_spk_ids, padded_feats, padding_labels, feats_lengths,
                   label_lengths)
        else:
            yield (sorted_keys, padded_feats, padding_labels, feats_lengths,
                   label_lengths)
