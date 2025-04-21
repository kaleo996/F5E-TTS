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

import random
#import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
import yaml
import sys
import copy
#sys.path.append("/apdcephfs/private_nenali/lina/wenet/examples/pgg/s0")
#print(sys.path)
import wenet.dataset.processor as processor
from wenet.dataset.wav_augment import AugmentWAV
from wenet.utils.file_utils import read_lists
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols


class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()#not necessary here?
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        # TODO(Binbin Zhang): fix this
        # We can not handle uneven data for CV on DDP, so we don't
        # sample data by rank, that means every GPU gets the same
        # and all the CV data
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]#worker_id:0, num_workers:1
        return data


class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()#dict:{rank,world_size,worker_id,num_workers}
        indexes = self.sampler.sample(self.lists)#(1/world_size) data for each  rank
        for index in indexes:
            # yield dict(src=src)
            data = dict(src=self.lists[index])
            data.update(sampler_info)#dict:{src,rank,world_size,worker_id,num_workers}
            yield data

def load_emb_h5(emb_conf, speed_perturb=False):
    if not speed_perturb:
        h5_path = emb_conf['h5_path']
    else:
        # TODO: with speed permutation
        h5_path = emb_conf['perb_h5_path']
    spk_emb = {}
    # import ipdb; ipdb.set_trace()
    with h5py.File(h5_path, 'r') as h5_file:
        # fn_ids = h5_file['fn_ids'][:].astype(np.str_)
        spks = h5_file['spks'][:].astype(np.str_)
        embs = h5_file['embs'][:].astype(np.float32)
        # segs = h5_file['segs'][:].astype(np.float32)
        # there may be repeated spks 
        for spk, emb in zip(spks, embs):
            spk_emb[spk] = torch.from_numpy(emb)
    print(f"--Load {len(spk_emb)} speakers")
    return spk_emb

def Dataset(data_type,
            data_list_file,
            symbol_table,
            conf,
            bpe_model=None,
            non_lang_syms=None,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            conf(dict): config for dataset
            data_type(str): raw/shard
            bpe_model(str): model for english bpe part
            partition(bool): whether to do data partition in terms of rank
    """
    assert data_type in ['raw', 'shard']
    lists = read_lists(data_list_file)
    shuffle = conf.get('shuffle', True)
    dataset = DataList(lists, shuffle=shuffle, partition=partition)#iterable:dict(src=shard_name,rank=,world_size=,...)

    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)#add stream key (fid) to each dict 
        #{key, spk_id, wav, txt, sample_rate}, key is utt name
        dataset = Processor(dataset, processor.tar_file_and_group)
    else:
        dataset = Processor(dataset, processor.parse_raw)

    # convert token into ideces
    # Iterable[{key, spk_id, wav, txt, tokens(list), label(list), sample_rate}]
    dataset = Processor(dataset, processor.tokenize, symbol_table, bpe_model,
                        non_lang_syms, conf.get('split_with_space', True))
    # filter out the shorter and longer segments
    filter_conf = conf.get('filter_conf', {})
    dataset = Processor(dataset, processor.filter, **filter_conf)

    resample_conf = conf.get('resample_conf', {})
    dataset = Processor(dataset, processor.resample, **resample_conf)

    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = Processor(dataset, processor.speed_perturb)
    # do wav augmentation
    wav_aug = conf.get('wav_aug', False)
    wav_aug_conf = conf.get('wav_aug_conf', {})
    if wav_aug:
        augmentor = AugmentWAV(**wav_aug_conf)
        dataset = Processor(dataset, processor.wav_aug, augmentor)

    # Iterable[{key, spk_id, feat(flen,mel), label}]
    fbank_conf = conf.get('fbank_conf', {})
    dataset = Processor(dataset, processor.compute_fbank, **fbank_conf)

    # repeat overlapped frames to extend the feat at the input
    use_seg = conf.get('use_seg', False)
    if use_seg:
        #update feat
        dataset = Processor(dataset, processor.re_seg, filter_conf['max_length'])

    # involve speaker embedding
    use_emb = conf.get('use_emb', False)
    # valid when use_seg is true
    # use_emb = use_emb and use_seg
    if use_emb:
        # Iterable[{key, feat, label, (spks, embs, frame_segs)}]
        emb_conf = conf.get('emb_conf', {})
        # load embeddings 
        # import ipdb; ipdb.set_trace()
        spk_emb_dict = load_emb_h5(emb_conf, speed_perturb)
        dataset = Processor(dataset, processor.add_emb, spk_emb_dict,
                            use_seg=use_seg)

    spec_aug = conf.get('spec_aug', True)
    if spec_aug:
        #update feat
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = Processor(dataset, processor.spec_aug, **spec_aug_conf)


    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = Processor(dataset, processor.shuffle, **shuffle_conf)

    sort = conf.get('sort', True)#short at front
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = Processor(dataset, processor.sort, **sort_conf)

    batch_conf = conf.get('batch_conf', {})
    #Iterable[List[{key, feat, label}]], no. of dict in the list is batch_size
    dataset = Processor(dataset, processor.batch, **batch_conf)
    dataset = Processor(dataset, processor.padding, use_emb)

    print(f"Initial the dataset with use emb = {use_emb}")
    return dataset

if __name__ == '__main__':
    data_type='shard'
    train_data="data/train/tmp.list"
    cv_data="data/dev/tmp.list"
    symbol_table = read_symbol_table("data/dict/dict_phn.txt")
    #symbol_table = read_symbol_table("data/dict/lang_char.txt")
    use_seg="false"
    use_emb="false"
    train_config=f"conf/tmp.yaml"
    with open(train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf) # keep all most the same
    cv_conf['speed_perturb'] =False
    cv_conf['spec_aug'] = False
    cv_conf['wav_aug'] = False
    cv_conf['shuffle'] = False
    if train_conf['use_emb']:
        train_conf['emb_conf'] = train_conf['train_emb_conf']
        cv_conf['emb_conf'] = cv_conf['cv_emb_conf']
    bpe_model=None
    non_lang_syms = read_non_lang_symbols(None)
    train_dataset = Dataset(data_type, train_data, symbol_table,
                            train_conf, bpe_model, non_lang_syms, True)

    cv_dataset = Dataset(data_type,
                         cv_data,
                         symbol_table,
                         cv_conf,
                         bpe_model,
                         non_lang_syms,
                         partition=False)

    utt_num = 0
#    for item in cv_dataset: # train_dataset: # 
    for item in train_dataset: # 
        import ipdb; ipdb.set_trace()
        if len(item) == 7:
            #sorted_keys:list, feats:(B,T,feadim), labels:(B,L),
            #feats_lengths:(B,), label_lengths:(B,), embs:(B,T,embdim),
            #frame_segs:list
            sorted_keys, padded_feats, padding_labels, feats_lengths, label_lengths, padded_embs, frame_segs = item
        else:
            sorted_keys, sorted_spk_ids, padded_feats, padding_labels, feats_lengths, label_lengths = item
        print(sorted_keys)
        print(sorted_spk_ids)
        print(feats_lengths)
        print(label_lengths)
        # import ipdb; ipdb.set_trace()
        utt_num += len(sorted_keys)

    print(f"utt num is {utt_num}")

