from transformers import BertConfig
from transformers import XLMRobertaConfig, RobertaConfig
from torch.utils.data import DataLoader, TensorDataset
from .xlmr import XLMRModel
import torch
import os
import json
from collections import Counter
import pandas as pd
from glob import glob
import numpy as np
from collections import namedtuple

from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
T_co = TypeVar('T_co', covariant=True)

from torch.utils.data import RandomSampler, SequentialSampler, Sampler


class RandomSiameseSampler(Sampler):
    r"""Samples elements randomly.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self._num_samples = len(data_source)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self):
        n = self._num_samples // 2
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        yield from [x for y in [(i * 2, i * 2 + 1) for i in torch.randperm(n, generator=generator)] for x in y]

    def __len__(self):
        return self.num_samples


class DataProcessor:

    def get_examples(self, source_dir):
        data_files = glob(os.path.join(source_dir, '*.data'))
        Example = namedtuple('Example', ['docId', 'pos', 'text_1', 'text_2', 'label', 'start_1', 'end_1', 'start_2', 'end_2', 'lemma', 'target_id', 'sense_id'])
        examples = []
        for file in data_files:
            filename = '.'.join(file.split('/')[-1].split('.')[:-1])
            data = json.load(open(file, encoding='utf-8'))
            gold_file = f'{file[:-5]}.gold'
            gold_labels = json.load(open(gold_file, encoding='utf-8')) if os.path.exists(gold_file) else [{'tag': 'F'}] * len(data)
            for ex, lab in zip(data, gold_labels):
                pos = ex['pos'].lower()
                label = lab['tag']
                if 'target_id' in ex:
                    target_id = ex['target_id']
                else:
                    target_id = '?'
                if 'wordnet_sense' in ex:
                    sense_id = ex['wordnet_sense']
                else:
                    sense_id = '?'
                ex_id = ex['id'] if ex['id'].isnumeric() else ex['id'].split('.')[-1]
                examples.append(Example(f'{filename}.{ex_id}', pos, ex['sentence1'], ex['sentence2'], label, ex['start1'], ex['end1'], ex['start2'], ex['end2'], ex['lemma'], target_id, sense_id))
        return examples

def convert_docId(docId: str):
    lang2idx = {'en': 0, 'ru': 1, 'fr': 2, 'zh': 3, 'ar': 4}
    split, lg, num = docId.split('.')
    if split == 'dev':
        split = 1
    elif split == 'test':
        split = 2
    else: # 'train' or 'training'
        split = 0

    lg = lg.split('-')
    for i in (0, 1):
        lg[i] = lang2idx[lg[i]]
    lg = lg[0] * 10 + lg[1]

    num = int(num)
    return [split, lg, num]


def get_dataloader_and_tensors(
        features: list,
        batch_size: int,
        sampler_type: str = 'sequential',
):
    assert sampler_type in {'sequential', 'random', 'siamese_random'}
    input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )
    token_type_ids = torch.tensor(
        [f.token_type_ids for f in features],
        dtype=torch.long
    )
    syn_labels = torch.tensor(
        [f.syn_label for f in features]
    )
    positions = torch.tensor(
        [f.positions for f in features],
        dtype=torch.long
    )
    docIds = torch.tensor(
        [convert_docId(f.example.docId) for f in features],
        dtype=torch.int
    )
    eval_data = TensorDataset(
        input_ids, input_mask, token_type_ids,
        syn_labels, positions, docIds
    )
    if sampler_type == 'sequential':
        sampler = None
    else:
        sampler = RandomSiameseSampler(eval_data) if sampler_type == 'siamese_random' else RandomSampler(eval_data)

    dataloader = DataLoader(eval_data, batch_size=batch_size, sampler=sampler)

    return dataloader

# siamese_models = {
#     "roberta-base": SiameseXLMRModel,
#     "roberta-large": SiameseXLMRModel,
#     "xlm-roberta-base": SiameseXLMRModel,
#     "xlm-roberta-large": SiameseXLMRModel
# }

models = {
    "roberta-base": XLMRModel,
    "roberta-large": XLMRModel,
    "xlm-roberta-base": XLMRModel,
    "xlm-roberta-large": XLMRModel
}

configs = {
    "roberta-large": RobertaConfig,
    "roberta-base": RobertaConfig,
    "xlm-roberta-base": XLMRobertaConfig,
    "xlm-roberta-large": XLMRobertaConfig
}
