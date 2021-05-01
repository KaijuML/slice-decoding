from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from onmt.utils.misc import format_device

from onmt.rotowire.dataset import RotowireDataset
from onmt.rotowire import (
    RotowireTrainingDataset,
    RotowireGuidedInferenceDataset
)

from typing import Union

import torch


def build_dataset_iter(dataset, opt, device_id=-1, debug_batches=None):
    """
    Builds an iterable from dataset, with each batch on the correct device.

    :param dataset: a rotowire.RotowireDataset object
    :param opt: the training / inference options
    :param device_id: the id of the device (-1 for cpu)
    """
    if not isinstance(dataset, RotowireDataset):
        raise TypeError(f'Unexpected dataset type: {type(dataset)}')

    sampler = None
    if debug_batches is not None:
        sampler = DebugSampler(dataset, debug_batches)
        opt.batch_size = len(debug_batches[0])
    elif isinstance(dataset, RotowireTrainingDataset):
        sampler = InfiniteRandomSampler(dataset)

    loader = DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler,
                        num_workers=opt.num_threads, collate_fn=collate_fn,
                        pin_memory=True, drop_last=False)

    return IterOnDevice(loader, device_id)


class IterOnDevice:
    """
    Send items from `iterable` on `device_id` and yield.
    
    Adapted from onmt to work with our custom RotoWire.
    """

    def __init__(self, iterable, device_id):
        self.iterable = iterable
        self.device = format_device(device_id)

        self.do_context, self.do_target = False, False
        if isinstance(iterable.dataset, RotowireTrainingDataset):
            self.do_context, self.do_target = True, True
        if isinstance(iterable.dataset, RotowireGuidedInferenceDataset):
            self.do_context = True

    def __len__(self):
        return len(self.iterable)

    def obj_to_device(self, obj):
        if isinstance(obj, tuple):
            return tuple(item.to(self.device) for item in obj)
        return obj.to(self.device)

    def batch_to_device(self, batch):
        """Move `batch` to `device`"""
        curr_device = batch.indices.device
        if curr_device != self.device:

            # Batch always has source tensors
            batch.n_primaries = self.obj_to_device(batch.n_primaries)
            batch.src_map = self.obj_to_device(batch.src_map)
            batch.indices = self.obj_to_device(batch.indices)
            batch.src = self.obj_to_device(batch.src)

            # During training and guided inference Batch has additional tensors
            if self.do_context:
                batch.elaborations = self.obj_to_device(batch.elaborations)
                batch.contexts = self.obj_to_device(batch.contexts)

            if self.do_target:
                batch.alignments = self.obj_to_device(batch.alignments)
                batch.sentences = self.obj_to_device(batch.sentences)

    def __iter__(self):
        for batch in self.iterable:
            self.batch_to_device(batch)
            yield batch


class InfiniteRandomSampler(Sampler):
    """
    Samples elements randomly, without stopping. 
    When all elements have been returned, start over.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.num_samples = len(data_source)

    def __iter__(self):
        """
        Cycles over random permutations of the data indefinitely
        """
        while True:
            yield from torch.randperm(self.num_samples).tolist()

    def __len__(self):
        return self.num_samples


class DebugSampler(Sampler):
    """
    Samples only elements contained in debug_batches.
    For debug purposes, obviously.

    Args:
        data_source (Dataset): dataset to sample from
        debug_batches (list of list): lists of batches (list of indices)
    """

    def __init__(self, data_source, debug_batches):
        self.data_source = data_source
        self.num_samples = len(debug_batches)
        self.batches = [idx for batch in debug_batches for idx in batch]

    def __iter__(self):
        """
        Yields specific batches and stops.
        """
        yield from self.batches

    def __len__(self):
        return self.num_samples


class Batch:
    """
    Batch object, to comply wiht onmt's API as much as possible.
    """

    _batch_dim_per_item = {
        'src': [1, 0],
        'sentences': [1, 1, 1],
        'n_primaries': 0,
        'elaborations': 1,
        'contexts': 2,
        'alignments': 1,
        'src_map': 1,
        'indices': 0,
        'src_ex_vocab': None,
        'elaborations_query_mapping': None,
        'elaboration_view_idxs': None,
    }

    def __init__(self, fields):

        # A Batch always has source tensors
        self.n_primaries = fields.pop('n_primaries')
        self.src_ex_vocab = fields.pop('src_ex_vocab')
        self.src_map = fields.pop('src_map')
        self.src = fields.pop('src')

        self.indices = fields.pop('indices')

        self.elaboration_view_idxs = fields.pop('elaboration_view_idxs')

        if 'contexts' in fields:
            self.elaborations = fields.pop('elaborations')
            self.contexts = fields.pop('contexts')

            if 'sentences' in fields:
                self.alignments = fields.pop('alignments')
                self.sentences = fields.pop('sentences')
        else:
            self.elaborations_query_mapping = fields.pop('elaborations_query_mapping')

        assert len(fields) == 0, list(fields)

    @property
    def batch_size(self):
        return self.indices.size(0)

    @property
    def device(self):
        return self.indices.device

    def index_select(self, indices):
        """
        Only keep indices of interest in the batch. Used during inference, when
        an example is done, and we want to remove its element from the batch,
        to continue generation for other examples of the batch.

        WARNING: if indices are not sorted, this will reorder the batch.

        :param indices: indices to keep
        :return:
        """
        for name, dim in self._batch_dim_per_item.items():
            if (item := getattr(self, name, None)) is None:
                continue

            if isinstance(item, tuple):
                setattr(self, name, tuple([
                    _item.index_select(_dim, indices)
                    for _item, _dim in zip(item, dim)
                ]))
            elif isinstance(item, list):
                setattr(self, name, [
                    _item for idx, _item in enumerate(item)
                    if idx in indices
                ])
            else:
                setattr(self, name, item.index_select(dim, indices))

    def __repr__(self):
        s = f'Batch[{self.batch_size}]\n'
        for name in self._batch_dim_per_item:
            if (item := getattr(self, name, None)) is None:
                continue

            if isinstance(item, (tuple, list)):
                if isinstance(item[0], torch.Tensor):
                    for i, obj in enumerate(item):
                        _cls = obj.dtype
                        s += f'  [{name} {i}] ({_cls}): {list(obj.shape)}\n'
                else:
                    _cls = type(item).__name__
                    s += f'  {name} ({_cls}): [{len(item)}]\n'
            elif isinstance(item, torch.Tensor):
                _cls = item.dtype
                s += f'  [{name}] ({_cls}): {list(item.shape)}\n'
            s += '\n'
        return s.strip()


def classic_pad(minibatch, return_lengths=True):
    """This is largely borrowed from torchtext v0.8"""
    lengths = [x.size(0) for x in minibatch]
    padded = pad_sequence(minibatch, padding_value=1)
    if return_lengths:
        return padded, torch.LongTensor(lengths)
    return padded


def nested_pad(minibatch: Union[list], include_idxs: bool = True):
    """
    This functions aims to build a target object from a minibatch of lists of
    sentences.

    Args:
        minibatch (list): List of lists of sentences.
        include_idxs (bool): Return a list of idx-to-sentence for each example

    Returns:
        padded (torch.LongTensor): [tgt_len, bsz] padded target sentences
        indices (torch.LongTensor): [tgt_len, bsz] index of sentence for each token

    """
    padded_sentences, sentence_starts, sentence_indices = list(), list(), list()
    for sentences in minibatch:

        document, starts, indices = list(), list(), list()

        for idx, sentence in enumerate(sentences, 1):

            if include_idxs:

                # Optionally remembers the index at which the sentence starts
                starts.append(len(document))

                # Optionally add a mapping to this index for each token of the sent
                indices.extend([idx] * len(sentence))

            document.extend(sentence)

        padded_sentences.append(torch.LongTensor(document))

        if include_idxs:
            sentence_starts.append(torch.LongTensor(starts + [len(document)]))
            sentence_indices.append(torch.LongTensor(indices))

    padded_sentences = pad_sequence(padded_sentences)

    if include_idxs:
        sentence_starts = pad_sequence(sentence_starts)
        sentence_indices = pad_sequence(sentence_indices)
        return padded_sentences, sentence_starts, sentence_indices
    return padded_sentences


def make_src_map(data):
    """
    Taken from onmt.
    """
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(src_size, len(data), src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[j, i, t] = 1
    return alignment


def make_contexts(data):
    """
    Using the same trick as make_src_map
    Note that we are padding with -1 because 0 is a valid value.
    """
    n_sentences = max([t.size(0) for t in data])
    n_entities = max([t.size(1) for t in data])
    contexts = -torch.ones(n_sentences, n_entities, len(data), dtype=torch.long)
    for batch_idx, context in enumerate(data):
        a, b = context.size(0), context.size(1)
        contexts[:a, :b, batch_idx] = context
    return contexts


def collate_fn(examples):
    """
    Here are the operations need to collate a batch:
        - src should be padded
        - src_lengths should be computed
        - each sentence of sentences should be padded
        - there should be an equal number of sentences in sentences
        
    Everything should be put inside a batch object, which has several utilities
    for Training and Inference purposes.
    """

    # Basic collate, to be improved for specific fields
    batch = {key: list() for key in list(examples[0])}
    for example in examples:
        for key, value in example.items():
            batch[key].append(value)

    # A batch always has source tensors & indices
    batch['n_primaries'] = torch.LongTensor(batch['n_primaries'])
    batch['indices'] = torch.LongTensor(batch['indices'])
    batch['src_map'] = make_src_map(batch['src_map'])
    batch['src'] = classic_pad(batch['src'], return_lengths=True)

    # We are in Training or GuidedInference mode
    if (contexts := batch.get('contexts', None)) is not None:
        if (elaborations := batch.get('elaborations', None)) is None:
            raise RuntimeError('Both Contexts and Elaborations are needed for '
                               'Training OR GuidedInference. (Elaborations are '
                               'missing.)')

        batch['contexts'] = make_contexts(contexts)
        batch['elaborations'] = classic_pad(elaborations, return_lengths=False)

        # We are definitely in Training mode
        if (sentences := batch.get('sentences', None)) is not None:
            if (alignments := batch.get('alignments', None)) is None:
                raise RuntimeError('Both Sentences and Alignments are needed '
                                   'for Training. (Alignments are missing.)')
            batch['sentences'] = nested_pad(sentences, include_idxs=True)
            batch['alignments'] = nested_pad(alignments, include_idxs=False)

    # We are in Inference mode
    else:
        for key in {'elaboration_view_idxs', 'elaborations_query_mapping'}:
            if batch.get(key, None) is None:
                raise RuntimeError(f'{key} is needed for Inference')

    return Batch(batch)
