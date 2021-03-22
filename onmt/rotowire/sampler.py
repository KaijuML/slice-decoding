from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from typing import Union

import torch


def build_dataset_iter(dataset, opt, device_id=-1, train=True):
    """
    Builds an iterable from dataset, with each batch on the correct device.

    :param dataset: a rotowire.RotoWire object
    :param opt: the training / inference options
    :param device_id: the id of the device (-1 for cpu)
    :param train: When train, an inifinite number of batches are returned, in random order.
    :return:
    """
    sampler = InfiniteRandomSampler(dataset) if train else None
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
        self.device = torch.device(device_id if device_id >= 0 else 'cpu')

    @classmethod
    def obj_to_device(cls, obj, device):
        if isinstance(obj, tuple):
            return tuple(item.to(device) for item in obj)
        return obj.to(device)

    @classmethod
    def batch_to_device(cls, batch, device):
        """Move `batch` to `device`"""
        curr_device = batch.indices.device
        if curr_device != device:
            batch.src = cls.obj_to_device(batch.src, device)
            batch.sentences = cls.obj_to_device(batch.sentences, device)
            batch.alignments = cls.obj_to_device(batch.alignments, device)
            batch.elaborations = cls.obj_to_device(batch.elaborations, device)
            batch.contexts = cls.obj_to_device(batch.contexts, device)
            batch.src_map = cls.obj_to_device(batch.src_map, device)
            batch.indices = cls.obj_to_device(batch.indices, device)

    def __iter__(self):
        for batch in self.iterable:
            self.batch_to_device(batch, self.device)
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


class Batch:
    """
    Batch object, to comply wiht onmt's API as much as possible.
    """

    def __init__(self, fields):
        self.src = fields.pop('src')
        self.sentences = fields.pop('sentences')

        self.elaborations = fields.pop('elaborations')
        self.contexts = fields.pop('contexts')

        self.src_map = fields.pop('src_map')
        self.src_ex_vocab = fields.pop('src_ex_vocab')
        self.alignments = fields.pop('alignments')

        self.indices = fields.pop('indices')

        assert len(fields) == 0, list(fields)

    @property
    def batch_size(self):
        return self.src[0].size(1)


def classic_pad(minibatch, return_lengths=True):
    """This is largely borrowed for torchtext v0.8"""
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
    """
    n_sentences = max([t.size(0) for t in data])
    n_entities = max([t.size(1) for t in data])
    contexts = - torch.ones(n_sentences, len(data), n_entities)
    for batch_idx, context in enumerate(data):
        a, b = context.size(0), context.size(1)
        contexts[:a, batch_idx, :b] = context
    return contexts


def collate_fn(examples):
    """
    Here are the operations need to collate a batch:
        - src should be padded
        - src_lengths should be computed
        - each sentence of sentences should be padded
        - there should be an equal number of sentences in sentences
        
    Everything should be put inside a batch object, 
    which is able to change device with a `.to(device)` method.
    """

    # Basic collate, to be improved for specific fields
    batch = {key: list() for key in list(examples[0])}
    for example in examples:
        for key, value in example.items():
            batch[key].append(value)

    batch['src'] = classic_pad(batch['src'], return_lengths=True)
    batch['sentences'] = nested_pad(batch['sentences'], include_idxs=True)

    batch['elaborations'] = classic_pad(batch['elaborations'],
                                        return_lengths=False)

    batch['contexts'] = make_contexts(batch['contexts'])

    batch['alignments'] = nested_pad(batch['alignments'],
                                     include_idxs=False)

    batch['src_map'] = make_src_map(batch['src_map'])

    batch['indices'] = torch.LongTensor(batch['indices'])

    return Batch(batch)
