"""
Here I create a custom RotoWire dataset object.
TODO: multiprocessing + logging
"""
from onmt.rotowire.config import RotowireConfig
from onmt.rotowire.utils import FileIterable
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from collections import Counter

import more_itertools
import functools
import torch
import tqdm
import json
import os


@functools.singledispatch
def numericalize(sentence, vocab: Vocab, add_special_tokens: bool=False):
    """
    Base function for numericalize:
        - replace all tokens by their id in vocab
        - maybe add <s> and </s> tokens
        
    Note: using singledispatch is for fun and not really required here!
    """
    raise TypeError(f'This function will not run for {type(sentence)}')

@numericalize.register
def _(sentence: str, vocab: Vocab, add_special_tokens: bool=False):
    sentence = [vocab[tok] for tok in sentence.split()]
    if add_special_tokens:
        sentence = [vocab['<s>']] + sentence + [vocab['</s>']]
    return sentence

@numericalize.register
def _(sentence: list, vocab: Vocab, add_special_tokens: bool=False):
    sentence = [vocab[tok] for tok in sentence]
    if add_special_tokens:
        sentence = [vocab['<s>']] + sentence + [vocab['</s>']]
    return sentence


class DataAlreadyExistsError(Exception):
    pass


class RotowireParser:
    """
    Custom parser for custom RotoWire data.

    One example <jsonline> should be a dictionary with inputs/outputs keys
    (all other keys will be ignored)
        inputs: list of entities, as key-value pairs
        ouputs: list of sentences, with their grounding entities
                (as indexed in inputs)

    If you want to know more, self.parse_example has detailed comments.
    """

    def __init__(self, config):
        if not isinstance(config, RotowireConfig):
            raise TypeError(f'Unknown config object: {type(config)}')

        self.config = config

    def parse_example(self, jsonline):

        inputs, outputs = jsonline['inputs'], jsonline['outputs']

        # Quickly convert inputs as an ordered list instead of a dict
        if isinstance(inputs, dict):
            inputs = [inputs[str(idx)] for idx in range(len(inputs))]

        # What this function will return
        example = dict()

        # This is a counter restricted to this example
        # It'll be used to build a vocab specifically for the copy mechanism.
        source_vocab = Counter()

        # This is a counter on column names
        cols_vocab = Counter()

        # Lists from input_sequence will be flattend and used model inputs.
        # First list is cell values, second list is cell column names.
        input_sequence = [list(), list()]

        for slice_dict in inputs:

            src_text = ['<ent>']
            src_cols = ['<ent>']

            for key, value in slice_dict['data'].items():
                if value == 'N/A' and not self.config.keep_na:
                    continue

                src_text.append(value.replace(' ', '_'))
                src_cols.append(key)

                source_vocab.update(src_text)
                cols_vocab.update(src_cols)

            input_sequence[0].append(self.pad_entity(src_text))
            input_sequence[1].append(self.pad_entity(src_cols))

        example['src'] = [' '.join(more_itertools.collapse(seq))
                          for seq in input_sequence]
        example['source_vocab'] = Vocab(
            source_vocab, specials=['<unk>', '<pad>', '<ent>'])

        # We also build a src_map. This mapping assigns to each source token
        # its position in the source_vocab. This is used by the copy mechanism
        # to gather probas over source_vocab using attention over src.
        # At this stage, the map is flat, but DataLoader will one-hot everything.
        src_map = torch.LongTensor([example['source_vocab'][tok]
                                    for tok in example['src'][0].split()])
        example['src_map'] = src_map

        # This is a 'global' counter, used to build a vocab for all examples.
        # As source and target share the same vocab, it is initialized with
        # the source_vocab (which is still a counter at this point)
        main_vocab = Counter() + source_vocab

        # This is a list containing all sentences from the summary
        sentences = list()

        # This is a list containing which elements of sentences are copied
        # In practice, for each token, it gives its id in source_vocab.
        # Note that if it's not found, the id is 0, for <unk>
        alignments = list()

        # This is list containing the type of elaboration required to
        # write the associated sentence. Currently supports:
        #  - None
        #  - End of Document.
        elaborations = list()

        # Tracks which entities are relevant for a given sentence
        contexts = list()

        for sentence in outputs:

            # TODO: bring support for other type of elaborations
            elaborations.append('<none>')
            contexts.append(sentence['slices'])

            # See self._clean_sentence.__doc__
            sentence_str = self._clean_sentence(
                sentence['text'], example['source_vocab'].itos)

            main_vocab.update(sentence_str.split())
            sentences.append(sentence_str)

            # To compute alignment for this sentence, we iterate over all tokens
            # and check whether they can be found in grounding slices.
            alignment = list()
            for token in sentence_str.split():
                _algn = 0
                for slice_idx in sentence['slices']:
                    if token in input_sequence[0][slice_idx]:
                        _algn = example['source_vocab'].stoi[token]
                        assert _algn != 0
                        break
                alignment.append(_algn)

            # Note that we add 0 at start and end of sentence, because
            # <s> and </s> token are never copied.
            alignments.append([0] + alignment + [0])

        example['sentences'] = sentences
        example['alignments'] = alignments
        example['elaborations'] = elaborations + ['<eod>']
        example['contexts'] = contexts

        return example, main_vocab, cols_vocab

    @staticmethod
    def _clean_sentence(sentence, vocab):
        """
        In here, we slightly help the copy mechanism.
        When we built the source sequence, we took all multi-words value
        and repalaced spaces by underscores. We replace those as well in
        the summaries, so that the copy mechanism knows it was a copy.
        It only happens with city names like "Los Angeles".
        """
        for token in vocab:
            if '_' in token:
                token_no_underscore = token.replace('_', ' ')
                sentence = sentence.replace(token_no_underscore, token)
        return sentence

    def pad_entity(self, entity):
        """
        For parallelisation purposes, we will split the input tensor into entities.
        All entities should therefore have the same size, so that it fits into
        a pytorch.LongTensor.
        """
        if (pad_size := self.config.entity_size - len(entity)) > 0:
            entity.extend(['<pad>'] * pad_size)

        # sanity check
        if not len(entity) == self.config.entity_size:
            msg = f"""
                The entity size {self.config.entity_size} given in config 
                appears to be too small: an entity of size {len(entity)} 
                was encountered during preprocessing.
            """
            raise RuntimeError(msg.replace('\n', '').replace('    ', ''))

        return entity


class RotoWireDataset(Dataset):
    """
    Custom RotoWire dataset.
    """
    def __init__(self, examples, main_vocab, cols_vocab, config=None):
        
        self.main_vocab = main_vocab
        self.cols_vocab = cols_vocab
        
        self.examples = examples

        if config is not None:
            self.config = config
        else:
            self.config = RotowireConfig.from_defaults()

        self.elab_vocab = self.config.elaboration_vocab
        
    def get_vocabs(self):
        return {
            'main_vocab': self.main_vocab,
            'cols_vocab': self.cols_vocab,
            'elab_vocab': self.elab_vocab
        }
        
    def __getitem__(self, item):
        # Get raw example
        raw_example = self.examples[item]
        
        # Start building the actual tensor example
        example = dict()
        
        # Numericalize the source, and add column names
        example['src'] = [numericalize(seq, voc) 
                          for seq, voc in zip(raw_example['src'],
                                            [self.main_vocab, self.cols_vocab])]
        example['src'] = torch.LongTensor(example['src']).transpose(0, 1)
        
        # Numericalize the target sentences
        example['sentences'] = [numericalize(sentence, self.main_vocab, True)
                                for sentence in raw_example['sentences']]
        example['sentences'] = [torch.LongTensor(sentence)
                                for sentence in example['sentences']]
        
        # Numericalize elaborations
        elaborations = numericalize(raw_example['elaborations'], self.elab_vocab)
        example['elaborations'] = torch.LongTensor(elaborations)

        # Create and pad the contexts
        example['contexts'] = pad_sequence([
            torch.LongTensor(c) for c in raw_example['contexts']
        ], padding_value=-1)
        
        # Adding all stuff that doesn't require processing
        example['src_map'] = raw_example['src_map']
        example['src_ex_vocab'] = raw_example['source_vocab']
        example['alignments'] = [torch.LongTensor(alignment)
                                 for alignment in raw_example['alignments']]
        example['indices'] = torch.LongTensor([item])
        
        return example
    
    def __len__(self):
        return len(self.examples)
    
    def __iter__(self):
        yield from self.examples
        
    @staticmethod
    def check_paths(relative_prefix, overwrite=False):
        save_path = os.path.abspath(relative_prefix)
        dirs = os.path.dirname(save_path)
        prefix = os.path.basename(save_path)
        
        path_to_data = os.path.join(dirs, f'{prefix}.examples.pt')
        path_to_vocabs = os.path.join(dirs, f'{prefix}.vocabs.pt')
        path_to_config = os.path.join(dirs, f'{prefix}.config.pt')
        
        if os.path.exists(path_to_data) and not overwrite:
            raise DataAlreadyExistsError(f'{path_to_data}')
        if os.path.exists(path_to_vocabs) and not overwrite:
            raise DataAlreadyExistsError(f'{path_to_vocabs}')
        if os.path.exists(path_to_config) and not overwrite:
            raise DataAlreadyExistsError(f'{path_to_config}')
                
        return {
            'examples': path_to_data,
            'vocabs': path_to_vocabs,
            'config': path_to_config,
        }
        
    def dump(self, prefix, overwrite=False):
    
        paths = self.check_paths(prefix, overwrite)
        
        # saving examples
        torch.save(self.examples, paths['examples'])
        
        # saving vocabs (not including elaboration vocab, which is always fixed)
        vocabs = {'main_vocab': self.main_vocab, 'cols_vocab': self.cols_vocab}
        torch.save(vocabs, paths['vocabs'])

        # saving config
        torch.save(self.config, paths['config'])
        
    @classmethod
    def load(cls, prefix):
        examples = torch.load(f'{prefix}.examples.pt')
        vocabs = torch.load(f'{prefix}.vocabs.pt')
        config = torch.load(f'{prefix}.config.pt')
        return cls(examples, **vocabs, config=config)
    
    @classmethod
    def from_raw_json(cls, filename, config=None):
        """
        TODO: Use multiprocessing to improve performances.
        """
        if config is None:
            config = RotowireConfig.from_defaults()

        examples = list()
        main_vocab = Counter()
        cols_vocab = Counter()

        parser = RotowireParser(config=config)

        iterable = FileIterable.from_filename(filename, fmt='jsonl')
        desc = "Reading and formatting raw data"

        for jsonline in tqdm.tqdm(iterable, desc=desc, total=len(iterable)):
            ex, sub_main_vocab, sub_cols_vocab = parser.parse_example(jsonline)
            
            examples.append(ex)
            main_vocab += sub_main_vocab
            cols_vocab += sub_cols_vocab
            
        main_vocab = Vocab(main_vocab, max_size=config.vocab_size,
                           specials=['<unk>', '<pad>', '<s>', '</s>', '<ent>'])
        
        cols_vocab = Vocab(cols_vocab, specials=['<unk>', '<pad>', '<ent>'])
            
        return cls(examples, main_vocab, cols_vocab, config=config)
