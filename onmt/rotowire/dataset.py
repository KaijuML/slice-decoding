"""
Here I create a custom RotoWire dataset object.
TODO: multiprocessing + logging
"""
from onmt.rotowire.parser import RotowireTrainingParser, RotowireInferenceParser
from onmt.rotowire.config import RotowireConfig
from onmt.rotowire.utils import FileIterable
from torch.nn.utils.rnn import pad_sequence
from onmt.utils.logging import logger
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from collections import Counter

from onmt.rotowire.exceptions import (
    DataAlreadyExistsError,
)

import functools
import torch
import tqdm
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


class RotowireDataset(Dataset):
    """
    Base class for custom RotoWire datasets.
    Children of this class should implement __getitem__ and build_from_json
    """
    def __init__(self, examples, main_vocab, cols_vocab, config=None):
        
        self.main_vocab = main_vocab
        self.cols_vocab = cols_vocab
        
        self.examples = examples

        if config is not None:
            self.config = config
        else:
            logger.info('Loading default config.')
            self.config = RotowireConfig.from_defaults()

        self.elab_vocab = self.config.elaboration_vocab

        logger.info('Dataset loaded with the following config:')
        logger.info(self.config)
        logger.info(f'    Number of examples: {len(self)}')
        logger.info(f'    Size of vocabulary: {len(self.main_vocab)}')
        logger.info(f'    Number of known columns: {len(self.cols_vocab)}')
        logger.info(f'    Number of known elaborations: {len(self.elab_vocab)}')

        latest_commit = getattr(self.config, "commit", None)
        latest_commit = "Unknown" if latest_commit is None else latest_commit
        logger.info(f'    Lastest commit: {latest_commit}')

    @property
    def vocabs(self):
        return {
            'main_vocab': self.main_vocab,
            'cols_vocab': self.cols_vocab,
            'elab_vocab': self.elab_vocab
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __iter__(self):
        yield from self.examples

    @staticmethod
    def check_vocabs(vocabs):
        if isinstance(vocabs, (tuple, list)):
            vocabs = {'main_vocab': vocabs[0], 'cols_vocab': vocabs[1]}
        if not isinstance(vocabs, dict):
            raise TypeError('vocabs should be Dict[Vocab]. '
                            f'Instead, vocabs are {type(vocabs).__name__}')

        main_vocab = vocabs.get('main_vocab', None)
        if main_vocab is None:
            raise ValueError('vocabs are missing main_vocab')
        elif not isinstance(main_vocab, Vocab):
            raise ValueError('main_vocab should be Vocab. '
                             f'Instead, main_vocab is {type(main_vocab.__name__)}')

        cols_vocab = vocabs.get('cols_vocab', None)
        if cols_vocab is None:
            raise ValueError('vocabs are missing main_vocab')
        elif not isinstance(cols_vocab, Vocab):
            raise ValueError('main_vocab should be Vocab. '
                             f'Instead, main_vocab is {type(cols_vocab.__name__)}')

        return {'main_vocab': main_vocab, 'cols_vocab': cols_vocab}
        
    @staticmethod
    def check_paths(relative_prefix, overwrite=False, mkdirs=False):
        save_path = os.path.abspath(relative_prefix)
        dirs = os.path.dirname(save_path)
        prefix = os.path.basename(save_path)
        
        path_to_data = os.path.join(dirs, f'{prefix}.examples.pt')
        path_to_vocabs = os.path.join(dirs, f'{prefix}.vocabs.pt')
        path_to_config = os.path.join(dirs, f'{prefix}.config.pt')
        
        if os.path.exists(path_to_data) and not overwrite:
            raise DataAlreadyExistsError(path_to_data)
        if os.path.exists(path_to_vocabs) and not overwrite:
            raise DataAlreadyExistsError(path_to_vocabs)
        if os.path.exists(path_to_config) and not overwrite:
            raise DataAlreadyExistsError(path_to_config)

        if not os.path.exists(dirs) and mkdirs:
            os.makedirs(dirs)
                
        return {
            'examples': path_to_data,
            'vocabs': path_to_vocabs,
            'config': path_to_config,
        }
        
    def dump(self, prefix, overwrite=False):
    
        paths = self.check_paths(prefix, overwrite, mkdirs=True)
        
        # saving examples
        logger.info(f"Saving examples to {paths['examples']}")
        torch.save(self.examples, paths['examples'])
        
        # saving vocabs (not including elaboration vocab, which is always fixed)
        logger.info(f"Saving vocabularies to {paths['vocabs']}")
        vocabs = {'main_vocab': self.main_vocab, 'cols_vocab': self.cols_vocab}
        torch.save(vocabs, paths['vocabs'])

        # saving config
        logger.info(f"Saving config to {paths['config']}")
        torch.save(self.config, paths['config'])

        logger.info('All saved.')
        
    @classmethod
    def load(cls, prefix):
        examples = torch.load(f'{prefix}.examples.pt')
        vocabs = torch.load(f'{prefix}.vocabs.pt')
        config = torch.load(f'{prefix}.config.pt')
        return cls(examples, **vocabs, config=config)

    @classmethod
    def build_from_raw_json(cls, *args, **kwargs):
        raise NotImplementedError()

    def _build_example_source(self, raw_example, item):

        # Start building the actual tensor example
        example = dict()

        # Numericalize the source, and add column names
        example['src'] = [numericalize(seq, voc)
                          for seq, voc in zip(raw_example['src'],
                                              [self.main_vocab, self.cols_vocab])]
        example['src'] = torch.LongTensor(example['src']).transpose(0, 1)

        # Adding all stuff that doesn't require processing
        example['src_map'] = raw_example['src_map']
        example['n_primaries'] = torch.LongTensor([raw_example['n_primaries']])
        example['src_ex_vocab'] = raw_example['source_vocab']
        example['indices'] = torch.LongTensor([item])

        return example

    def __getitem__(self, item):
        raise NotImplementedError()


class RotowireTrainingDataset(RotowireDataset):

    def __getitem__(self, item):
        # Get raw example
        raw_example = self.examples[item]

        example = self._build_example_source(raw_example, item)

        # Numericalize the target sentences
        example['sentences'] = [numericalize(sentence, self.main_vocab, True)
                                for sentence in raw_example['sentences']]
        example['sentences'] = [torch.LongTensor(sentence)
                                for sentence in example['sentences']]

        # Adding alignment information to train the copy mechanism
        example['alignments'] = [torch.LongTensor(alignment)
                                 for alignment in raw_example['alignments']]

        # Numericalize elaborations
        elaborations = numericalize(raw_example['elaborations'], self.elab_vocab)
        example['elaborations'] = torch.LongTensor(elaborations)

        # Create and pad the contexts
        example['contexts'] = pad_sequence([
            torch.LongTensor(c) for c in raw_example['contexts']
        ], batch_first=True, padding_value=-1)

        return example
    
    @classmethod
    def build_from_raw_json(cls, filename, config=None,
                            dest=None, overwrite=False,
                            raise_on_error=True):
        """
        Build a training RotowireDataset from the jsonl <filename>.
        ARGS:
            filename (str): Where to find raw jsonl
            config (RotowireConfig): see onmt.rotowire.config.py for info
            dest (filename): if provided, will save the dataset to <dest>
            overwrite (Bool): whether to replace existing data
            raise_on_error (Bool): whether to raise an error when a problematic
                                   line is encountered

        Note that some checks/warnings are performed early to save time
        when something goes wrong.

        TODO: Use multiprocessing to improve performances.
        """
        if config is None:
            logger.info('No config file was given, using defaults.')
            config = RotowireConfig.from_defaults()

        if dest is not None:
            cls.check_paths(dest, overwrite=overwrite)

        logger.info(f'Prepocessing Rotowire file, found at {filename}')
        logger.info(config)

        examples = list()
        main_vocab = Counter()
        cols_vocab = Counter()

        parser = RotowireTrainingParser(config=config)

        iterable = FileIterable.from_filename(filename, fmt='jsonl')
        desc = "Reading and formatting raw data"

        for idx, jsonline in tqdm.tqdm(enumerate(iterable),
                                       desc=desc, total=len(iterable)):
            ex, sub_main_vocab, sub_cols_vocab = parser.parse_example(idx, jsonline)

            if ex is not None:
                examples.append(ex)
                main_vocab += sub_main_vocab
                cols_vocab += sub_cols_vocab
            elif raise_on_error:
                break

        # If any error was found, log them with a warning.
        # If on_error == 'raise', first found error would result in break
        parser.log_error_and_maybe_raise(do_raise=raise_on_error)

        main_specials = ['<unk>', '<pad>', '<s>', '</s>', '<ent>']
        cols_specials = ['<unk>', '<pad>', '<ent>']
        vocabs = {
            'main_vocab': Vocab(main_vocab, max_size=config.vocab_size,
                                specials=main_specials),
            'cols_vocab': Vocab(cols_vocab, specials=cols_specials)
        }
        
        dataset = cls(examples, **vocabs, config=config)

        if dest is not None:
            dataset.dump(dest, overwrite=overwrite)

        return dataset


class RotowireInferenceDataset(RotowireDataset):

    def __getitem__(self, item):
        # Only building the source during Inference
        return self._build_example_source(self.examples[item], item)

    @classmethod
    def build_from_raw_json(cls, filename, config, vocabs, raise_on_error=True):

        parser = RotowireInferenceParser(config=config, guided_inference=False)
        return cls._build_from_raw_json(filename=filename,
                                        config=config,
                                        vocabs=vocabs,
                                        parser=parser,
                                        raise_on_error=raise_on_error)

    @classmethod
    def _build_from_raw_json(cls, filename, config, vocabs, parser, raise_on_error):
        """
        Build a RotowireDataset from the jsonl <filename>.
        ARGS:
            filename (str): Where to find raw jsonl
            config (RotowireConfig): see onmt.rotowire.config.py for info
            raise_on_error (Bool): whether to raise an error when a problematic
                                   line is encountered
            vocabs (Dict[Vocab]): if not None, not build vocabs.

        Note that some checks/warnings are performed early to save time
        when something goes wrong.

        TODO: Use multiprocessing to improve performances.
        """
        if config is None:
            logger.info('No config file was given, using defaults.')
            config = RotowireConfig.from_defaults()

        # Checking that vocabularies are valid
        vocabs = cls.check_vocabs(vocabs)

        logger.info(f'Prepocessing Rotowire file, found at {filename}')
        logger.info(config)

        examples = list()

        iterable = FileIterable.from_filename(filename, fmt='jsonl')
        desc = "Reading and formatting raw data"

        for idx, jsonline in tqdm.tqdm(enumerate(iterable),
                                       desc=desc, total=len(iterable)):
            ex = parser.parse_example(idx, jsonline)

            if ex is not None:
                examples.append(ex)
            elif raise_on_error:
                break

        # If any error was found, log them with a warning.
        # If on_error == 'raise', first found error would result in break
        parser.log_error_and_maybe_raise(do_raise=raise_on_error)

        dataset = cls(examples, **vocabs, config=config)

        return dataset


class RotowireGuidedInferenceDataset(RotowireInferenceDataset):

    @classmethod
    def build_from_raw_json(cls, filename, config, vocabs, raise_on_error=True):

        parser = RotowireInferenceParser(config=config, guided_inference=True)
        return cls._build_from_raw_json(filename=filename,
                                        config=config,
                                        vocabs=vocabs,
                                        parser=parser,
                                        raise_on_error=raise_on_error)

    def __getitem__(self, item):
        # In guided inference, we also need elaborations and context

        raw_example = self.examples[item]
        example = self._build_example_source(raw_example, item)

        # Numericalize elaborations
        elaborations = numericalize(raw_example['elaborations'], self.elab_vocab)
        example['elaborations'] = torch.LongTensor(elaborations)

        # Create and pad the contexts
        example['contexts'] = pad_sequence([
            torch.LongTensor(c) for c in raw_example['contexts']
        ], batch_first=True, padding_value=-1)

        return example
