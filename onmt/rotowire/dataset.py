"""
Here I create a custom RotoWire dataset object.
TODO: multiprocessing + logging
"""
from onmt.rotowire.config import RotowireConfig
from onmt.rotowire.utils import FileIterable
from torch.nn.utils.rnn import pad_sequence
from onmt.utils.logging import logger
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from collections import Counter

from onmt.rotowire.exceptions import (
    RotowireParsingError,
    DataAlreadyExistsError,
    UnknownElaborationError,
    ContextTooLargeError,
)

import more_itertools
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

    elaboration_mapping = {
        # Good elaborations
        'PRIMARY': '<primary>',
        'EVENT': '<event>',
        'TIME': '<time>',

        # Weird elaborations
        'CONFLICT': '<primary>',
        'TOO_MANY': '<none>',
        'NONE': '<none>'

    }

    reverse_elaboration_mapping = {
        '<primary>': 'PRIMARY',
        '<event>': 'EVENT',
        '<time>': 'TIME',
    }

    def __init__(self, config):
        if not isinstance(config, RotowireConfig):
            raise TypeError(f'Unknown config object: {type(config)}')

        self.config = config

        self.error_logs = dict()

    def parse_example(self, idx, jsonline):
        """
        Parse an example of raw Rotowire data. Log any error found
        :param idx: index of jsonline (used for logging errors)
        :param jsonline: actual data
        :return: (parsed_example, example_main_vocab, example_cols_vocab)
                 OR
                 (None, None, None) when there was an error
        """
        try:
            return self._parse_example(jsonline)
        except Exception as err:
            err_name = err.__class__.__name__
            if err_name not in self.error_logs:
                self.error_logs[err_name] = list()
            self.error_logs[err_name].append(idx)
            return [None] * 3

    def log_error_and_maybe_raise(self, do_raise=True):

        # empty line to emphasize warnings
        if len(self.error_logs):
            logger.warn('')

        for err_name, problematic_idxs in self.error_logs.items():
            logger.warn(f'{err_name} was encountered at line'
                        f'{"s" if len(problematic_idxs)>1 else ""} '
                        f'{problematic_idxs if len(problematic_idxs)>1 else problematic_idxs[0]}')
        if len(self.error_logs):
            if do_raise:
                raise RotowireParsingError(self.error_logs)
            error_counts = [len(idxs) for _, idxs in self.error_logs.items()]
            logger.warn(f'{sum(error_counts)} lines were ignored.')

        # empty line to emphasize warnings
        if len(self.error_logs):
            logger.warn('')

    def _parse_example(self, jsonline):

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

        # Before building the input tensors, we must also consider that some
        # primary entities will be elaborated during the summary.
        # We take this opportunity to standardize elaboration names, and when
        # needed cast some weird elaborations to <none>.
        entity_elaborations = list()
        for sentence in outputs:
            gt = self.elaboration_mapping.get(sentence['grounding_type'], None)
            sentence['grounding_type'] = gt
            if gt is None:
                raise UnknownElaborationError(sentence['grounding_type'])

            if sentence['grounding_type'] in {'<time>', '<event>'}:
                entity_elaborations.extend([
                    [int(view_idx_str), sentence['grounding_type']]
                    for view_idx_str in sentence['grounding_data']
                ])

        # We first add all primary entities to the input tensors
        for view_dict in inputs:
            view_data = view_dict["data"][self.reverse_elaboration_mapping['<primary>']]
            self.add_input_view(view_data, cols_vocab, source_vocab, input_sequence)

        # We then add all elaborations that will be needed for the summary
        # We also remember for each one it index in the input_sequence
        elaboration_view_idxs = dict()
        for view_idx, elaboration in entity_elaborations:
            view_data = inputs[view_idx]["data"][self.reverse_elaboration_mapping[elaboration]]
            self.add_input_view(view_data, cols_vocab, source_vocab, input_sequence)
            elaboration_view_idxs[view_idx, elaboration] = len(input_sequence[0]) - 1

        # We can now join everything as a long string, to be split on spaces
        example['src'] = [' '.join(more_itertools.collapse(seq))
                          for seq in input_sequence]
        example['source_vocab'] = Vocab(
            source_vocab, specials=['<unk>', '<pad>', '<ent>'])

        # The encoder final representation is based on primary entities only
        example['n_primaries'] = len(inputs)

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

        for sidx, sentence in enumerate(outputs):

            elaborations.append(sentence['grounding_type'])

            # If elaboration is <none> then we do not add the slice.
            # The network will have to do with empty context.
            if sentence['grounding_type'] == '<none>':
                contexts.append([])
            else:
                assert len(sentence['grounding_data']) <= 2
                contexts.append(list(map(int, sentence['grounding_data'])))

            # We also add the slice used for elaborations <time> & <event>
            if sentence['grounding_type'] in {'<time>', '<event>'}:

                contexts[-1].extend([
                    elaboration_view_idxs[view_idx, sentence['grounding_type']]
                    for view_idx in contexts[-1]
                ])

            # Sanity check: we should have at most 2 primaries & 2 elaborations
            if not len(contexts[-1]) <= 4:
                raise ContextTooLargeError(sidx, len(contexts[-1]))

            # See self._clean_sentence.__doc__
            sentence_str = self._clean_sentence(
                sentence['text'], example['source_vocab'].itos)

            main_vocab.update(sentence_str.split())
            sentences.append(sentence_str)

            # To compute alignment for this sentence, we iterate over all tokens
            # and check whether they can be found in grounding slices.
            # Note that it may appear naive, because we break at the first
            # found token, but position of copied token is not relevant here,
            # only that it is copied. During training, scores will be gathered
            # across all appearances of this token.
            alignment = list()
            for token in sentence_str.split():
                _algn = 0
                for slice_idx in contexts[-1]:
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

    def add_input_view(self, view_data, cols_vocab, source_vocab, input_sequence):
        """
        Add a single view to the inputs of this example. This method is called
        for all primary entities AND for elaborations relevant to the summary.
        """

        # Starting all entities with an <ent> token, to learn aggregation
        src_text = ['<ent>']
        src_cols = ['<ent>']

        # Iterating over all (key, value) of the entity
        for key, value in view_data.items():
            if value == 'N/A' and not self.config.keep_na:
                continue

            # noinspection PyUnresolvedReferences
            if self.config.lowercase:
                value = value.lower()

            src_text.append(value.replace(' ', '_'))
            src_cols.append(key)

            source_vocab.update(src_text)
            cols_vocab.update(src_cols)

        input_sequence[0].append(self.pad_entity(src_text))
        input_sequence[1].append(self.pad_entity(src_cols))

    def _clean_sentence(self, sentence, vocab):
        """
        In here, we slightly help the copy mechanism.
        When we built the source sequence, we took all multi-words value
        and repalaced spaces by underscores. We replace those as well in
        the summaries, so that the copy mechanism knows it was a copy.
        It only happens with city names like "Los Angeles".
        """
        # noinspection PyUnresolvedReferences
        if self.config.lowercase:
            sentence = sentence.lower()

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
            logger.info('Loading default config.')
            self.config = RotowireConfig.from_defaults()

        self.elab_vocab = self.config.elaboration_vocab

        logger.info('Dataset loaded with the following config:')
        logger.info(self.config)
        logger.info(f'Number of examples: {len(self)}')
        logger.info(f'Size of vocabulary: {len(self.main_vocab)}')
        logger.info(f'Number of known columns: {len(self.cols_vocab)}')
        logger.info(f'Number of known elaborations: {len(self.elab_vocab)}')

    @property
    def vocabs(self):
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
        ], batch_first=True, padding_value=-1)
        
        # Adding all stuff that doesn't require processing
        example['src_map'] = raw_example['src_map']
        example['n_primaries'] = torch.LongTensor([raw_example['n_primaries']])
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
    def build_from_raw_json(cls, filename, config=None,
                            dest=None, overwrite=False,
                            raise_on_error=True,
                            vocabs=None):
        """
        Build a RotowireDataset from the jsonl <filename>.
        ARGS:
            filename (str): Where to find raw jsonl
            config (RotowireConfig): see onmt.rotowire.config.py for info
            dest (filename): if provided, will save the dataset to <dest>
            overwrite (Bool): whether to replace existing data
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

        if vocabs is not None:
            vocabs = cls.check_vocabs(vocabs)

        if dest is not None:
            cls.check_paths(dest, overwrite=overwrite)

        logger.info(f'Prepocessing Rotowire file, found at {filename}')
        logger.info(config)

        examples = list()
        main_vocab = Counter()
        cols_vocab = Counter()

        parser = RotowireParser(config=config)

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

        if vocabs is None:
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
