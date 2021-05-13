from torchtext.vocab import Vocab
from collections import Counter

import more_itertools
import torch

from onmt.rotowire.template import TemplatePlan
from onmt.rotowire import RotowireConfig
from onmt.utils.logging import logger

from onmt.rotowire.exceptions import (
    RotowireParsingError,
    UnknownElaborationError,
    ContextTooLargeError
)


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
        Parse an example of raw Rotowire data. Log any error found.
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
        raise NotImplementedError()

    def build_input_view(self, view_data):
        """
        Builds a single view to the inputs of this example.
        This method is called on:
            - training & inference: all primary entities
            - training & guided inference: elaborations relevant to the summary.
            - inference: all elaborations
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

        return self.pad_entity(src_text), self.pad_entity(src_cols)

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
        For parallelization purposes, we will split the input tensor into entities.
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


class RotowireTrainingParser(RotowireParser):

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
            src_text, src_cols = self.build_input_view(view_data)

            source_vocab.update(src_text)
            cols_vocab.update(src_cols)

            input_sequence[0].append(src_text)
            input_sequence[1].append(src_cols)

        # We then add all elaborations that will be needed for the summary
        # We also remember for each one it index in the input_sequence
        elaboration_view_idxs = dict()
        for view_idx, elaboration in entity_elaborations:
            view_data = inputs[view_idx]["data"][self.reverse_elaboration_mapping[elaboration]]
            src_text, src_cols = self.build_input_view(view_data)

            source_vocab.update(src_text)
            cols_vocab.update(src_cols)

            input_sequence[0].append(src_text)
            input_sequence[1].append(src_cols)

            # Tracking where each elaboration is stored in the input sequence.
            # Very helpful when building contexts for decoding target sentences
            elaboration_view_idxs[view_idx, elaboration] = len(input_sequence[0]) - 1

        # We can now join everything as a long string, to be split on spaces
        example['src'] = [' '.join(more_itertools.collapse(seq))
                          for seq in input_sequence]
        example['source_vocab'] = Vocab(
            source_vocab, specials=['<unk>', '<pad>', '<ent>'])

        # The encoder final representation is based on primary entities only
        example['n_primaries'] = len(inputs)
        example['elaboration_view_idxs'] = elaboration_view_idxs

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


class RotowireInferenceParser(RotowireTrainingParser):

    def __init__(self, config, template_file=None, guided_inference=True):
        super().__init__(config=config)
        self.template_file = template_file
        self.guided_inference = guided_inference

        if self.template_file is not None and not self.guided_inference:
            raise ValueError('Templates can only be used during GuidedInference')

    def _parse_example(self, jsonline):

        inputs, outputs = jsonline['inputs'], jsonline['outputs']

        template = None
        if self.template_file is not None:
            template = TemplatePlan(
                self.template_file,  # formal templating language
                [e['data']['PRIMARY'] for e in inputs],  # raw data
                self.config  # config file
            )

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
        # primary entities will be elaborated during the summary. Elaborations
        # and grounding entities can be decided by the true plan or by the
        # user-defined template.
        # When going with the true plan, we take this opportunity to standardize
        # elaboration names, and when needed cast some weird elaborations to <none>.
        entity_elaborations = list()
        if template is None and self.guided_inference:
            for sentence in outputs:
                gt = self.elaboration_mapping.get(sentence['grounding_type'], None)
                if gt is None:
                    raise UnknownElaborationError(sentence['grounding_type'])
                sentence['grounding_type'] = gt  # normalization

                if sentence['grounding_type'] in {'<time>', '<event>'}:
                    entity_elaborations.extend([
                        [int(view_idx_str), sentence['grounding_type']]
                        for view_idx_str in sentence['grounding_data']
                    ])

        elif template is not None:
            for elaboration, view_idxs in template:
                if elaboration in {'<time>', '<event>'}:
                    entity_elaborations.extend([
                        [view_idx, elaboration]
                        for view_idx in view_idxs
                    ])

        # We first add all primary entities to the input tensors
        for view_dict in inputs:
            primary_key = self.reverse_elaboration_mapping['<primary>']
            view_data = view_dict["data"][primary_key]
            src_text, src_cols = self.build_input_view(view_data)

            source_vocab.update(src_text)
            cols_vocab.update(src_cols)

            input_sequence[0].append(src_text)
            input_sequence[1].append(src_cols)

        # At this point, we have included all primary views already, and that
        # might be enough for starting the decoding process. However, we need
        # to include grounding views in case of GuidedInference OR include all
        # views in case of Inference.

        # We need to remember the index of each elaboration in the source sequence.
        # For non-guided inference, it's pretty easy since there's no elabs.
        elaboration_view_idxs = dict()
        if self.guided_inference:
            # For GuidedInference, we also need the grounding views for each
            # of planned sentences (either true plan or template plan).
            for view_idx, elaboration in entity_elaborations:
                elaboration_key = self.reverse_elaboration_mapping[elaboration]
                view_data = inputs[view_idx]["data"][elaboration_key]
                src_text, src_cols = self.build_input_view(view_data)

                source_vocab.update(src_text)
                cols_vocab.update(src_cols)

                input_sequence[0].append(src_text)
                input_sequence[1].append(src_cols)

                # Tracking where each elaboration is stored in the input sequence.
                # Very helpful when building contexts for decoding target sentences
                elaboration_view_idxs[view_idx, elaboration] = len(input_sequence[0]) - 1

        # For both types of Inference, we want to parse all views, and build an
        # efficient mapping that can either be queried at each new sentence for
        # new grounding views, or used to build human readable plans
        elaborations_query_mapping = dict()
        for idx, view_dict in enumerate(inputs):
            elaborations_query_mapping[idx] = dict()
            for key in ['<time>', '<event>']:
                elaborations_key = self.reverse_elaboration_mapping[key]
                view_data = view_dict["data"][elaborations_key]
                src_text, src_cols = self.build_input_view(view_data)
                elaborations_query_mapping[idx][key] = [src_text, src_cols]
            elaborations_query_mapping[idx]['<primary>'] = view_dict["data"][self.reverse_elaboration_mapping['<primary>']]

        example['elaborations_query_mapping'] = elaborations_query_mapping

        # We can now join everything as a long string, to be split on spaces
        example['src'] = [' '.join(more_itertools.collapse(seq))
                          for seq in input_sequence]
        example['source_vocab'] = Vocab(
            source_vocab, specials=['<unk>', '<pad>', '<ent>'])

        # The encoder final representation is based on primary entities only
        example['n_primaries'] = len(inputs)
        example['elaboration_view_idxs'] = elaboration_view_idxs

        # We also build a src_map. This mapping assigns to each source token
        # its position in the source_vocab. This is used by the copy mechanism
        # to gather probas over source_vocab using attention over src.
        # At this stage, the map is flat, but DataLoader will one-hot everything.
        src_map = torch.LongTensor([example['source_vocab'][tok]
                                    for tok in example['src'][0].split()])
        example['src_map'] = src_map

        # The job might be done at that point. If we do not guide the inference
        # process, then the Inference helper has all it needs already.
        if not self.guided_inference:
            return example

        # From now on, we are in GuidedInference territory. We want to build
        # the elaboration and contexts lists, to fit one of two cases:
        #    1) Inference guided by the plan that was used by human annotators.
        #    2) Inference guided by template plan

        # This is a list containing the type of elaboration required to
        # write the associated sentence. Currently supports:
        #  - None
        #  - End of Document.
        elaborations = list()

        # Tracks which entities are relevant for a given sentence
        contexts = list()

        # Inference guided by template plan
        if template is not None:
            for elaboration, view_idxs in template:
                elaborations.append(elaboration)
                contexts.append(view_idxs)

                # We also add the slice used for elaborations <time> & <event>
                if elaboration in {'<time>', '<event>'}:
                    contexts[-1].extend([
                        elaboration_view_idxs[view_idx, elaboration]
                        for view_idx in contexts[-1]
                    ])

            # Everything is done for this type of GuidedInference

            example['elaborations'] = elaborations + ['<eod>']
            example['contexts'] = contexts

            return example

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

        example['elaborations'] = elaborations + ['<eod>']
        example['contexts'] = contexts

        return example
