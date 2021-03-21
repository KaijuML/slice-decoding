from configargparse import ArgumentParser
from torchtext.vocab import Vocab
from collections import Counter


# Below are hard-coded global configs. Try and know what you're doing
# if you want to modify those!

DEFAULTS = {
    'vocab_size': int(1e4),
    'entity_size': 30,
    'num_primary_slices': 28,
}


class RotowireConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in DEFAULTS:
                raise ValueError(f'{key} is not a known RotowireConfig option.')
            setattr(self, key, value)

        # Setting defaults when not specified by user
        for key, value in DEFAULTS.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

        # Note that the elab vocab also includes <unk> and <pad> on purpose.
        elaborations = ['<none>', '<eod>']
        self.elaboration_vocab = Vocab(Counter(elaborations))

    @classmethod
    def from_defaults(cls):
        return cls(
            vocab_size=DEFAULTS['vocab_size'],
            entity_size=DEFAULTS['entity_size'],
            num_primary_slices=DEFAULTS['num_primary_slices'],
        )

    @classmethod
    def from_opts(cls, opts):
        return cls(
            vocab_size=opts.vocab_size,
            entity_size=opts.entity_size,
            num_primary_slices=opts.num_primary_slices,
        )

    @staticmethod
    def add_rotowire_specific_args(parent_parser):
        """
        This methods is used for the preprocessing script to add any relevant
        arguments to the parser, without having to hunt them down / store them
        in a separate folder.

        :param parent_parser: the parser used in preprocessing.py
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Rotowire Dataset')

        group.add_argument("--vocab-size", type=int,
                           default=DEFAULTS['vocab_size'],
                           help="Size of the known vocabulary.")

        return parser
