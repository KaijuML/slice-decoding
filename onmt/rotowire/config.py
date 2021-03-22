from configargparse import ArgumentParser
from torchtext.vocab import Vocab
from collections import Counter


# Below are hard-coded global configs. Try and know what you're doing
# if you want to modify those!

DEFAULTS = {
    'vocab_size': int(1e4),
    'entity_size': 30,
    'num_primary_slices': 28,
    'keep_na': False,
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

    def __repr__(self):
        current_config = [f'{key}={getattr(self, key)}' for key in DEFAULTS]
        return f'RotowireConfig({", ".join(current_config)})'

    @classmethod
    def from_defaults(cls):
        return cls(**DEFAULTS)

    @classmethod
    def from_opts(cls, opts):
        kwargs = {key: getattr(opts, key, value)
                  for key, value in DEFAULTS.items()}
        return cls(**kwargs)

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
        group.add_argument("--entity-size", type=int,
                           default=DEFAULTS['entity_size'],
                           help="Size of each entity.")
        group.add_argument('--keep-na', action='store_true',
                           help="Do not discard N/A values in data.")

        return parser
