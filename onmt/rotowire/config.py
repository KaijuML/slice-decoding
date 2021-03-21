from torchtext.vocab import Vocab
from collections import Counter


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Below are hard-coded global configs. Try and know what you're doing
# if you want to modify those!

# Note that the elab vocab also includes <unk> and <pad> on purpose.
elaborations = ['<none>', '<eod>']
elaboration_vocab = Vocab(Counter(elaborations))

config = Config(
    entity_size=30,
    num_primary_slices=28,
    elaboration_vocab=elaboration_vocab
)