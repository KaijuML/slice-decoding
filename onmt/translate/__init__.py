""" Modules for translation """
from onmt.translate.beam_search import BeamSearch, GNMTGlobalScorer
from onmt.translate.decode_strategy import DecodeStrategy
from onmt.translate.penalties import PenaltyBuilder

__all__ = ['BeamSearch',
           'GNMTGlobalScorer',
           'PenaltyBuilder',
           "DecodeStrategy", ]
