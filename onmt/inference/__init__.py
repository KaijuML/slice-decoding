""" Modules for translation """
from onmt.inference.beam_search import BeamSearch, GNMTGlobalScorer
from onmt.inference.decode_strategy import DecodeStrategy
from onmt.inference.penalties import PenaltyBuilder
from onmt.inference.inference import GuidedInference, Inference, build_inference

__all__ = ['BeamSearch',
           'GNMTGlobalScorer',
           'PenaltyBuilder',
           "DecodeStrategy",
           "Inference",
           "GuidedInference",
           "build_inference"]
