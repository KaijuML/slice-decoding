"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.hierarchical_transformer import HierarchicalTransformerEncoder


__all__ = ["EncoderBase", "HierarchicalTransformerEncoder"]
