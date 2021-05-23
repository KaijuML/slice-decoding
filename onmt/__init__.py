""" Main entry point of the ONMT library """
from __future__ import division, print_function

import onmt.utils.optimizers
import onmt.encoders
import onmt.decoders
import onmt.models
import onmt.utils
import onmt.modules

from onmt.utils import logger

import sys
onmt.utils.optimizers.Optim = onmt.utils.optimizers.Optimizer
sys.modules["onmt.Optim"] = onmt.utils.optimizers

# For Flake
__all__ = [onmt.encoders, onmt.decoders, onmt.models,
           onmt.utils, onmt.modules, logger]

__version__ = "1.2.0"
