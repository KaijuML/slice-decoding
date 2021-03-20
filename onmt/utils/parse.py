import configargparse as cfargparse
import os

import torch

import onmt.opts as opts
from onmt.utils.logging import logger


class ArgumentParser(cfargparse.ArgumentParser):
    def __init__(
            self,
            config_file_parser_class=cfargparse.YAMLConfigFileParser,
            formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
            **kwargs):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

    @classmethod
    def defaults(cls, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls()
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    @classmethod
    def update_model_opts(cls, opt):
        
        if isinstance(opt.encoder_layers, int) and opt.encoder_layers > 0:
            logger.info('opt.encoder_layers is specified, over-riding'
                        'low_level_layers and high_level_layers.')
            opt.low_level_layers = opt.encoder_layers
            opt.high_level_layers = opt.encoder_layers
            
        if isinstance(opt.encoder_heads, int) and opt.encoder_heads > 0:
            logger.info('opt.encoder_heads is specified, over-riding'
                        'low_level_heads and high_level_heads.')
            opt.low_level_heads = opt.encoder_heads
            opt.high_level_heads = opt.encoder_heads
            
        if isinstance(opt.encoder_glu_depth, int) and opt.encoder_glu_depth > 0:
            logger.info('opt.encoder_glu_depth is specified, over-riding'
                        'low_level_glu_depth and high_level_glu_depth.')
            opt.low_level_glu_depth = opt.encoder_glu_depth
            opt.high_level_glu_depth = opt.encoder_glu_depth

        opt.brnn = True

        if opt.copy_attn_type is None:
            opt.copy_attn_type = opt.global_attention

    @classmethod
    def validate_model_opts(cls, model_opt):
        pass

    @classmethod
    def ckpt_model_opts(cls, ckpt_opt):
        # Load default opt values, then overwrite with the opts in
        # the checkpoint. That way, if there are new options added,
        # the defaults are used.
        opt = cls.defaults(opts.model_opts)
        opt.__dict__.update(ckpt_opt.__dict__)
        return opt

    @classmethod
    def validate_train_opts(cls, opt):
        if opt.epochs:
            raise AssertionError(
                  "-epochs is deprecated please use -train_steps.")
      
        if torch.cuda.is_available() and not opt.use_gpu:
            logger.warn("You have a CUDA device, should run with --use_gpu")

    @classmethod
    def validate_translate_opts(cls, opt):
        if opt.beam_size != 1 and opt.random_sampling_topk != 1:
            raise ValueError('Can either do beam search OR random sampling.')

    @classmethod
    def validate_preprocess_args(cls, opt):
        assert opt.max_shard_size == 0, \
            "-max_shard_size is deprecated. Please use \
            -shard_size (number of examples) instead."
        assert opt.shuffle == 0, \
            "-shuffle is not implemented. Please shuffle \
            your data before pre-processing."

        assert len(opt.train_src) == len(opt.train_tgt), \
            "Please provide same number of src and tgt train files!"

        assert len(opt.train_src) == len(opt.train_ids), \
            "Please provide proper -train_ids for your data!"

        for file in opt.train_src + opt.train_tgt:
            assert os.path.isfile(file), "Please check path of %s" % file

        if len(opt.train_align) == 1 and opt.train_align[0] is None:
            opt.train_align = [None] * len(opt.train_src)
        else:
            assert len(opt.train_align) == len(opt.train_src), \
                "Please provide same number of word alignment train \
                files as src/tgt!"
            for file in opt.train_align:
                assert os.path.isfile(file), "Please check path of %s" % file

        assert not opt.valid_align or os.path.isfile(opt.valid_align), \
            "Please check path of your valid alignment file!"

        assert not opt.valid_src or os.path.isfile(opt.valid_src), \
            "Please check path of your valid src file!"
        assert not opt.valid_tgt or os.path.isfile(opt.valid_tgt), \
            "Please check path of your valid tgt file!"

        assert not opt.src_vocab or os.path.isfile(opt.src_vocab), \
            "Please check path of your src vocab!"
        assert not opt.tgt_vocab or os.path.isfile(opt.tgt_vocab), \
            "Please check path of your tgt vocab!"
