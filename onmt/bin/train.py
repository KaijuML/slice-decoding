#!/usr/bin/env python
"""I have merge train/train_single for this project."""


from onmt.rotowire import RotoWireDataset, build_dataset_iter
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.utils.parse import ArgumentParser
from onmt.model_builder import build_model
from onmt.models import build_model_saver
from onmt.trainer import build_trainer
from onmt.utils.logging import logger

import onmt.opts
import torch
import os


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def train(opt):
    
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)
    
    # If you want another device, set CUDA_VISIBLE_DEVICES
    device_id = 0 if opt.use_gpu else -1
    
    configure_process(opt, device_id)
    logger.init_logger(opt.log_file, overwrite_log_file=opt.overwrite_log_file)
    
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info(f'Loading checkpoint & vocab from {opt.train_from}')
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
    else:
        checkpoint = None
        model_opt = opt
        
    # Load dataset examples and vocabs
    dataset = RotoWireDataset.load(opt.data)
    vocabs = dataset.get_vocabs()
    if checkpoint is not None:
        assert all(v == vocabs[k] for k, v in checkpoint['vocabs'].items())

    # Build model.
    model = build_model(model_opt, opt, vocabs, dataset.config, checkpoint)
    model.count_parameters(log=logger.info)
    
    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt,
                                    model, vocabs, dataset.config,
                                    optim)

    trainer = build_trainer(opt, model, vocabs, optim,
                            model_saver=model_saver)

    train_iter = build_dataset_iter(dataset, opt, device_id)
    # Note: for now we do not deal with validation data during training.

    if device_id == 0:
        _true_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        logger.info(f'Starting training on GPU: {_true_gpu}')
    else:
        logger.info('Starting training on CPU, could be very slow')
        
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()

        
def get_parser():
    parser = ArgumentParser(description='train.py')

    onmt.opts.config_opts(parser)
    onmt.opts.model_opts(parser)
    onmt.opts.train_opts(parser)

    return parser


def main(args=None):
    
    parser = get_parser()
    args = parser.parse_args(args) if args else parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()