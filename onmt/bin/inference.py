from onmt.utils.misc import set_random_seed, Container, grouped
from onmt.inference import build_inference
from onmt.utils.logging import logger
import multiprocessing as mp

import configargparse as argparse
import torch
import math
import os


def configure_process(seed, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(seed, device_id >= 0)


def guess_if_validation_or_test(filename):
    """
    Currently, our filenames are test.jsonl or validation.jsonl
    """
    return 'test' if 'test' in filename else 'validation'


def regularize_args(args):
    """
    For several arguments, config file accept both single or multi args.
    In this function, I regularize everything by creating a list with one item
    if needed.
    """
    if args.gpu is not None:
        args.gpus = [args.gpu]
    elif args.gpus is None:
        args.gpus = [-1]

    if args.checkpoint is not None:
        args.checkpoints = [args.checkpoint]

    return args


def build_container(args, step, gpu):
    """
    Build a fake Namespace for an inference run.
    """
    hparams = [
        ['step', step],
        ['guided', args.guided_inference],
        ['bms', args.beam_size],
        ['blk', args.block_ngram_repeat]
    ]

    dest = guess_if_validation_or_test(args.source_file)
    dest = os.path.join('experiments', args.experiment, 'gens', dest)
    dest = os.path.join(dest, f'{".".join(f"{k}={v}" for k, v in hparams)}.txt')

    model_path = os.path.join('experiments',
                              args.experiment,
                              "models",
                              f'model_step_{step}.pt')

    return Container(
        guided_inference=args.guided_inference,

        source_file=args.source_file,
        model_path=model_path,
        dest=dest,

        batch_size=args.batch_size,
        beam_size=args.beam_size,

        min_sent_length=args.min_sent_length,
        max_sent_length=args.max_sent_length,

        block_ngram_repeat=args.block_ngram_repeat,
        ignore_when_blocking=args.ignore_when_blocking,

        log_file=args.log_file,
        seed=args.seed,
        gpu=gpu,
    )


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', required=False,
                        is_config_file_arg=True, help='config file path')
    parser.add_argument('--save-config', dest='save_config', required=False,
                        is_write_out_config_file_arg=True,
                        help='config file save path')

    group = parser.add_argument_group('File System')
    group.add_argument('--source-file', dest='source_file', required=True,
                       help='path to evaluation set file. Should be .jsonl')
    group.add_argument('--experiment', dest='experiment', required=True,
                       help="Experiment folder (e.g. exp-1/)")
    group.add_argument('--log-file', dest='log_file',
                       help='Logging preprocessing info')

    group = parser.add_argument_group('Inference (high level)')
    group.add_argument('--guided-inference', dest='guided_inference',
                       action='store_true', help='Use the true plans or not.')

    # Model checkpoints, one or many
    ckpts = parser.add_mutually_exclusive_group(required=True)
    ckpts.add_argument('--checkpoints', dest="checkpoints",
                       nargs='+', type=int, help="model steps to evaluate")
    ckpts.add_argument('--checkpoint', dest="checkpoint", type=int,
                       help="model step to evaluate")

    group = parser.add_argument_group('Decoding')
    group.add_argument('--batch-size', dest="batch_size", type=int)
    group.add_argument('--beam-size', dest='beam_size', type=int)
    group.add_argument('--min-sent-length', dest='min_sent_length', type=int)
    group.add_argument('--max-sent-length', dest='max_sent_length', type=int)
    group.add_argument('--block-ngram-repeat', dest="block_ngram_repeat",
                       type=int, help="Block repetitions of ngrams")
    group.add_argument('--ignore-when-blocking', dest="ignore_when_blocking",
                       nargs='+', type=str, default=list(),
                       help="Don't block those ngrams")

    # GPUs, one or many
    gpus = parser.add_mutually_exclusive_group(required=False)
    gpus.add_argument('--gpu', dest='gpu', type=int,
                       help='One gpu to use for all inference runs')
    gpus.add_argument('--gpus', dest='gpus', type=int, nargs='+',
                       help='GPUs to accelerate computations. Only one is used'
                            'per inference run. If multiple ckpts are evaluated'
                            'and multiple gpus are selected, then multiple runs'
                            'can be done in parallel.')

    parser.add_argument('--seed', dest='seed', type=int, default=None)

    return parser


def single_main(args):
    configure_process(args.seed, args.gpu)
    inference = build_inference(args, logger)
    inference.run(args.source_file, args.batch_size, if_file_exists='overwrite')


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args) if args else parser.parse_args()
    args = regularize_args(args)

    logger.init_logger(args.log_file)

    if len(args.gpus) == 1:
        logger.info(f'Running inference script on {args.gpu=}')

        for step in args.checkpoints:
            single_main(build_container(args, step, args.gpus[0]))

    else:
        logger.info(f'Doing {len(args.gpus)} inference runs in parallel, on '
                    f'gpus {", ".join(str(gpu) for gpu in args.gpus)}.')

        gpus = args.gpus * math.ceil(len(args.checkpoints) / len(args.gpus))
        containers = [
            build_container(args, ckpt, gpu)
            for ckpt, gpu in zip(args.checkpoints, gpus)
            if ckpt is not None
        ]

        with mp.Pool(processes=len(args.gpus)) as pool:
            _iterable = pool.imap(
                single_main,
                containers,
            )

            for _ in _iterable:
                pass


if __name__ == '__main__':
    main()
