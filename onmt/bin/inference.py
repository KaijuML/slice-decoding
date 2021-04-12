from onmt.utils.misc import set_random_seed
from onmt.inference import build_translator
from onmt.utils.logging import logger

import configargparse as argparse
import torch


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


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
    group.add_argument('--dest', dest="dest", required=True,
                       help="where to write decoded descriptions")
    group.add_argument('--model-path', dest="model_path", required=True,
                       help="trained model")
    group.add_argument('--log-file', dest='log_file',
                       help='Logging preprocessing info')

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

    group = parser.add_argument_group('GPUs')
    group.add_argument('--gpu', dest='gpu', type=int)
    group.add_argument('--seed', dest='seed', default=None)

    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args) if args else parser.parse_args()

    logger.init_logger(args.log_file)

    translator = build_translator(args, logger)

    translator.run(args.source_file, args.batch_size)


if __name__ == '__main__':
    main()