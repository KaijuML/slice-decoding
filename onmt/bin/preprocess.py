from onmt.rotowire import RotoWireDataset
from onmt.rotowire import RotowireConfig
from onmt.utils.logging import logger


import configargparse as argparse


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', required=False,
                        is_config_file_arg=True, help='config file path')
    parser.add_argument('--save-config', dest='save_config', required=False,
                        is_write_out_config_file_arg=True,
                        help='config file save path')
    
    group = parser.add_argument_group('File System')
    group.add_argument('--train-file', dest='train_file', required=True,
                       help='path to training set file. Should be .jsonl')
    group.add_argument('--save-data', dest='save_data', required=True,
                       help='Where to save the data, as a prefix. '
                            '(We will save examples and vocabs in separate '
                            'files)')
    group.add_argument('--log-file', dest='log_file', required=True,
                       help='Logging preprocessing info')
    group.add_argument('--overwrite', action="store_true",
                       help="Overwrite existing data if any.")
    
    group = parser.add_argument_group('Computing')
    group.add_argument('--num-threads', dest='num_threads', type=int, default=1,
                       help="Number of shards to build in parallel.")

    parser = RotowireConfig.add_rotowire_specific_args(parser)
    
    return parser


def main(args=None):
    
    parser = get_parser()
    args = parser.parse_args(args) if args else parser.parse_args()

    logger.init_logger(args.log_file)
    
    config = RotowireConfig.from_opts(args)

    # The following classmethod builds and save the dataset on its own
    RotoWireDataset.build_from_raw_json(
        filename=args.train_file,
        config=config,
        dest=args.save_data,
        overwrite=args.overwrite
    )


if __name__ == '__main__':
    main()
