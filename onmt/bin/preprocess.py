from onmt.rotowire import RotoWireDataset
from onmt.rotowire import RotowireConfig


import configargparse as argparse


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add('-config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save-config', dest='save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')
    
    group = parser.add_argument_group('File System')
    group.add_argument('--train-file', dest='train_file', required=True,
                        help='path to training set file. Should be .jsonl')
    group.add_argument('--save-data', dest='save_data', required=True,
                        help='Where to save the data, as a prefix. ' \
                             '(We will save examples and vocabs in separate '\
                             'files)')
    group.add('--overwrite', action="store_true",
              help="Overwrite existing data if any.")
    
    group = parser.add_argument_group('Computing')
    group.add('--num-threads', dest='num_threads', type=int, default=1,
              help="Number of shards to build in parallel.")

    parser = RotowireConfig.add_rotowire_specific_args(parser)
    
    return parser


def main(args=None):
    
    parser = get_parser()
    args = parser.parse_args(args) if args else parser.parse_args()
    
    config = RotowireConfig.from_opts(args)
    dataset = RotoWireDataset.from_raw_json(args.train_file, config=config)

    dataset.dump(args.save_data, args.overwrite)


if __name__ == '__main__':
    main()
    
    
