from onmt.rotowire.parser import RotowireTemplateParser
from onmt.rotowire.utils import MultiOpen
from onmt.utils.logging import logger

import configargparse as argparse
import json
import tqdm
import os


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', required=False,
                        is_config_file_arg=True, help='config file path')
    parser.add_argument('--save-config', dest='save_config', required=False,
                        is_write_out_config_file_arg=True,
                        help='config file save path')

    group = parser.add_argument_group('File System')
    group.add_argument('--template-file', dest='template_file', required=True,
                       help="path toward template specification file")
    group.add_argument('--data-file', dest='data_file', required=True,
                       help='path to evaluation set file. Should be .jsonl')
    group.add_argument('--dest', dest='dest', required=True,
                       help="Where to save the realized templates")
    group.add_argument('--log-file', dest='log_file', default=None,
                       help='Logging preprocessing info')

    group = parser.add_argument_group('Script Behavior')
    group.add_argument('--dynamic-template', dest='dynamic_template',
                       action='store_true', help='set to true to use a different'
                                                 'template for all examples.')

    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args) if args else parser.parse_args()

    logger.init_logger(args.log_file)

    parser = RotowireTemplateParser(args.template_file, args.dynamic_template)

    if os.path.exists(args.dest):
        logger.warn(f'Overwriting {os.path.abspath(args.dest)}')

    n_examples = int(os.popen(f'wc -l < {args.data_file}').read())

    filenames = [args.data_file, args.dest]
    with MultiOpen(*filenames, modes=['r', 'w'], encoding='utf8') as files:
        datafile, destfile = files

        for jidx, jsonline in enumerate(tqdm.tqdm(datafile, total=n_examples)):
            realized_template = parser.parse_example(jidx, json.loads(jsonline))
            for line in realized_template:
                destfile.write(f'{line}\n')
            destfile.write('\n')

    logger.info(f'Realized templates available at: {os.path.abspath(args.dest)}')


if __name__ == '__main__':
    main()
