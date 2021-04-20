from onmt.utils.logging import Logger

import pkg_resources
import argparse
import shutil
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Create an experiment folder")
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--copy', dest='copy', default=None, type=str,
                        help="If valid experiment, copy its .yaml files")

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    logger = Logger()
    args = parse_args()

    experiments = pkg_resources.resource_filename(__name__, 'experiments')
    if not os.path.exists(experiments):
        logger.info("Creating an 'experiments' folder.")
        os.makedirs(experiments)

    exp = os.path.join(experiments, args.name)
    if os.path.exists(exp):
        raise ValueError('An experiment with this name already exists')

    os.mkdir(exp)
    os.mkdir(os.path.join(exp, 'data'))
    os.mkdir(os.path.join(exp, 'models'))
    os.mkdir(os.path.join(exp, 'gens'))
    os.mkdir(os.path.join(exp, 'gens', 'test'))
    os.mkdir(os.path.join(exp, 'gens', 'validation'))

    if args.copy is not None:
        copied_exp = os.path.join(experiments, args.copy)
        if not os.path.exists(copied_exp):
            logger.warn(f'{copied_exp} is not a valid experiment to copy')
            logger.warn(f'Experiment {exp} was created empty.')
        else:
            yamls = [file for file in os.listdir(copied_exp)
                     if file.endswith('.yaml')]
            if not len(yamls):
                logger.warn(f'No *.yaml were found in {copied_exp}')
                logger.warn(f'Experiment {exp} was created empty.')
            else:
                logger.info(f'Experiment {args.name} created.')
                logger.info(f'Copying the following files from {copied_exp}')
                for yaml in yamls:
                    src = os.path.join(copied_exp, yaml)
                    dst = os.path.join(exp, yaml)
                    shutil.copyfile(src, dst)
                    logger.info(f'{yaml}')
    else:
        logger.info(f'Experiment {args.name} created.')
