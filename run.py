from onmt.bin.preprocess import main as preprocess
from onmt.bin.inference import main as inference
from onmt.bin.train import main as train
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parser shenanigans: I want to be able to also print the help
    # of downstream scripts. I remove the help action from this one
    # and only call it if no downstream script has been selected.
    help_action = parser._actions[0]
    parser._remove_action(help_action)
    [parser._option_string_actions.pop(s) for s in help_action.option_strings]

    # Simply add an argument for preprocess, train, inference
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preprocess", dest='preprocess', action='store_true',
                      help="Preprocess data")
    mode.add_argument("--train", dest='train', action='store_true',
                      help="Run training script")
    mode.add_argument("--inference", dest='inference', action='store_true',
                      help="Run inference script using a trained model")

    group = parser.add_argument_group('Help')
    group.add_argument('--help', dest='help', action='store_true')

    current_args, remaining_args = parser.parse_known_args()

    if current_args.preprocess:
        preprocess(remaining_args)
    elif current_args.train:
        train(remaining_args)
    elif current_args.inference:
        inference(remaining_args)
    else:
        help_action(parser, current_args, None)
