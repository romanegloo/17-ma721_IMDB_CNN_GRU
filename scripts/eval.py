import argparse
import logging
import os
from pathlib import PosixPath

from IMDB import CnnImdbSA


def run(args):
    dataset = CnnImdbSA(args)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to aclImdb')
    args = parser.parse_args()

    # initiate logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # set defaults
    if args.data_path is None:
        args.data_path = os.path.join(
            PosixPath(__file__).absolute().parents[2].as_posix(), 'data/aclImdb')
    if not os.path.exists(args.data_path):
        raise RuntimeError('imdb path does not exist %s', args.data_path)

    run(args)

