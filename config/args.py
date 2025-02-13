import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], dest='mode', default='train')
parser.add_argument('--history_length', type=int, dest='history_length', default=100,
                    help='Specify the length of the historical item sequence')
parser.add_argument('--embedding_size', type=int, dest='embedding_size', default=32,
                    help='Embedding_size')

args = parser.parse_known_args()[0]
