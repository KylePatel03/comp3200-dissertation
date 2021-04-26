import config
import argparse

from initialiser_edge import InitialiserEdge
from initialiser_vanilla import InitialiserVanilla


def main():
    # Command-line parser
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=('edge', 'vanilla'), default='edge', nargs='?')
    parser.add_argument('-num_clients', type=int, default=config.NUM_CLIENTS)
    parser.add_argument('-iterations', type=int, default=config.ITERATIONS)
    parser.add_argument('-client_fraction', type=float, default=config.CLIENT_FRACTION)
    parser.add_argument('-accuracy_threshold', type=float, default=config.ACCURACY_THRESHOLD)
    parser.add_argument('-epochs', type=int, default=config.EPOCHS)
    parser.add_argument('-batch_size', type=int, default=config.BATCH_SIZE)

    args_dict = vars(parser.parse_args())
    t = args_dict['type']
    args_dict.pop('type')

    if t == 'edge':
        run_simulation(InitialiserEdge, **args_dict)
    else:
        run_simulation(InitialiserVanilla, **args_dict)


def run_simulation(initialiser_f, **args):
    initialiser = initialiser_f(**args)
    initialiser.run_simulation(iterations=args['iterations'])


if __name__ == '__main__':
    main()
