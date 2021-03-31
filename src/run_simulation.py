import datetime
import random

import config
from data_formatting import *
from initialiser import Initialiser

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    initialiser = Initialiser(
        num_clients=config.NUM_CLIENTS,
        client_fraction=config.CLIENT_FRACTION,
        iterations=config.ITERATIONS,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE
    )
    a = datetime.datetime.now()
    initialiser.run_simulation(iterations=config.ITERATIONS)
    b = datetime.datetime.now()