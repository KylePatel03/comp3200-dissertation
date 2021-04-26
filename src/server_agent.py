from abc import abstractmethod, ABC
from typing import Tuple, Dict, List, Set

import math
import sys
import numpy as np
import tensorflow as tf

from datetime import datetime
import multiprocessing
from multiprocessing.pool import ThreadPool

from client_agent import ClientAgent
from directory import Directory
from message import Message

from agent import Agent, find_slowest_time

sys.path.append('..')

""" Server agent that averages new_weights and returns them to clients"""
class ServerAgent(Agent, ABC):

    def __init__(self, agent_number, model, test_dataset):
        super(ServerAgent, self).__init__(agent_number=agent_number, agent_type='server_agent')
        self.averaged_weights = {}
        self.model: tf.keras.Model = model
        self.test_dataset = test_dataset

    """
        The logic to simulate the t(th) FL round
        active_clients: Collection of participating clients by their id
    """
    @abstractmethod
    def fl_round(self, t, active_clients):
        pass

    """
        Run FL simulation, selecting a fraction of clients per round
        The simulation terminates when the number of iterations/model accuracy reaches a specified value
    """
    def main(self, num_clients, num_iterations, client_fraction, accuracy_threshold):
        assert (num_clients >= 1)
        assert (num_iterations >= 1)
        assert (0 < client_fraction <= 1)
        assert (0 < accuracy_threshold <= 1)

        # The probabilities of selecting each client to participate in a round
        p = np.full(num_clients, 1 / num_clients)
        # The number of clients to sample per round
        num_active_clients = math.ceil(client_fraction * num_clients)

        rng = np.random.default_rng(seed=0)  # Seed for reproducibility
        # Each row contains the ids of the clients to participate in a particular FL round
        active_clients = np.array([rng.choice(a=num_clients, size=num_active_clients, replace=False, p=p)
                                   for _ in range(num_iterations)])

        print('Selecting {}/{} clients per FL round'.format(num_active_clients, num_clients))
        for i, clients in enumerate(active_clients):
            t = i + 1
            print('FL Round {}\nSelected clients {} to participate'.format(t, clients))
            self.fl_round(t, clients)
            print('Evaluating model performance...')
            loss, accuracy = self.evaluate_model()
            print('Model loss & accuracy = {} {}%'.format(loss, 100 * accuracy))
            if accuracy >= accuracy_threshold:
                print('Accuracy threshold of {} has been achieved! Ending simulation at round {}'.
                      format(accuracy_threshold, t))
                break
            print()
        print('Finished FL simulation ')

    """
        Return loss and accuracy of model on testing dataset
    """
    def evaluate_model(self):
        x_test, y_test = self.test_dataset
        return self.model.evaluate(x_test, y_test, batch_size=2, verbose=0)
