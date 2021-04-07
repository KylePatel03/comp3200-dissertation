import math
import sys
from abc import abstractmethod, ABC
from typing import Tuple, Dict, List, Set

import numpy as np
import tensorflow as tf

from agent import Agent

sys.path.append('..')

Dataset = Tuple[np.array, np.array]
AgentDataset = Dict[int, Dataset]
Weights = List[np.array]

""" Server agent that averages new_weights and returns them to clients"""
class ServerAgent(Agent, ABC):

    def __init__(self, agent_number, model, test_dataset):
        super(ServerAgent, self).__init__(agent_number=agent_number, agent_type='server_agent')
        self.averaged_weights: Dict[int, Weights] = {}
        self.model: tf.keras.Model = model
        self.test_dataset: AgentDataset = test_dataset

    @abstractmethod
    def fl_round(self, t: int, active_clients: Set[int]):
        pass

    """
        Run FL simulation, selecting a fraction of clients per round
        The simulation terminates when the number of iterations/model accuracy reaches a specified value
    """
    def main(self, num_clients, num_iterations, client_fraction, accuracy_threshold):
        assert (num_clients >= 1)
        assert (num_iterations >= 1)
        assert (0 < client_fraction <= 1)

        # The probabilities of selecting each client to participate in a round
        p = np.full(num_clients, 1 / num_clients)
        # The number of clients to sample per round
        num_active_clients = math.ceil(client_fraction * num_clients)
        print('Selecting {}/{} clients per FL round'.format(num_active_clients, num_clients))

        for t in range(1, num_iterations + 1):
            # Subset of clients to participate in FL round
            active_clients = set(np.random.choice(a=num_clients, size=num_active_clients, replace=False, p=p))
            print('FL Round {}\nSelected clients {} to participate'.format(t, active_clients))
            self.fl_round(t, active_clients)
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

    # def final_statistics(self):
    #     """
    #     USED FOR RESEARCH PURPOSES.
    #     """
    #     # for research purposes
    #     client_accs = []
    #     fed_acc = []
    #     for client_name, client_instance in self.directory.clients.items():
    #         fed_acc.append(list(client_instance.federated_accuracy.values()))
    #         client_accs.append(list(client_instance.personal_accuracy.values()))
    #
    #     if config.CLIENT_DROPOUT:
    #         print('Federated accuracies are {}'.format(dict(zip(self.directory.clients, fed_acc))))
    #     else:
    #         client_accs = list(np.mean(client_accs, axis=0))
    #         fed_acc = list(np.mean(fed_acc, axis=0))
    #         print('Personal accuracy on final iteration is {}'.format(client_accs))
    #         print('Federated accuracy on final iteration is {}'.format(fed_acc))  # should all be the same if no dropout
