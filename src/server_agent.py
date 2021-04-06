import math
import multiprocessing
import sys
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool

import numpy as np
import tensorflow as tf

from agent import Agent
from client_agent import ClientAgent
from directory import Directory
from message import Message
from utils import *

from typing import Tuple, Dict, List, Set

sys.path.append('..')

Dataset = Tuple[np.array, np.array]
AgentDataset = Dict[int, Dataset]
Weights = List[np.array]

""" Server agent that averages new_weights and returns them to clients"""


class ServerAgent(Agent):

    def __init__(self, agent_number, model, test_dataset):
        super(ServerAgent, self).__init__(agent_number=agent_number, agent_type='server_agent')
        self.averaged_weights: Dict[int, Weights] = {}
        self.model: tf.keras.Model = model
        self.test_dataset: AgentDataset = test_dataset

    """
        Execute the t (th) round of FL
        :param t The FL round in range [1..T]
        :param active_clients: The set of active clients (client ids)
    """

    def __fl_round(self, t, active_clients: Set[int]):
        server_logic_start = datetime.now()
        # Mapping of client id to their new_weights
        weights: Dict[int, Weights] = {}
        directory: Directory = Directory.get_instance()

        m = multiprocessing.Manager()
        # Send a message to each active client and invoke them to begin local model training
        with ThreadPool(len(active_clients)) as calling_pool:
            args = []
            for client_id in active_clients:
                # Create a message for each active client to begin training
                client_instance: ClientAgent = directory.clients[client_id]
                client_name = client_instance.name

                arg = Message(
                    sender_name=self.name,
                    recipient_name=client_name,
                    iteration=t,
                    simulated_time=directory.latency_dict[self.name][client_name],
                    body={}
                )
                args.append((client_instance, arg))
            # Invoke ClientAgent.produce_weights() and store their returned message containing weight updates
            messages = calling_pool.map(lambda x: x[0].produce_weights(message=x[1]), args)

        # The time it takes for ALL clients to do model training and send their weights
        # This is identical to the simulated time for the LAST client to execute model training
        receive_weights_time = find_slowest_time(messages)

        print('{}: Received all of the selected clients weights at time {}'.format(self.name, receive_weights_time))

        # The edge weights weighted by the number of training datapoints
        edge_weights_nk = [map(lambda x: message.body['num_data'] * x, message.body['new_weights'])
                           for message in messages]
        # The total number of training datapoints
        nk_sum = sum([message.body['num_data'] for message in messages])

        # Aggregate (average) each of the clients new_weights, weighted by the number of local training datapoints
        averaged_edge_weights: Weights = list(map(lambda x: sum(x) / nk_sum, zip(*edge_weights_nk)))
        num_trainable_weights = len(self.model.trainable_weights)

        averaged_weights = self.model.get_weights()
        averaged_weights[-num_trainable_weights:] = averaged_edge_weights

        # Update the model's weights with the average weights
        self.model.set_weights(averaged_weights)
        # Set the averaged/federated new_weights and intercepts for the current timestep
        self.averaged_weights[t] = averaged_weights

        server_logic_end = datetime.now()
        server_logic_time = server_logic_end - server_logic_start
        # The total time taken to request for weights, perform aggregation and send averaged weights to EdgeServer
        simulated_time = receive_weights_time + server_logic_time + directory.latency_dict[self.name][directory.edge.name]

        print('{}: Simulated time to send EdgeServer the federated weights = {}'.format(self.name, simulated_time))

        message = Message(
            sender_name=self.name,
            recipient_name=directory.edge.name,
            iteration=t,
            simulated_time=simulated_time,
            body={
                'averaged_weights': averaged_edge_weights,
            }
        )
        # Invoke the EdgeServer to receive the message and receive an acknowledgement
        return_msg: Message = directory.edge.receive_weights(message)
        print('{}: Simulated time = {}'.format(self.name, return_msg.simulated_time))

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
            self.__fl_round(t, active_clients)
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
