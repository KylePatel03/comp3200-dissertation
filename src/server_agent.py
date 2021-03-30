import math
import sys
from typing import Dict, List, Set, Tuple
import numpy as np
from datetime import datetime
import multiprocessing
from multiprocessing.pool import ThreadPool

from client_agent import ClientAgent
from directory import Directory
from utils import *
from agent import Agent
from message import Message
import tensorflow as tf
sys.path.append('..')

def client_computation_caller(inp):
    client_instance, message = inp
    return_message = client_instance.produce_weights(message=message)
    return return_message


def client_weights_returner(inp):
    client_instance, message = inp
    converged = client_instance.receive_weights(message)
    return converged


def client_agent_dropout_caller(inp):
    client_instance, message = inp
    __ = client_instance.remove_active_clients(message)
    return None

""" Server agent that averages weights and returns them to clients"""
class ServerAgent(Agent):

    def __init__(self, agent_number, model, test_dataset):
        super(ServerAgent, self).__init__(agent_number=agent_number, agent_type='server_agent')
        self.averaged_weights: Dict[int, List[np.array]] = {}
        self.model: tf.keras.Model = model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
        self.test_dataset: Tuple[np.array,np.array] = test_dataset


    """
        Execute the t (th) round of FL
        :param t The FL round in range [1..T]
        :param active_clients: The set of active clients (client ids)
    """
    def fl_round(self, t, active_clients: Set[int]):
        # Mapping of client id to their weights
        weights: Dict[int, List[np.array]] = {}
        directory: Directory = Directory.get_instance()

        m = multiprocessing.Manager()
        lock = m.Lock()
        # Send a message to each active client and invoke them to begin local model training
        with ThreadPool(len(active_clients)) as calling_pool:
            args = []
            for client_id in active_clients:
                # Create a message for each active client to begin training
                client_instance: ClientAgent = directory.clients[client_id]
                client_name = client_instance.name

                body = {
                    'iteration': t,
                    'lock': lock,
                    # The simulated time it takes to send the message
                    'simulated_time': directory.latency_dict[self.name][client_name]
                }
                arg = Message(sender_name=self.name, recipient_name=client_name, body=body)
                args.append((client_instance, arg))
            # For each active client, invoke ClientAgent.produce_weights() and store the returned message containing their updated local weights
            messages = calling_pool.map(client_computation_caller, args)

        server_logic_start = datetime.now()

        vals = {message.sender: message.body['weights'] for message in messages}
        #The time it takes for the last client - straggler to send the message
        simulated_time = find_slowest_time(messages)

        # Store the local weights and intercepts for each client
        for client_name, w in vals.items():
            weights[client_name] = w

        # Aggregate (average) each of the clients weights
        weights_total = weights.values()
        averaged_weights = list(map(lambda x: sum(x) / len(x), zip(*weights_total)))

        # Update the model weights
        self.model.set_weights(averaged_weights)
        #Model.main_model.set_weights(averaged_weights)

        # Set the averaged/federated weights and intercepts for the current timestep
        self.averaged_weights[t] = averaged_weights

        # Add time server logic takes
        server_logic_end = datetime.now()
        server_logic_time = server_logic_end - server_logic_start
        simulated_time += server_logic_time

        # Send a message containing the averaged weights etc. to each active client for this timestep
        with ThreadPool(len(active_clients)) as returning_pool:
            args = []
            for client_id in active_clients:
                client_instance: ClientAgent = directory.clients[client_id]
                client_name = client_instance.name
                body = {
                    'iteration': t,
                    'averaged_weights': averaged_weights,
                    'simulated_time': simulated_time + directory.latency_dict[self.name][client_name]
                }
                message = Message(sender_name=self.name, recipient_name=client_name, body=body)
                args.append((client_instance, message))
            # Invokes the client to receive the message => calls ClientAgent.receive_weights()
            return_messages = returning_pool.map(client_weights_returner, args)

    def main(self, num_clients, num_iterations, client_fraction):
        """
        Method invoked to start simulation. Prints out what clients have converged on what iteration.
        Also prints out accuracy for each client on each iteration (what weights would be if not for the simulation) and federated accuaracy.
        :param client_fraction: The fraction of clients to select per round
        :param num_clients: The number of clients
        :param num_iterations: Number of rounds/iterations to simulate
        """
        assert (num_clients >= 1)
        assert (num_iterations >= 1)
        assert (0 < client_fraction <= 1)

        # The probabilties of selecting each client to participate in a round
        p = np.full(num_clients, 1 / num_clients)
        # The number of clients to sample per round
        num_active_clients = math.ceil(client_fraction * num_clients)
        print('Selecting {}/{} clients per FL round'.format(num_active_clients, num_clients))

        # Execute num_iterations rounds of FL
        for t in range(1, num_iterations + 1):
            # Randomly select k clients
            active_clients = set(np.random.choice(a=num_clients, size=num_active_clients, replace=False, p=p))
            print('FL Round {}\nSelected clients {} to participate'.format(t, active_clients))
            self.fl_round(t, active_clients)
            print('Evaluating model performance at round {}\n{}'.format(t,self.evaluate_model()))
            self.evaluate_model()
        print('Finished FL simulation')

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
