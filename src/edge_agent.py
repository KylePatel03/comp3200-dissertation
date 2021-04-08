import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf

from agent import Agent
from directory import Directory
from message import Message
from queue import PriorityQueue
from typing import Dict, List

Weights = List[np.array]

"""
    Singleton Class
"""
class EdgeAgent(Agent):
    __edge_agent = None

    @staticmethod
    def get_instance():
        if EdgeAgent.__edge_agent is None:
            raise ValueError('EdgeAgent has not been initialised')
        else:
            return EdgeAgent.__edge_agent

    def __init__(self, agent_number, model, num_active_clients, epochs, batch_size):
        if EdgeAgent.__edge_agent is None:
            super(EdgeAgent, self).__init__(agent_number=agent_number, agent_type='edge_agent')
            self.__model: tf.keras.Model = model
            # Queue of messages to process (sorted by simulated times)
            self.__queue: PriorityQueue = PriorityQueue(num_active_clients)
            # Mapping of client name to mapping of round and local weights
            self.client_local_weights: Dict[str, Dict[int, Weights]] = {}
            self.epochs: int = epochs
            self.batch_size: int = batch_size
            EdgeAgent.__edge_agent = self
        else:
            raise ValueError('The EdgeAgent instance has already been created - access using get_instance()')


    """
        Invoked by the ClientAgent to add their message to the queue
    """
    def receive_message(self, message: Message):
        self.__queue.put((message.simulated_time, message))

    """
     Invoked by the ServerAgent to receive federated weights and update the local model
     :return: Return a message back to the ServerAgent to acknowledge the weights
     """
    def receive_weights(self, message: Message) -> Message:

        start_time = datetime.now()
        directory: Directory = Directory.get_instance()

        averaged_weights = message.body['averaged_weights']
        simulated_time = message.simulated_time

        self.__model.set_weights(averaged_weights)

        end_time = datetime.now()
        computation_time = end_time - start_time
        simulated_time += computation_time + directory.latency_dict[self.name][directory.main_server.name]

        return Message(sender_name=self.name,
                       recipient_name=directory.main_server.name,
                       iteration=message.iteration,
                       simulated_time=simulated_time,
                       body={}
                       )

    """
    Invoked by the ClientAgent when it is asked by the Server to produce weights
    Process a message from the queue of a client (receives features and labels and performs their local model update)
    :return: Return a message to the ServerAgent containing the updated new_weights
    """
    def produce_weights(self) -> Message:
        start_time = datetime.now()
        directory: Directory = Directory.get_instance()

        # Process the message from the client that is first received
        simulated_time, message = self.__queue.get()
        print('{}: Processing message from {}. Simulated time = {}'.format(self.name,message.sender,message.simulated_time))

        # Extract message contents
        features, labels = message.body['features'], message.body['labels']

        # Perform local model DNN training (using the federated weights at this round)
        initial_weights = self.__model.get_weights()
        self.__model.fit(x=features, y=labels, batch_size=self.batch_size, epochs=self.epochs, verbose=0)

        # Store the updated client weights and reset the models weights to be the federated weights at this round
        new_weights = self.__model.get_weights()
        self.__model.set_weights(initial_weights)

        # Store the clients local weights at this particular round
        self.client_local_weights.setdefault(message.sender, {})
        self.client_local_weights[message.sender][message.iteration] = new_weights

        # Update times
        end_time = datetime.now()
        computation_time = end_time - start_time
        # The time taken to do local model training + send a message to the ServerAgent
        simulated_time += computation_time + directory.latency_dict[self.name][directory.main_server.name]
        print('{}: Finished processing for {}. Simulated Time = {}'.format(self.name, message.sender, simulated_time))

        # Forward the message containing the client's local weights to the main server
        return Message(sender_name=message.sender,
                       recipient_name=directory.main_server.name,
                       iteration=message.iteration,
                       simulated_time=simulated_time,
                       body={
                           'new_weights': new_weights,
                           'num_data': labels.shape[0]
                       }
                       )


