import copy
import numpy as np
from datetime import datetime
from warnings import simplefilter

import tensorflow as tf

from agent import Agent
from config import VERBOSITY
from directory import Directory
from edge_agent import EdgeAgent
from message import Message

from typing import Optional, Tuple, Dict, List

simplefilter(action='ignore', category=FutureWarning)

Dataset = Tuple[np.array, np.array]
AgentDataset = Dict[int, Dataset]
Weights = List[np.array]

class ClientAgent(Agent):

    """
    Attributes:
        train_datasets: Client training dataset
        model: The client local model
        computation_times: Dictionary mapping iteration to computation time
        personal_weights, federated_weights: Dictionary mapping iteration to new_weights (list of numpy arrays)
        personal_accuracy, federated_accuracy: Dictionary mapping iteration to accuracies on test dataset
    """
    def __init__(self, agent_number, train_datasets, model):
        super(ClientAgent, self).__init__(agent_number=agent_number, agent_type="client_agent")
        self.train_datasets: AgentDataset = train_datasets
        self.model: tf.keras.Model = model

        # Dictionaries indexed by timestep/iteration
        self.computation_times: Dict[int, float] = {}
        self.personal_accuracy: Dict[int, float] = {}
        self.federated_accuracy: Dict[int, float] = {}
        self.personal_weights: Dict[int, Weights] = {}
        self.federated_weights: Dict[int, Weights] = {}


    """
        Invoked by the ServerAgent to do model training
        Client computes the forward pass on their training data and invoke the EdgeServer to produce the weight update
        :returns: Message to the EdgeServer to produce weight update (not necessarily from this Client)
    """
    def produce_weights(self, message: Message) -> Message:
        start_time = datetime.now()

        directory: Directory = Directory.get_instance()
        edge_server: EdgeAgent = EdgeAgent.get_instance()
        simulated_time = message.simulated_time

        # Retrieve the training dataset for the current iteration
        training_data = self.train_datasets[message.iteration]
        x_train, y_train = training_data

        # Compute the local model forward pass
        features = self.model.predict(x_train, batch_size=edge_server.batch_size)

        # Update times
        end_time = datetime.now()
        computation_time = end_time - start_time
        self.computation_times[message.iteration] = computation_time
        simulated_time += computation_time + directory.latency_dict[self.name][edge_server.name]

        body = {
            'features': features,
            'labels': y_train,
        }

        msg = Message(sender_name=self.name,
                      recipient_name=edge_server.name,
                      iteration=message.iteration,
                      simulated_time=simulated_time,
                      body=body
                      )
        edge_server.receive_message(msg)
        return edge_server.produce_weights()
