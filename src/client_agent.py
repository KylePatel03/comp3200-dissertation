import numpy as np
from warnings import simplefilter
import tensorflow as tf

from agent import Agent

from abc import ABC, abstractmethod
from typing import Tuple, Dict, List

from message import Message

simplefilter(action='ignore', category=FutureWarning)

Dataset = Tuple[np.array, np.array]
AgentDataset = Dict[int, Dataset]
Weights = List[np.array]


class ClientAgent(Agent, ABC):

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


    @abstractmethod
    def produce_weights(self, message: Message) -> Message:
        pass

    @abstractmethod
    def receive_weights(self, message: Message) -> Message:
        pass

