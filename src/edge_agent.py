from datetime import datetime, timedelta
import tensorflow as tf

from agent import Agent
from directory import Directory
from message import Message
from queue import PriorityQueue

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
            self.__queue: PriorityQueue = PriorityQueue(num_active_clients)
            self.epochs: int = epochs
            self.batch_size: int = batch_size
            EdgeAgent.__edge_agent = self
        else:
            raise ValueError('The EdgeAgent instance has already been created - access using get_instance()')


    def receive_message(self, message: Message):
        time = message.simulated_time
        self.__queue.put((time, message))

    """
     Invoked by the ServerAgent to receive federated weights
     """
    def receive_weights(self, message: Message):

        b = message.body
        iteration, averaged_weights = b['iteration'], b['averaged_weights']
        self.__model.set_weights(averaged_weights)

    """
    Receives features from client and performs local model training
    :return: Return a message to the ServerAgent containing the updated new_weights
    """
    def produce_weights(self) -> Message:
        start_time = datetime.now()
        directory: Directory = Directory.get_instance()

        # Process the message from the client that is first received
        simulated_time, message = self.__queue.get()
        print('Processing {}'.format(message))

        # Extract message contents
        iteration = message.body['iteration']
        simulated_time = message.simulated_time
        features, labels = message.body['features'], message.body['labels']

        # Perform local model DNN training
        print('Training for client: {}'.format(message.sender) + '...')
        self.__model.fit(x=features, y=labels, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        weights = self.__model.get_weights()
        print('Finished training for client: {}'.format(message.sender))

        # Store the clients local weights in dictionary...

        # Update times
        end_time = datetime.now()
        computation_time = end_time - start_time
        # The time taken to do local model training + send a message to the ServerAgent
        simulated_time += computation_time + directory.latency_dict[self.name][directory.main_server.name]

        body = {
            'iteration': iteration,
            'new_weights': weights,
            'simulated_time': simulated_time,
            'computation_time': computation_time
        }
        # Forward the message from the client to the main server
        return Message(sender_name=message.sender, recipient_name=directory.main_server.name, body=body)


