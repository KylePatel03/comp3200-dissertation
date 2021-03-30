from config import VERBOSITY
import copy
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from warnings import simplefilter
import numpy as np
import tensorflow as tf
from agent import Agent
from directory import Directory
from message import Message

simplefilter(action='ignore', category=FutureWarning)

class ClientAgent(Agent):
    epochs: Optional[int] = None
    batch_size: Optional[int] = None

    """
    Attributes:
        train_datasets: Dictonary mapping iteration (int) to a training dataset (numpy array)
        evaluator: ModelEvaluator used to evaluate local/federated weights on a testing dataset
        directory: Directory instance
        model: The client local model
        computation_times: Dictionary mapping tieration to computation ti,e
        personal_weights, federated_weights: Dictionary mapping iteration to weights (list of numpy arrays)
        personal_accuracy, federated_accuracy: Dictionary mapping iteration to accuracies on test dataset
    """

    def __init__(self, agent_number, train_datasets, model):
        """
        Initializes an instance of client agent & dictionaries indexed by iteration (int)
        """
        super(ClientAgent, self).__init__(agent_number=agent_number, agent_type="client_agent")

        self.train_datasets: Dict[int, Tuple[np.array, np.array]] = train_datasets

        self.model: tf.keras.Model = model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        # Dictionaries indexed by timestep/iteration (int) in range [1,T]
        self.computation_times: Dict[int, float] = {}
        self.personal_accuracy: Dict[int, float] = {}
        self.federated_accuracy: Dict[int, float] = {}

        self.personal_weights: Dict[int, List[np.array]] = {}
        self.federated_weights: Dict[int, List[np.array]] = {}

    def produce_weights(self, message: Message) -> Message:
        """
        Invoked by the ServerAgent to begin local model training, and return a message containing the weights/intercepts (to the server)
        :param message: Message containing information necessary to produce weights for the iteration
        :return: Message containing weights with security and/or DP noise added, as specified in config.py (to the server)
        """
        start_time = datetime.now()
        body = message.body
        iteration, lock, simulated_time = body['iteration'], body['lock'], body['simulated_time']
        directory: Directory = Directory.get_instance()

        # Retrieve the training dataset for the current iteration
        training_data = self.train_datasets[iteration]
        x_train, y_train = training_data
        # Extract the list of tuples (image,label) into separate numpy arrays
        # x_train, y_train = zip(*training_data)
        # x_train, y_train = np.array(x_train), np.array(y_train)


        # Perform local model training
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        weights = self.model.get_weights()

        # Update the client's local weights and intercepts at the given iteration/timestep
        self.personal_weights[iteration] = weights

        # Create copies of weights and intercepts since we may be adding to them
        # final_weights = copy.deepcopy(weights)

        end_time = datetime.now()
        # The time it took to perform local training once the client has received the message
        computation_time = end_time - start_time

        self.computation_times[iteration] = computation_time
        # The total time it takes to do the computation and send a message to the main server
        simulated_time += computation_time + directory.latency_dict[self.name][directory.main_server.name]

        body = {
            'weights': weights,
            'iteration': iteration,
            'computation_time': computation_time,
            'simulated_time': simulated_time
        }
        return Message(sender_name=self.name, recipient_name=directory.main_server.name, body=body)

    def receive_weights(self, message: Message) -> Message:
        """
        Called by server agent to receive federated weights (after an FL round)
        :param message: Message containing iteration, averaged_weights and return_intercepts (federatd weights and intercepts), and simulated_time
        :type message: Message
        :return: Message indicating whether client has converged in training this iteration, which only
        matters if config.CLIENT_DROPOUT is True.
        :rtype: Message
        """
        body = message.body
        iteration, averaged_weights, simulated_time = body['iteration'], body['averaged_weights'], body['simulated_time']
        directory: Directory = Directory.get_instance()

        # Federated weights
        averaged_weights = copy.deepcopy(averaged_weights)

        # Update the federated weights for the given iteration using the information stored in the message
        self.federated_weights[iteration] = averaged_weights

        # Obtain the client's local weights for the given iteration
        personal_weights = self.personal_weights[iteration]

        # Check for weight convergence
        # converged = self.satisfactory_weights(personal_weights, averaged_weights)

        # Compute personal accuracy and federated accuracy on test dataset
        personal_accuracy = self.evaluate_model(iteration,personal_weights)
        federated_accuracy = self.evaluate_model(iteration,averaged_weights)
        #personal_accuracy = self.evaluator.accuracy(personal_weights)
        #federated_accuracy = self.evaluator.accuracy(averaged_weights)

        # Update the client's accuracies
        self.personal_accuracy[iteration] = personal_accuracy
        self.federated_accuracy[iteration] = federated_accuracy

        args = [self.name, iteration, personal_accuracy, federated_accuracy]
        iteration_report = 'Performance Metrics for {} on iteration {} \n' \
                           '------------------------------------------- \n' \
                           'Personal accuracy: {} \n' \
                           'Federated accuracy: {} \n'

        # Simulate Latencies
        args.append(self.computation_times[iteration])
        iteration_report += 'Personal computation time: {} \n'
        args.append(simulated_time)
        iteration_report += 'Simulated time to receive federated weights: {} \n \n'

        if VERBOSITY:
            print(iteration_report.format(*args))

        main_server_name = directory.main_server.name
        body = {
            'simulated_time': simulated_time + directory.latency_dict[self.name][main_server_name]
        }
        return Message(sender_name=self.name, recipient_name=main_server_name, body=body)

    def evaluate_model(self, iteration, weights=None):
        x_train, y_train = self.train_datasets[iteration]

        #x_train, y_train = zip(*self.train_datasets[iteration])
        #x_train, y_train = np.array(x_train), np.array(y_train)

        loss, accuracy = self.model.evaluate(x_train, y_train, batch_size=10, verbose=0)
        return accuracy


def copy_model(model, weights=None) -> tf.keras.Model:
    model_ = tf.keras.Sequential().from_config(model.get_config())
    if weights is None:
        model_.set_weights(model.get_weights())
    else:
        model_.set_weights(weights)
    return model_

    # def satisfactory_weights(self, personal, federated) -> bool:
    #     # List of numpy booleans (checking whether the difference in weights is smaller than the tolerance level)
    #     diff_list = list(map(lambda x: np.abs(x[1] - x[0]) < config.tolerance, zip(personal, federated)))
    #     # Flatten the result into a numpy array
    #     diff_np = np.concatenate(diff_list, axis=None)
    #     return diff_np.all()