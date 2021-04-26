from abc import ABC, abstractmethod
from datetime import timedelta

from server_agent import ServerAgent
from directory import Directory
from data_partition import *

import tensorflow as tf
from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

class Initialiser(ABC):

    mnist_path = r'mnist'
    pretrained_weights_path = r'resources/pretrained-weights'

    def __init__(self,
                 num_clients,
                 client_fraction,
                 iterations,
                 epochs,
                 batch_size,
                 accuracy_threshold,
                 ):
        # Initialise the main FL parameters
        self.iterations: int = iterations
        self.num_clients: int = num_clients
        self.client_fraction: float = client_fraction
        self.accuracy_threshold: float = accuracy_threshold
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        self.main_server = None
        self.edge_server = None
        self.clients = None
        self.directory = None


    """
        Instantiate a latency-dict - a dictionary that stores the times for exchanging messages between two agents
        There is no account for Client->Client latency since they do not communicate directly amongst each other
        param client_names: Collection of clients indexed by id
        :param edge_simulation: Whether there is an EdgeServer
    """
    def _init_latency_dict(self, client_names, edge_simulation):
        latency_dict = {}
        main_server_name = self.main_server.name
        latency_dict.setdefault(main_server_name, {})
        if edge_simulation:
            latency_dict.setdefault(self.edge_server.name, {})

        generator = np.random.default_rng(seed=0)
        # Handle communication between ClientAgent & Agents
        for client_name in client_names:
            latency_dict.setdefault(client_name, {})

            # Handle communication between ClientAgent & ServerAgent
            latency_dict[client_name][main_server_name] = timedelta(seconds=generator.normal(loc=5.0, scale=2.0))
            latency_dict[main_server_name][client_name] = latency_dict[client_name][main_server_name]

            # Handle communication between ClientAgent & EdgeServer
            if edge_simulation:
                edge_server_name = self.edge_server.name
                latency_dict[client_name][edge_server_name] = timedelta(seconds=generator.normal(loc=2.1, scale=1.0))
                latency_dict[edge_server_name][client_name] = latency_dict[client_name][edge_server_name]

        # Handle communication between ServerAgent & EdgeServer
        if edge_simulation:
            edge_server_name = self.edge_server.name
            # MainServer -> EdgeServer & vice versa
            latency_dict[main_server_name][edge_server_name] = timedelta(seconds=2.04)
            latency_dict[edge_server_name][main_server_name] = latency_dict[main_server_name][edge_server_name]

        return latency_dict

    """
        Create and compile the global model
    """
    @abstractmethod
    def build_model(self, **kwargs) -> tf.keras.Model:
        pass

    """
        Create the shared convolution model 
    """
    def _build_conv_model(self) -> tf.keras.Model:
        model = Sequential([
            Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Flatten()
        ])
        return model

    """
        Create the shared fully connected layers used in classification
    """
    def _build_dense_model(self) -> tf.keras.Model:
        model = Sequential([
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ])
        return model


    """
        Given a model, return a identical copy of the model architecture initialised with the same weights
    """
    def _copy_model(self, model: tf.keras.Model) -> tf.keras.Model:
        # Clone the model architecture
        model_copy = tf.keras.Sequential().from_config(model.get_config())
        # Copy the weights
        model_copy.set_weights(model.get_weights())
        return model_copy


    """
        Run FL on the main server 
    """
    def run_simulation(self, iterations, type=''):
        i = self.iterations
        if 1 <= iterations <= self.iterations:
            i = iterations
        main_server: ServerAgent = self.directory.main_server
        self._print_config(i, type)
        main_server.main(
            num_clients=self.num_clients,
            num_iterations=i,
            client_fraction=self.client_fraction,
            accuracy_threshold=self.accuracy_threshold
        )

    def _print_latency_dict(self, latency_dict, edge_simulation):
        main_server_name = self.main_server.name

        print('Main Server...')
        for k, v in latency_dict[main_server_name].items():
            print('{} \t {}'.format(k, v))
        print()

        clients_dict = {
            k: v for k,v in latency_dict.items()
            if k not in {main_server_name}
        }

        # Print EdgeServer latencies and ignore EdgeServer latencies when printing Client latencies
        if edge_simulation:
            print('Edge Server...')
            for k, v in latency_dict[self.edge_server.name].items():
                print('{} \t {}'.format(k, v))
            print()
            clients_dict = {
                k: v for k, v in clients_dict
                if k not in {self.edge_server.name}
            }

        print('Clients...')
        for k,v in clients_dict.items():
            print('{} \t {}'.format(k, v))

    """
        Print configuration details (hyperparameters + type of simulation - edge or vanilla)
    """
    def _print_config(self, iterations, type):
        main_msg = 'Running {} FL simulation with {} clients for {} rounds\n'.format(type, self.num_clients, iterations)
        params_msg = 'Hyperparameters...\nFraction of clients selected per round = {}\nLocal Batch Size = {}\nLocal ' \
                     'Epochs = {}'.format(self.client_fraction, self.batch_size, self.epochs)
        print(main_msg + params_msg + '\n')