from datetime import timedelta
from client_agent import ClientAgent
from server_agent import ServerAgent
from directory import Directory
from data_formatting import *
from utils import print_config

import tensorflow as tf
from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D


class Initializer:
    def __init__(self,
                 num_clients,
                 client_fraction,
                 iterations,
                 epochs,
                 batch_size
                 ):
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(path=r'C:\Users\KyleP\Desktop\project\mnist')
        # Normalise values from [0,255] -> [0,1]
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Add channels dimension
        x_train, x_test = x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))

        self.num_clients: int = num_clients
        self.client_fraction: float = client_fraction
        self.iterations: int = iterations
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        # Partition the training dataset to each client
        client_datasets = partition_data_iid(x_train, y_train, k=self.num_clients, iterations=self.iterations)

        # Build model and initialise main server
        model = self.__build_model()
        self.main_server: ServerAgent = ServerAgent(0, model, (x_test, y_test))

        # Initialise client agents
        self.edges = {}
        ClientAgent.epochs = self.epochs
        ClientAgent.batch_size = self.batch_size
        self.clients = {
            i: ClientAgent(i, client_datasets[i], self.__copy_model(model))
            for i in range(num_clients)
        }
        client_names = list(map(lambda x: x.name, self.clients.values()))
        latency_dict = self.__init_latency_dict(client_names)

        # Initialise Directory
        self.directory: Directory = Directory(
            main_server=self.main_server,
            clients=self.clients,
            latency_dict=latency_dict,
            edges=self.edges
        )
        print_client_dataset(client_datasets)
        print_config()

    """
        Instantiate a latency-dict - a dictionary that stores the times for exchanging messages between two agents
    """
    def __init_latency_dict(self, client_names) -> Dict[str, Dict[str, timedelta]]:
        latency_dict = {}
        main_server_name = self.main_server.name
        latency_dict[main_server_name] = {}

        for client_name in client_names:
            # Client->Client
            latency_dict[client_name] = {
                client_name2: (timedelta(seconds=0) if client_name == client_name2 else timedelta(seconds=1))
                for client_name2 in client_names
            }
            # Client->Main Server
            latency_dict[client_name][main_server_name] = timedelta(seconds=1)
            # Main Server->Client
            latency_dict[main_server_name][client_name] = timedelta(seconds=0.69)
        return latency_dict


    def __build_model(self) -> tf.keras.Model:
        conv_model = Sequential([
            Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Flatten()
        ])
        # Pretrain the model...
        conv_model.trainable = False
        dense_model = Sequential([
            Dense(20, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model = Sequential([
            conv_model,
            dense_model
        ])
        return model

    """
        Clone the given model structure and weights
    """
    def __copy_model(self, model: tf.keras.Model) -> tf.keras.Model:
        model_ = tf.keras.Sequential().from_config(model.get_config())
        # model_ = tf.keras.models.clone_model(model)
        model_.set_weights(model.get_weights())
        return model_

    """
        Run FL on the main server 
    """
    def run_simulation(self,iterations):
        i = self.iterations
        if iterations >= 1 and iterations <= self.iterations:
            i = iterations
        main_server = self.directory.main_server
        main_server.main(num_clients=self.num_clients, num_iterations=i, client_fraction=self.client_fraction)
        # main_server.final_statistics()
