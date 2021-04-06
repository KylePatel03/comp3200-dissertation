import os
import re
from datetime import timedelta
from client_agent import ClientAgent
from edge_agent import EdgeAgent
from server_agent import ServerAgent
from directory import Directory
from data_formatting import *

import tensorflow as tf
from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D


class Initialiser:
    def __init__(self,
                 num_clients,
                 client_fraction,
                 iterations,
                 epochs,
                 batch_size,
                 accuracy_threshold
                 ):

        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(path=r'C:\Users\KyleP\Desktop\project\mnist')
        # Normalise values from [0,255] -> [0,1]
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Add channels dimension
        x_train, x_test = x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))

        # Initialise the main FL parameters
        self.num_clients: int = num_clients
        self.client_fraction: float = client_fraction
        self.iterations: int = iterations
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.accuracy_threshold: float = accuracy_threshold

        # Filepath of the saved convolution layer weights
        self.filename = 'pretrained-weights'

        # Partition the training dataset to each client
        client_datasets = partition_data_iid(x_train, y_train, k=self.num_clients, iterations=self.iterations)

        # Build the compiled global model
        model = self.__build_model(x_train, y_train)
        client_model, edge_model = model.layers[0], model.layers[1]

        # Initialise the main parameter server with the global model and validation dataset
        self.main_server: ServerAgent = ServerAgent(0, model, (x_test, y_test))

        # Initialise the edge server with its model
        self.edge_server: EdgeAgent = \
            EdgeAgent(0, edge_model, num_active_clients=self.num_clients,
                      epochs=self.epochs, batch_size=self.batch_size)

        # Initialise the clients with their dataset and compiled (pre-trained) model
        self.clients = {
            i: ClientAgent(i, client_datasets[i], client_model)
            for i in range(num_clients)
        }
        client_names = list(map(lambda x: x.name, self.clients.values()))
        latency_dict = self.__init_latency_dict(client_names)

        # Initialise Directory
        self.directory: Directory = Directory(
            main_server=self.main_server,
            clients=self.clients,
            latency_dict=latency_dict,
            edge=self.edge_server
        )
        # self.__print_latency_dict(latency_dict)
        # print_client_dataset(client_datasets)

    """
        Instantiate a latency-dict - a dictionary that stores the times for exchanging messages between two agents
        There is no account for Client->Client latency since they do not communicate directly amongst each other
    """
    def __init_latency_dict(self, client_names) -> Dict[str, Dict[str, timedelta]]:
        latency_dict = {}
        main_server_name, edge_server_name = self.main_server.name, self.edge_server.name
        latency_dict[main_server_name], latency_dict[edge_server_name] = {}, {}

        generator = np.random.default_rng(seed=0)
        # Handle communication from ClientAgent to Agent (and vice versa)
        for client_name in client_names:
            latency_dict.setdefault(client_name,{})
            # Client -> MainServer & vice versa
            latency_dict[client_name][main_server_name] = timedelta(seconds=generator.normal(loc=5.0, scale=2.0))
            latency_dict[main_server_name][client_name] = latency_dict[client_name][main_server_name]

            # Client -> EdgeServer & vice versa
            latency_dict[client_name][edge_server_name] = timedelta(seconds=generator.normal(loc=2.1, scale=1.0))
            latency_dict[edge_server_name][client_name] = latency_dict[client_name][edge_server_name]

        # MainServer -> EdgeServer & vice versa
        latency_dict[main_server_name][edge_server_name] = timedelta(seconds=2.04)
        latency_dict[edge_server_name][main_server_name] = latency_dict[main_server_name][edge_server_name]

        return latency_dict

    """
        Build the shared (pretrained) convolution model 
    """
    def __build_conv_model(self) -> tf.keras.Model:
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
        Pretrain the convolutional model
        :param train: Whether to perform model training or not - ignored when the model weights have not been saved
    """
    def __pretrain_conv_model(self, x_train, y_train, train=False) -> tf.keras.Model:
        conv_model = self.__build_conv_model()
        edge_model = Sequential([
            Dense(10, activation='softmax')
        ])
        model = Sequential([
            conv_model,
            edge_model
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        r = re.compile(self.filename)
        # Check if the model has been previously fit and has its weights stored in a file
        weights_saved = any(r.match(file) for file in os.listdir())

        # Pretrain the conv model if required
        if train or not weights_saved:
            print('Pretraining Conv Model...')

            # Training on digits [0,5)
            indices = np.where(y_train < 5)[0]
            x_train_lt5, y_train_lt5 = x_train[indices], y_train[indices]

            model.fit(x_train_lt5, y_train_lt5, batch_size=32, epochs=1, verbose=0)
            # Save the weights of the convolution layers to self.filename
            conv_model.save_weights(self.filename, save_format=None, overwrite=True)
        else:
            # Load the pretrained weights
            conv_model.load_weights(self.filename)
        # Freeze the model during training
        conv_model.trainable = False
        return conv_model


    """
        Create and compile the main model
    """
    def __build_model(self, x_train, y_train) -> tf.keras.Model:
        conv_model = self.__pretrain_conv_model(x_train, y_train)
        dense_model = Sequential([
            Dense(20, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model = Sequential([
            conv_model,
            dense_model
        ])

        # Compile the edge-side model and main model
        dense_model.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    """
        Run FL on the main server 
    """
    def run_simulation(self, iterations):
        i = self.iterations
        if 1 <= iterations <= self.iterations:
            i = iterations
        main_server: ServerAgent = self.directory.main_server
        self.__print_config(i)
        main_server.main(
            num_clients=self.num_clients,
            num_iterations=i,
            client_fraction=self.client_fraction,
            accuracy_threshold=self.accuracy_threshold
        )
        # main_server.final_statistics()

    def __print_latency_dict(self, latency_dict):
        main_server_name = self.main_server.name
        edge_server_name = self.edge_server.name

        print('Main Server...')
        for k, v in latency_dict[main_server_name].items():
            print('{} \t {}'.format(k, v))
        print()

        print('Edge Server...')
        for k, v in latency_dict[edge_server_name].items():
            print('{} \t {}'.format(k, v))
        print()

        clients_dict = {
            k: v for k, v in latency_dict.items()
            if k not in {main_server_name, edge_server_name}
        }
        print('Clients...')
        for k,v in clients_dict.items():
            print('{} \t {}'.format(k,v))

    def __print_config(self, iterations):
        main_msg = 'Running FL simulation with {} clients for {} rounds\n'.format(self.num_clients, iterations)
        params_msg = 'Hyperparameters...\nFraction of clients selected per round = {}\nLocal Batch Size = {}\nLocal ' \
                     'Epochs = {}'.format(self.client_fraction, self.batch_size, self.epochs)
        print(main_msg + params_msg + '\n')
