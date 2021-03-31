from datetime import timedelta
from client_agent import ClientAgent
from edge_agent import EdgeAgent
from server_agent import ServerAgent
from directory import Directory
from data_formatting import *

import tensorflow as tf
from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, InputLayer


class Initialiser:
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

        # Initialise the main FL parameters
        self.num_clients: int = num_clients
        self.client_fraction: float = client_fraction
        self.iterations: int = iterations
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        # Partition the training dataset to each client
        client_datasets = partition_data_iid(x_train, y_train, k=self.num_clients, iterations=self.iterations)

        # Build the compiled global model and initialise main server
        model = self.__build_model()
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

        # print('Main Model...')
        # for layer in model.layers:
        #     print(layer.name)
        #     print('Input = {}'.format(layer.input))
        #     print('Output = {}'.format(layer.output))
        #     print()
        #
        # print('Client Model...')
        # print(client_model.output,client_model.output_shape)

        # N = 100
        # x,y = x_train[:N], y_train[:N]
        # batch_size = self.batch_size
        #
        # print('Shapes of training dataset = {} {}'.format(x.shape, y.shape))
        #
        # feature_output = client_model.predict(x,batch_size=batch_size)
        # classification_output = edge_model(feature_output)
        #
        # edge_model.fit(feature_output,y, batch_size=batch_size)

        #print_client_dataset(client_datasets)

    """
        Instantiate a latency-dict - a dictionary that stores the times for exchanging messages between two agents
    """
    def __init_latency_dict(self, client_names) -> Dict[str, Dict[str, timedelta]]:
        latency_dict = {}
        main_server_name = self.main_server.name
        edge_server_name = self.edge_server.name
        latency_dict[main_server_name] = {}
        latency_dict[edge_server_name] = {}

        #Handle communication from ClientAgent to Agent (and vice versa)
        for client_name in client_names:
            # Client -> Client
            latency_dict[client_name] = {
                client_name2: (timedelta(seconds=0) if client_name == client_name2 else timedelta(seconds=1))
                for client_name2 in client_names
            }
            # Client -> MainServer
            latency_dict[client_name][main_server_name] = timedelta(seconds=3)

            # Client -> EdgeServer
            latency_dict[client_name][edge_server_name] = timedelta(seconds=0.8)

            # EdgeServer -> Client
            latency_dict[edge_server_name][client_name] = timedelta(seconds=0.8)

            # MainServer -> Client
            latency_dict[main_server_name][client_name] = timedelta(seconds=3)

        #MainServer -> EdgeServer
        latency_dict[main_server_name][edge_server_name] = timedelta(seconds=1)

        #EdgeServer -> MainServer
        latency_dict[edge_server_name][main_server_name] = latency_dict[main_server_name][edge_server_name]

        return latency_dict

    """
        Create and compile the main model and the partitioned client and edge model
    """
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
            InputLayer(input_shape=(conv_model.output_shape[1],)),
            Dense(20, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model = Sequential([
            conv_model,
            dense_model
        ])

        # Compile each model
        conv_model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
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
    def run_simulation(self,iterations):
        i = self.iterations
        if 1 <= iterations <= self.iterations:
            i = iterations
        main_server = self.directory.main_server
        self.__print_config(i)
        main_server.main(num_clients=self.num_clients, num_iterations=i, client_fraction=self.client_fraction)
        # main_server.final_statistics()

    def __print_config(self, iterations):
        main_msg = 'Running FL simulation with {} clients for {} rounds\n'.format(self.num_clients,iterations)
        params_msg = 'Hyperparameters...\nFraction of clients selected per round = {}\nLocal Batch Size = {}\nLocal ' \
                     'Epochs = {}'.format(self.client_fraction, self.batch_size, self.epochs)
        print(main_msg + params_msg + '\n')
