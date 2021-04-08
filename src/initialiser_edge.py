import os

from initialiser import *
from edge_agent import EdgeAgent
from client_agent_edge import ClientAgentEdge
from server_agent_edge import ServerAgentEdge


class InitialiserEdge(Initialiser):

    def __init__(self,
                 num_clients,
                 client_fraction,
                 iterations,
                 epochs,
                 batch_size,
                 accuracy_threshold):
        super().__init__(num_clients, client_fraction, iterations, epochs, batch_size, accuracy_threshold)

        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(path=self.mnist_path)
        # Normalise values from [0,255] -> [0,1]
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Add channels dimension
        x_train, x_test = x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))

        # Partition the training dataset to each client
        client_datasets = partition_data_iid(x_train, y_train, k=self.num_clients, iterations=self.iterations)

        # The pretrained_weights_path of the saved weights for the convolutional layers
        self.pretrained_weights_path = self.pretrained_weights_path

        # Build the compiled global model
        model = self.build_model(x_train, y_train)
        client_model, edge_model = model.layers[0], model.layers[1]

        # Initialise the main parameter server with the global model and validation dataset
        self.main_server: ServerAgent = ServerAgentEdge(0, model, (x_test, y_test))

        # Initialise the edge server with its model
        self.edge_server: EdgeAgent = \
            EdgeAgent(0, edge_model, num_active_clients=self.num_clients,
                      epochs=self.epochs, batch_size=self.batch_size)

        # Initialise the clients with their dataset and compiled (pre-trained) model
        self.clients = {
            i: ClientAgentEdge(i, client_datasets[i], client_model)
            for i in range(num_clients)
        }
        client_names = list(map(lambda x: x.name, self.clients.values()))
        latency_dict = self._init_latency_dict(client_names, True)

        # Initialise Directory
        self.directory: Directory = Directory(
            main_server=self.main_server,
            clients=self.clients,
            latency_dict=latency_dict,
            edge=self.edge_server
        )

    """
        Pretrain the convolutional model
        :param train: Whether to perform model training or not - ignored when the model weights have not been saved
    """
    def __pretrain_conv_model(self, x_train, y_train, train=False) -> tf.keras.Model:
        conv_model = self._build_conv_model()
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

        file_path = self.pretrained_weights_path + '.index'
        weights_saved = os.path.exists(file_path)

        # Pretrain the conv model if required
        if train or not weights_saved:
            print('Pretraining Conv Model...')

            # Training on digits [0,5)
            indices = np.where(y_train < 5)[0]
            x_train_lt5, y_train_lt5 = x_train[indices], y_train[indices]

            model.fit(x_train_lt5, y_train_lt5, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
            # Save the weights of the convolution layers to self.pretrained_weights_path
            conv_model.save_weights(self.pretrained_weights_path, save_format=None, overwrite=True)
        else:
            # Load the pretrained weights
            conv_model.load_weights(self.pretrained_weights_path)
        # Freeze the model during training
        conv_model.trainable = False
        return conv_model

    """
        Create and compile the global model
    """
    def build_model(self, x_train, y_train) -> tf.keras.Model:
        conv_model = self.__pretrain_conv_model(x_train, y_train)
        dense_model = self._build_dense_model()
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

    def run_simulation(self, iterations, type=''):
        return super().run_simulation(iterations, 'Edge-Based')
