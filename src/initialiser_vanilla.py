from initialiser import *
from client_agent_vanilla import ClientAgentVanilla
from server_agent_vanilla import ServerAgentVanilla



class InitialiserVanilla(Initialiser):

    def __init__(self, num_clients, client_fraction, iterations, epochs, batch_size, accuracy_threshold):
        super().__init__(num_clients, client_fraction, iterations, epochs, batch_size, accuracy_threshold)

        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(path=self.mnist_path)
        # Normalise values from [0,255] -> [0,1]
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Add channels dimension
        x_train, x_test = x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))

        # Partition the training dataset to each client
        client_datasets = partition_data_iid(x_train, y_train, k=self.num_clients, iterations=self.iterations)

        model = self.build_model()

        # Initialise the main parameter server with the global model and validation dataset
        self.main_server: ServerAgent = ServerAgentVanilla(0, model, (x_test, y_test))

        # Initialise the batch_size and epochs for all clients
        ClientAgentVanilla.batch_size = self.batch_size
        ClientAgentVanilla.epochs = self.epochs

        # Initialise the clients with training dataset and a local model copy
        self.clients = {
            i: ClientAgentVanilla(i, client_datasets[i], self._copy_model(model))
            for i in range(num_clients)
        }
        client_names = list(map(lambda x: x.name, self.clients.values()))
        latency_dict = self._init_latency_dict(client_names, False)

        # Initialise Directory
        self.directory: Directory = Directory(
            main_server=self.main_server,
            clients=self.clients,
            latency_dict=latency_dict,
            edge=None,
        )

    """
        Create and compile the global model (vanilla system)
    """
    def build_model(self) -> tf.keras.Model:
        conv_model = self._build_conv_model()
        edge_model = self._build_dense_model()
        model = Sequential([
            conv_model,
            edge_model
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def run_simulation(self, iterations, type=''):
        return super().run_simulation(iterations, 'Vanilla')
