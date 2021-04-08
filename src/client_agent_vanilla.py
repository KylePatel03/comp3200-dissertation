from client_agent import *
from datetime import datetime
from directory import Directory


class ClientAgentVanilla(ClientAgent):
    # Initialised in Initialiser()
    batch_size: int
    epochs: int

    def __init__(self, agent_number, train_datasets, model):
        super().__init__(agent_number, train_datasets, model)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    """
    Invoked by the ServerAgent to begin local model training, and return a message containing the new_weights (to the server)
    :param message: Message containing information necessary to produce new_weights for the iteration
    :return: Message containing new_weights to the server
    """
    def produce_weights(self, message: Message) -> Message:
        start_time = datetime.now()
        simulated_time = message.simulated_time
        directory: Directory = Directory.get_instance()

        # Retrieve the training dataset for the current iteration
        training_data = self.train_datasets[message.iteration]
        x_train, y_train = training_data

        # Perform local model training
        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        weights = self.model.get_weights()
        # Update the client's local new_weights using the new learnt weights
        self.personal_weights[message.iteration] = weights

        end_time = datetime.now()
        computation_time = end_time - start_time
        self.computation_times[message.iteration] = computation_time
        # The total time it takes to do the computation and send a message to the main server
        simulated_time += computation_time + directory.latency_dict[self.name][directory.main_server.name]

        return Message(sender_name=self.name,
                       recipient_name=directory.main_server.name,
                       iteration=message.iteration,
                       simulated_time=simulated_time,
                       body={
                           'new_weights': weights,
                           'num_data': y_train.shape[0],
                       })

    """
        Invoked by the ServerAgent to receive/update local model with federated weights
        Client sends an acknowledgement back to the server
    """
    def receive_weights(self, message: Message) -> Message:
        start_time = datetime.now()
        averaged_weights = message.body['averaged_weights']
        simulated_time = message.simulated_time
        directory: Directory = Directory.get_instance()

        self.model.set_weights(averaged_weights)
        self.federated_weights[message.iteration] = averaged_weights

        end_time = datetime.now()
        computation_time = end_time - start_time
        self.computation_times[message.iteration] = computation_time
        simulated_time += directory.latency_dict[self.name][directory.main_server.name]

        return Message(sender_name=self.name,
                       recipient_name=directory.main_server.name,
                       iteration=message.iteration,
                       simulated_time=simulated_time,
                       body={})
