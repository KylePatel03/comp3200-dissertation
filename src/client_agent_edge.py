from datetime import datetime

from client_agent import ClientAgent
from directory import Directory
from edge_agent import EdgeAgent
from message import Message


class ClientAgentEdge(ClientAgent):

    """
        Invoked by the ServerAgent to do model training
        Client computes the forward pass on their training data and invoke the EdgeServer to produce the weight update
        :returns: Message to the EdgeServer to produce weight update (not necessarily from this Client)
    """
    def produce_weights(self, message: Message) -> Message:
        start_time = datetime.now()
        directory: Directory = Directory.get_instance()
        edge_server: EdgeAgent = EdgeAgent.get_instance()
        simulated_time = message.simulated_time

        # Retrieve the training dataset for the current iteration
        training_data = self.train_datasets[message.iteration]
        x_train, y_train = training_data

        # Compute the local model forward pass
        features = self.model.predict(x_train, batch_size=edge_server.batch_size)

        # Update times
        end_time = datetime.now()
        computation_time = end_time - start_time
        self.computation_times[message.iteration] = computation_time
        simulated_time += computation_time + directory.latency_dict[self.name][edge_server.name]

        msg = Message(sender_name=self.name,
                      recipient_name=edge_server.name,
                      iteration=message.iteration,
                      simulated_time=simulated_time,
                      body={
                          'features': features,
                          'labels': y_train,
                      }
                      )
        edge_server.receive_message(msg)
        return edge_server.produce_weights()

    def receive_weights(self, message: Message) -> Message:
        pass
