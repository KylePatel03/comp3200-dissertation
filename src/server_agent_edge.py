from client_agent_edge import ClientAgentEdge
from server_agent import *


class ServerAgentEdge(ServerAgent):

    def fl_round(self, t: int, active_clients: Set[int]):
        server_logic_start = datetime.now()
        directory: Directory = Directory.get_instance()

        m = multiprocessing.Manager()

        # Send a message to each active client and invoke them to begin local model training
        with ThreadPool(len(active_clients)) as calling_pool:
            args = []
            for client_id in active_clients:
                # Create a message for each active client to begin training
                client_instance: ClientAgentEdge = directory.clients[client_id]
                client_name = client_instance.name

                arg = Message(
                    sender_name=self.name,
                    recipient_name=client_name,
                    iteration=t,
                    simulated_time=directory.latency_dict[self.name][client_name],
                    body={}
                )
                args.append((client_instance, arg))
            # Invoke ClientAgent.produce_weights() and store their returned message containing weight updates
            messages = calling_pool.map(lambda x: x[0].produce_weights(message=x[1]), args)

        # The time it takes for ALL clients to do model training and send their weights
        # This is identical to the simulated time for the LAST client to execute model training
        receive_weights_time = find_slowest_time(messages)
        print('{}: Received all of the selected clients weights at time {}'.format(self.name, receive_weights_time))

        # The edge weights weighted by the number of training datapoints
        edge_weights_nk = [map(lambda x: message.body['num_data'] * x, message.body['new_weights'])
                           for message in messages]
        # The total number of training datapoints
        nk_sum = sum([message.body['num_data'] for message in messages])

        # Aggregate (average) each of the clients new_weights, weighted by the number of local training datapoints
        averaged_edge_weights: Weights = list(map(lambda x: sum(x) / nk_sum, zip(*edge_weights_nk)))
        num_trainable_weights = len(self.model.trainable_weights)

        averaged_weights = self.model.get_weights()
        averaged_weights[-num_trainable_weights:] = averaged_edge_weights

        # Update the model's weights with the average weights
        self.model.set_weights(averaged_weights)
        # Set the averaged/federated new_weights and intercepts for the current timestep
        self.averaged_weights[t] = averaged_weights

        server_logic_end = datetime.now()
        server_logic_time = server_logic_end - server_logic_start
        # The total time taken to request for weights, perform aggregation and send averaged weights to EdgeServer
        simulated_time = receive_weights_time + server_logic_time + directory.latency_dict[self.name][
            directory.edge.name]

        print('{}: Simulated time to send EdgeServer the federated weights = {}'.format(self.name, simulated_time))

        message = Message(
            sender_name=self.name,
            recipient_name=directory.edge.name,
            iteration=t,
            simulated_time=simulated_time,
            body={
                'averaged_weights': averaged_edge_weights,
            }
        )
        # Invoke the EdgeServer to receive the message and receive an acknowledgement
        return_msg: Message = directory.edge.receive_weights(message)
        print('{}: Simulated time = {}'.format(self.name, return_msg.simulated_time))
