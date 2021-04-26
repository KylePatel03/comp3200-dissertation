from server_agent import *


class ServerAgentVanilla(ServerAgent):

    def fl_round(self, t, active_clients):
        server_logic_start = datetime.now()
        directory: Directory = Directory.get_instance()

        m = multiprocessing.Manager()

        # Send a message to each active client and invoke them to begin local model training
        with ThreadPool(len(active_clients)) as calling_pool:
            args = []
            for client_id in active_clients:
                # Create a message for each active client to begin training
                client_instance: ClientAgent = directory.clients[client_id]
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

        # The time it takes for all clients to perform training
        receive_weights_time = find_slowest_time(messages)
        #print('{}: Received all of the selected clients weights at time {}'.format(self.name, receive_weights_time))

        # The weights weighted by the number of training datapoints
        client_weights_nk = [map(lambda x: message.body['num_data'] * x, message.body['new_weights'])
                             for message in messages]
        # The total number of training datapoints
        nk_sum = sum([message.body['num_data'] for message in messages])

        # Aggregate (average) each of the clients new_weights, weighted by the number of local training datapoints
        averaged_weights = list(map(lambda x: sum(x) / nk_sum, zip(*client_weights_nk)))

        # Update the global model weights
        self.model.set_weights(averaged_weights)
        self.averaged_weights[t] = averaged_weights

        server_logic_end = datetime.now()
        server_logic_time = server_logic_end - server_logic_start
        # The total time taken to request for weights & perform aggregation
        simulated_time = receive_weights_time + server_logic_time

        # Send a message containing the averaged new_weights etc. to each active client
        with ThreadPool(len(active_clients)) as returning_pool:
            args = []
            for client_id in active_clients:
                client_instance: ClientAgent = directory.clients[client_id]
                client_name = client_instance.name
                message = Message(sender_name=self.name,
                                  recipient_name=client_name,
                                  iteration=t,
                                  simulated_time=simulated_time+directory.latency_dict[self.name][client_name],
                                  body={
                                      'averaged_weights': averaged_weights,
                                  })
                args.append((client_instance, message))
            # Invokes the client to receive the message => calls ClientAgent.receive_weights()
            return_messages = returning_pool.map(lambda x: x[0].receive_weights(message=x[1]), args)

        # The total simulated time
        simulated_time = find_slowest_time(return_messages)
        print('{}: Simulated time = {}'.format(self.name, simulated_time))

