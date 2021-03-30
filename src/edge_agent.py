import config
from agent import Agent
from message import Message


class EdgeAgent(Agent):
    """
    :type client: ClientAgent
    """
    def __init__(self, agent_number, model):
        super(EdgeAgent, self).__init__(agent_number=agent_number, agent_type='edge_agent')
        self.__model = model

    """
     Receives features from client and performs local model training
     :param message: Message received from its client, containing the features learnt
     :return: Return a message to the client containing the updated weights
     """
    def receive_weights(self, message: Message) -> Message:
        # Extract message contents
        features, labels = message.body['features'], message.body['labels']

        # Perform local model DNN training
        self.__model.fit(features,labels,batch_size=config.BATCH_SIZE,epochs=config.EPOCHS)
        weights = self.__model.get_weights()

        body = {
            'weights': weights,
            'simulated_time': None
        }
        return_message = Message(sender_name=self.name, recipient_name=message.sender, body=body)
        return return_message
