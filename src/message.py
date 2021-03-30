from typing import Dict, Any

class Message:
    """
    Used for all client-client and client-server communications
    """
    def __init__(self, sender_name, recipient_name, body):
        """
        :param sender_name: String name of sender
        :param recipient_name: String name of recipient
        :param body: Content depends no the message being sent.
        """
        self.sender: str = sender_name
        self.recipient: str = recipient_name
        # Dictionary mapping keys (iteration, weights, simulated_time etc. to their value)
        self.body: Dict[str, Any] = body

    def __str__(self):
        return "Message from {self.sender} to {self.recipient}.\n Body is : {self.body} \n \n"
