class Message:

    def __init__(self, sender_name, recipient_name, iteration, simulated_time, body):
        self.sender: str = sender_name
        self.recipient: str = recipient_name
        self.iteration: int = iteration
        self.simulated_time = simulated_time
        # Dictionary mapping additional keys (new_weights etc. to their value)
        self.body = body

    def __str__(self):
        return 'Message from {} to {}.\nSimulated time = {}\n \n'.\
            format(self.sender, self.recipient, self.simulated_time)
