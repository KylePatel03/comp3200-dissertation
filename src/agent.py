from abc import ABC


class Agent(ABC):

    def __init__(self, agent_number, agent_type):
        # A number in the range [0,N) where N is the number of clients
        self.agent_number: int = agent_number
        # String indicating agent type => server, edge or client
        self.agent_type: str = agent_type

    @property
    def name(self):
        return str(self.agent_type) + str(self.agent_number)


# Simulate the time it takes to receive ALL messages (the time it takes to receive the last message)
def find_slowest_time(messages):
    return max(map(lambda m: m.simulated_time, messages))
