from abc import ABC

class Agent(ABC):
    """
    Attributes:
        agent_number: A number in the range [0,N-1] where N is the number of clients
        agent_type: String indicating the type => server_agent, edge_agent, client_agent
    """

    def __init__(self, agent_number, agent_type):
        self.agent_number: int = agent_number
        self.agent_type: str = agent_type

    @property
    def name(self):
        return str(self.agent_type) + str(self.agent_number)