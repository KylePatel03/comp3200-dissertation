"""
    Singleton Class that stores the agents in the simulation
"""
class Directory:
    __directory = None

    @staticmethod
    def get_instance():
        if Directory.__directory is None:
            raise ValueError('Directory has not been initialised!')
        else:
            return Directory.__directory

    def __init__(self, main_server, clients, latency_dict, edge):
        if Directory.__directory is None:
            # ServerAgent
            self.main_server = main_server
            # Dict[int,ClientAgent]
            self.clients = clients
            # EdgeAgent
            self.edge = edge
            # Dict[str,Dict[str,timedelta]]
            self.latency_dict = latency_dict
            Directory.__directory = self
        else:
            raise ValueError('The Directory instance is already initialised - access using get_instance()')

    @staticmethod
    def get_agent(self, agent_name: str):
        if agent_name == self.edge.name:
            return self.edge
        elif agent_name == self.main_server.name:
            return self.main_server
        else:
            client_dict = {v.name: v for v in self.clients.values()}
            return client_dict.get(agent_name)