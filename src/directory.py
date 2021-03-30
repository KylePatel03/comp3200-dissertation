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

    def __init__(self, main_server, clients,latency_dict,edges=None):
        if Directory.__directory is None:
            # ServerAgent
            self.main_server = main_server
            # Dict[int,ClientAgent]
            self.clients = clients
            # Dict[int,EdgeAgent]
            self.edges = edges
            # Dict[str,Dict[str,timedelta]]
            self.latency_dict = latency_dict
            Directory.__directory = self
        else:
            raise ValueError('The Directory instance is already initialised - access using get_instance()')

    @staticmethod
    def get_agent(self, agent_name: str):
        client_dict = {v.name: v for v in self.clients.values()}
        edge_dict = {v.name: v for v in self.edges.values()}
        main_server_dict = {self.main_server.name: self.main_server}

        temp_dict = dict(client_dict, **edge_dict, **main_server_dict)
        return temp_dict.get(agent_name)

    @staticmethod
    def get_edge_agent(self, agent_number: int):
        return self.edges.get(agent_number)