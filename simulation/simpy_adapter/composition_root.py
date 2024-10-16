from simulation.simpy_adapter.simpy_agents_factory import SimpyAgentsFactory
from simulation.simpy_adapter.environment_wrapper import EnvironmentWrapper


class CompositionRoot:
    def __init__(self, timeout):
        self.simulation = EnvironmentWrapper(timeout=timeout)
        self.simpyAgentsFactory = SimpyAgentsFactory(env=self.simulation.env)
