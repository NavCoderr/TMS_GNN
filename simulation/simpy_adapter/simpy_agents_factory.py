from simulation.core.agents_factory import AgentsFactory
from simulation.simpy_adapter.agent import Agent


class SimpyAgentsFactory(AgentsFactory):
    def __init__(self, env):
        self.env = env
        self.counter = 0

    def createAgent(self, dependencies):
        self.counter += 1
        return Agent(env=self.env, number=self.counter, traverser=dependencies['traverser'])

