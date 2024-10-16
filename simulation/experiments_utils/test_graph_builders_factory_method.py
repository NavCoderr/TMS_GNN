from simulation.experiments_utils.test_graphs_builders import *


def createBuilder(builderName, creationParams):
    if builderName == 'FullGraphBuilder':
        return FullGraphBuilder(creationParams)
    if builderName == 'FullGraphBuilderLowServiceTime':
        return FullGraphBuilderLowServiceTime(creationParams)