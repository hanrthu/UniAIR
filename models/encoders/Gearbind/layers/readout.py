from torch import nn
from torch_scatter import scatter_add, scatter_mean


class Readout(nn.Module):

    def __init__(self, type="node"):
        super(Readout, self).__init__()
        self.type = type

    def get_index2graph(self, graph):
        if self.type == "node":
            input2graph = graph.node2graph
        elif self.type == "edge":
            input2graph = graph.edge2graph
        elif self.type == "residue":
            input2graph = graph.residue2graph
        else:
            raise ValueError("Unknown input type `%s` for readout functions" % self.type)
        return input2graph

class MeanReadout(Readout):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        output = scatter_mean(input, input2graph, dim=0, dim_size=graph.batch_size)
        return output


class SumReadout(Readout):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, graph, input):
        """
        Perform readout over the graph(s).

        Parameters:
            graph (Graph): graph(s)
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        input2graph = self.get_index2graph(graph)
        output = scatter_add(input, input2graph, dim=0, dim_size=graph.batch_size)
        return output