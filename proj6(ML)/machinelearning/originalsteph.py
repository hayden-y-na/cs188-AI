

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.1

        layers = 220

        self.w1 = nn.Variable(1, layers)
        self.w2 = nn.Variable(layers, layers)
        self.w3 = nn.Variable(layers, 1)
        self.b1 = nn.Variable(layers)
        self.b2 = nn.Variable(layers)
        self.b3 = nn.Variable(1)


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """

        graph = nn.Graph([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])


        input_x = nn.Input(graph, x)

        xw1 = nn.MatrixMultiply(graph, input_x, self.w1)
        xw1_plus_b1 = nn.MatrixVectorAdd(graph, xw1, self.b1)
        reluw1 = nn.ReLU(graph, xw1_plus_b1)

        reluw1_w2 = nn.MatrixMultiply(graph, reluw1, self.w2)
        reluw1_w2_plus_b2 = nn.MatrixVectorAdd(graph, reluw1_w2, self.b2)
        reluw2 = nn.ReLU(graph, reluw1_w2_plus_b2)

        reluw2_w3 = nn.MatrixMultiply(graph, reluw2, self.w3)
        reluw2_w3_plus_b3 = nn.MatrixVectorAdd(graph, reluw2_w3, self.b3)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.

            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, reluw2_w3_plus_b3, input_y)
            return graph

        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array

            return graph.get_output(reluw2_w3_plus_b3)
