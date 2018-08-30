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
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.001
        self.graph = None
 
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
        "*** YOUR CODE HERE ***"
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            w1 = nn.Variable(len(x)//2,len(x)//2)
            w2 = nn.Variable(len(x)//2,len(x)//2)
            w3 = nn.Variable(len(x)//2,len(x)//2)
            w4 = nn.Variable(len(x)//2,len(x)//2)
            b1 = nn.Variable(len(x)//2,1)
            b2 = nn.Variable(len(x)//2,1)
            b3 = nn.Variable(len(x)//2,1)
            b4 = nn.Variable(len(x)//2,1)
            # w1 = nn.Variable(len(x),len(x))
            # w2 = nn.Variable(len(x),len(x))
            # w3 = nn.Variable(len(x),len(x))
            # w4 = nn.Variable(len(x),len(x))
            # b1 = nn.Variable(len(x),1)
            # b2 = nn.Variable(len(x),1)
            # b3 = nn.Variable(len(x),1)
            # b4 = nn.Variable(len(x),1)
            self.graph = nn.Graph([w1,w2,w3,w4,b1,b2,b3,b4])
            input_x = nn.Input(self.graph,x)
            input_y = nn.Input(self.graph,y)
            left_x = nn.Input(self.graph,x[:len(x)//2])
            right_x = nn.Input(self.graph,x[len(x)//2:])
            # left_x = nn.Input(self.graph,x)
            # right_x = nn.Input(self.graph,y)
            mult1 = nn.MatrixMultiply(self.graph, w1, left_x)
            mult2 = nn.MatrixMultiply(self.graph, w2, right_x)
            add1 = nn.MatrixVectorAdd(self.graph, mult1, mult2)
            add2 = nn.MatrixVectorAdd(self.graph, mult2, mult1)
            add3 = nn.MatrixVectorAdd(self.graph, add1, b1)
            add4 = nn.MatrixVectorAdd(self.graph, add2, b2)
            relu1 = nn.ReLU(self.graph, add3)
            relu2 = nn.ReLU(self.graph, add4)
            mult3 = nn.MatrixMultiply(self.graph, w3, relu1)
            mult4 = nn.MatrixMultiply(self.graph, w4, relu2)
            add5 = nn.MatrixVectorAdd(self.graph, mult3, mult4)
            add6 = nn.MatrixVectorAdd(self.graph, mult4, mult3)
            add7 = nn.MatrixVectorAdd(self.graph, add5, b3)
            add8 = nn.MatrixVectorAdd(self.graph, add6, b4)
            left_y = nn.Input(self.graph, y[:len(x)//2])
            right_y = nn.Input(self.graph, y[len(x)//2:])
            # left_y = nn.Input(self.graph,x)
            # right_y = nn.Input(self.graph,y)
            loss1 = nn.SquareLoss(self.graph, add7, left_y)
            loss2 = nn.SquareLoss(self.graph, add8, right_y)
            add9 = nn.Add(self.graph, loss1, loss2)
            return self.graph
 
            #attempt 1
            # w1 = nn.Variable(len(x),len(x))
            # w2 = nn.Variable(len(x),len(x))
            # w3 = nn.Variable(len(x),len(x))
            # w4 = nn.Variable(len(x),len(x))
            # b1 = nn.Variable(len(x),1)
            # b2 = nn.Variable(len(x),1)
            # b3 = nn.Variable(len(x),1)
            # b4 = nn.Variable(len(x),1)
            # self.graph = nn.Graph([w1,w2,w3,w4,b1,b2,b3,b4])
            # input_x = nn.Input(self.graph,x)
            # input_y = nn.Input(self.graph,y)
            # mult1 = nn.MatrixMultiply(self.graph, w1, input_x)
            # add1 = nn.MatrixVectorAdd(self.graph, mult1, b1)
            # relu1 = nn.ReLU(self.graph, add1)
            # mult2 = nn.MatrixMultiply(self.graph, w2, relu1)
            # add2 = nn.MatrixVectorAdd(self.graph, mult2, b2)
            # relu2 = nn.ReLU(self.graph, add2)
            # mult3 = nn.MatrixMultiply(self.graph, w3, relu2)
            # add3 = nn.MatrixVectorAdd(self.graph, mult3, b3)
            # relu3 = nn.ReLU(self.graph, add3)
            # mult4 = nn.MatrixMultiply(self.graph, w3, relu3)
            # add4 = nn.MatrixVectorAdd(self.graph, mult4, b4)
            # loss = nn.SquareLoss(self.graph, add4, input_y)
            # return self.graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
 
            top_vec = self.graph.get_output(self.graph.get_nodes()[-4])
            bot_vec = self.graph.get_output(self.graph.get_nodes()[-5])
            # print(top_vec,bot_vec)
            return np.concatenate((top_vec, bot_vec), axis=0)
 
            # top_add = self.graph.get_output(self.graph.get_nodes()[-4])
            # bot_add = self.graph.get_output(self.graph.get_nodes()[-5])
            # return (top_add + bot_add) * (0.5)
 
            # return self.graph.get_output(self.graph.get_nodes()[-2])