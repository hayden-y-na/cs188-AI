import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

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
        self.learning_rate = 0.1

        hidden_layers = 100


        self.weight1 = nn.Variable(1, hidden_layers)
        self.weight2 = nn.Variable(hidden_layers, hidden_layers)
        self.weight3 = nn.Variable(hidden_layers, hidden_layers)
        self.weight4 = nn.Variable(hidden_layers, hidden_layers)
        self.weight5 = nn.Variable(hidden_layers, 1)
        self.bias1 = nn.Variable(hidden_layers)
        self.bias2 = nn.Variable(hidden_layers)
        self.bias3 = nn.Variable(hidden_layers)
        self.bias4 = nn.Variable(hidden_layers)
        self.bias5 = nn.Variable(1)




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
        graph = nn.Graph([self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3, self.weight4, self.bias4, self.weight5, self.bias5])

        inputx = nn.Input(graph, x)

        multi1 = nn.MatrixMultiply(graph, inputx, self.weight1)
        biased_added_multi1 = nn.MatrixVectorAdd(graph, multi1, self.bias1)
        relu1 = nn.ReLU(graph, biased_added_multi1)

        multi2 = nn.MatrixMultiply(graph, relu1, self.weight2)
        biased_added_multi2 = nn.MatrixVectorAdd(graph, multi2, self.bias2)
        relu2 = nn.ReLU(graph, biased_added_multi2)

        multi3 = nn.MatrixMultiply(graph, relu2, self.weight3)
        biased_added_multi3 = nn.MatrixVectorAdd(graph, multi3, self.bias3)
        relu3 = nn.ReLU(graph, biased_added_multi3)

        multi4 = nn.MatrixMultiply(graph, relu3, self.weight4)
        biased_added_multi4 = nn.MatrixVectorAdd(graph, multi4, self.bias4)
        relu4 = nn.ReLU(graph, biased_added_multi4)

        multi5 = nn.MatrixMultiply(graph, relu4, self.weight5)
        biased_added_multi5 = nn.MatrixVectorAdd(graph, multi5, self.bias5)






        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            inputy = nn.Input(graph,y)
            loss_node = nn.SquareLoss(graph, biased_added_multi5, inputy)

            return graph
        else:
            return graph.get_output(biased_added_multi5)

            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"

class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1

        hidden_layers = 100


        self.weight1 = nn.Variable(1, hidden_layers)
        self.weight2 = nn.Variable(hidden_layers, hidden_layers)
        self.weight3 = nn.Variable(hidden_layers, hidden_layers)
        self.weight4 = nn.Variable(hidden_layers, hidden_layers)
        self.weight5 = nn.Variable(hidden_layers, 1)
        self.bias1 = nn.Variable(hidden_layers)
        self.bias2 = nn.Variable(hidden_layers)
        self.bias3 = nn.Variable(hidden_layers)
        self.bias4 = nn.Variable(hidden_layers)
        self.bias5 = nn.Variable(1)


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

        graph = nn.Graph([self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3, self.weight4, self.bias4,
             self.weight5, self.bias5])

        inputx = nn.Input(graph, x)

        multi1 = nn.MatrixMultiply(graph, inputx, self.weight1)
        biased_added_multi1 = nn.MatrixVectorAdd(graph, multi1, self.bias1)
        relu1 = nn.ReLU(graph, biased_added_multi1)

        multi2 = nn.MatrixMultiply(graph, relu1, self.weight2)
        biased_added_multi2 = nn.MatrixVectorAdd(graph, multi2, self.bias2)
        relu2 = nn.ReLU(graph, biased_added_multi2)

        multi3 = nn.MatrixMultiply(graph, relu2, self.weight3)
        biased_added_multi3 = nn.MatrixVectorAdd(graph, multi3, self.bias3)
        relu3 = nn.ReLU(graph, biased_added_multi3)

        multi4 = nn.MatrixMultiply(graph, relu3, self.weight4)
        biased_added_multi4 = nn.MatrixVectorAdd(graph, multi4, self.bias4)
        relu4 = nn.ReLU(graph, biased_added_multi4)

        multi5 = nn.MatrixMultiply(graph, relu4, self.weight5)
        fx = nn.MatrixVectorAdd(graph, multi5, self.bias5)


        #negative f(-x)
        inputx = nn.Input(graph, -1*x)

        multi1 = nn.MatrixMultiply(graph, inputx, self.weight1)
        biased_added_multi1 = nn.MatrixVectorAdd(graph, multi1, self.bias1)
        relu1 = nn.ReLU(graph, biased_added_multi1)

        multi2 = nn.MatrixMultiply(graph, relu1, self.weight2)
        biased_added_multi2 = nn.MatrixVectorAdd(graph, multi2, self.bias2)
        relu2 = nn.ReLU(graph, biased_added_multi2)

        multi3 = nn.MatrixMultiply(graph, relu2, self.weight3)
        biased_added_multi3 = nn.MatrixVectorAdd(graph, multi3, self.bias3)
        relu3 = nn.ReLU(graph, biased_added_multi3)

        multi4 = nn.MatrixMultiply(graph, relu3, self.weight4)
        biased_added_multi4 = nn.MatrixVectorAdd(graph, multi4, self.bias4)
        relu4 = nn.ReLU(graph, biased_added_multi4)

        multi5 = nn.MatrixMultiply(graph, relu4, self.weight5)
        negfx = nn.MatrixVectorAdd(graph, multi5, self.bias5)

        identityMatrix = np.identity(np.shape(graph.get_output(negfx))[1])
        negidentityMatrix = np.negative(identityMatrix)
        negfx = nn.MatrixMultiply(graph,negfx, nn.Input(graph, negidentityMatrix))
        finalOutput = nn.Add(graph, fx, negfx)

        # self.weight1 = nn.Variable(1, hidden_layers)
        # self.weight2 = nn.Variable(hidden_layers, 1)
        #
        # self.bias1 = nn.Variable(hidden_layers)
        # self.bias2 = nn.Variable(1)
        #
        # graph = nn.Graph([self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3, self.weight4, self.bias4, self.weight5, self.bias5])
        #
        #
        #
        #
        # inputx = nn.Input(graph, x)
        # neginputx = nn.MatrixMultiply(graph, inputx, np.array([[-1]]))
        # multi1 = nn.MatrixMultiply(graph, neginputx, self.weight1)
        # biased_added_multi1 = nn.MatrixVectorAdd(graph, multi1, self.bias1)
        # relu1 = nn.ReLU(graph, biased_added_multi1)
        #
        # multi2 = nn.MatrixMultiply(graph, relu1, self.weight2)
        # biased_added_multi2 = nn.MatrixVectorAdd(graph, multi2, self.bias2)
        #
        # neg_biased_added_multi2 = nn.MatrixMultiply(graph, biased_added_multi2, np.array([[-1]]))
        # subtracted = nn.Add(graph, )


        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            inputy = nn.Input(graph, y)
            loss_node = nn.SquareLoss(graph, finalOutput, inputy)
            return graph

        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            output = graph.get_output(finalOutput)
            return output


class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "* YOUR CODE HERE *"
        self.learning_rate = 0.20

        layers = 250

        self.weight1 = nn.Variable(784, layers)
        self.weight2 = nn.Variable(layers, layers)
        self.weight3 = nn.Variable(layers, layers)
        self.weight4 = nn.Variable(layers, layers)
        self.weight5 = nn.Variable(layers, 10)

        self.bias1 = nn.Variable(layers)
        self.bias2 = nn.Variable(layers)
        self.bias3 = nn.Variable(layers)
        self.bias4 = nn.Variable(layers)
        self.bias5 = nn.Variable(10)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, y is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use nn.SoftmaxLoss as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "* YOUR CODE HERE *"
        graph = nn.Graph(
            [self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3, self.weight4, self.bias4,
             self.weight5, self.bias5])

        input_x = nn.Input(graph, x)

        xw1 = nn.MatrixMultiply(graph, input_x, self.weight1)
        plus1b1 = nn.MatrixVectorAdd(graph, xw1, self.bias1)
        relu1 = nn.ReLU(graph, plus1b1)

        relu1_2 = nn.MatrixMultiply(graph, relu1, self.weight2)
        plus2b2 = nn.MatrixVectorAdd(graph, relu1_2, self.bias2)
        relu2 = nn.ReLU(graph, plus2b2)

        relu2_3 = nn.MatrixMultiply(graph, relu2, self.weight3)
        plus3b3 = nn.MatrixVectorAdd(graph, relu2_3, self.bias3)
        relu3 = nn.ReLU(graph, plus3b3)

        relu3_4 = nn.MatrixMultiply(graph, relu3, self.weight4)
        plus4b4 = nn.MatrixVectorAdd(graph, relu3_4, self.bias4)
        relu4 = nn.ReLU(graph, plus4b4)

        relu4_5 = nn.MatrixMultiply(graph, relu4, self.weight5)
        plus5b5 = nn.MatrixVectorAdd(graph, relu4_5, self.bias5)

        if y is not None:
            "* YOUR CODE HERE *"
            input_y = nn.Input(graph, y)
            loss = nn.SoftmaxLoss(graph, plus5b5, input_y)
            return graph
        else:
            "* YOUR CODE HERE *"
            return graph.get_output(plus5b5)

class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.hidden_layers = 100
        self.weight1 = nn.Variable(4, self.hidden_layers)
        self.weight2 = nn.Variable(self.hidden_layers, self.hidden_layers)
        self.weight3 = nn.Variable(self.hidden_layers, 2)

        self.bias1 = nn.Variable(self.hidden_layers)
        self.bias2 = nn.Variable(self.hidden_layers)
        self.bias3 = nn.Variable(2)




    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3])
        inputs = nn.Input(graph, states)

        multi1 = nn.MatrixMultiply(graph, inputs, self.weight1)
        biased_added_multi1 = nn.MatrixVectorAdd(graph, multi1, self.bias1)
        relu1 = nn.ReLU(graph, biased_added_multi1)

        multi2 = nn.MatrixMultiply(graph, relu1, self.weight2)
        biased_added_multi2 = nn.MatrixVectorAdd(graph, multi2, self.bias2)
        relu2 = nn.ReLU(graph, biased_added_multi2)

        multi3 = nn.MatrixMultiply(graph, relu2, self.weight3)
        biased_added_multi3 = nn.MatrixVectorAdd(graph, multi3, self.bias3)
        relu3 = nn.ReLU(graph, biased_added_multi3)



        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            Q_targetInput = nn.Input(graph, Q_target)
            loss_node = nn.SquareLoss(graph, Q_targetInput, relu3)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            output = graph.get_output(graph.get_nodes()[-2])
            return output

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "* YOUR CODE HERE *"
        self.learning_rate = .05

        values = 300

        self.weight1 = nn.Variable(values, values)
        self.weight2 = nn.Variable(values, values)
        self.weight3 = nn.Variable(values, values)
        self.weight4 = nn.Variable(values, values)
        self.weight5 = nn.Variable(values, 5)

        self.bias1 = nn.Variable(values)

        self.hthing = nn.Variable(values)

        self.wthing = nn.Variable(self.num_chars, values)
        self.whthing = nn.Variable(values, values)

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here xs will be a list of length L. Each element of xs will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, y is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        xs into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use nn.SoftmaxLoss as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "* YOUR CODE HERE *"
        #create graph structure
        graph = nn.Graph([self.weight1, self.weight2, self.weight3, self.weight4, self.weight5, self.bias1, self.wthing, self.whthing, self.hthing])

        #use array of zeros of (batch size, d)
        hthing2 = nn.MatrixVectorAdd(graph, nn.Input(graph, np.zeros((batch_size, 300))), self.hthing)

        #loop through elements of xs
        #instantiate input_y below if y is not a null value
        for thing in xs:
            inpx = nn.Input(graph, thing)

            bthing = nn.Add(graph, nn.MatrixMultiply(graph, hthing2, self.whthing), nn.MatrixMultiply(graph, inpx, self.wthing))
            hthing2 = nn.ReLU(graph, bthing)

        #should have at least one ReLU, hidden size should be sufficiently large
        #make sure network isn't too deep and learning rate isn't too high (~.01 or .1 is good)
        m1 = nn.MatrixMultiply(graph, hthing2, self.weight1)
        a1 = nn.MatrixVectorAdd(graph, m1, self.bias1)
        r1 = nn.ReLU(graph, a1)
        m2 = nn.MatrixMultiply(graph, r1, self.weight2)
        r2 = nn.ReLU(graph, m2)
        m3 = nn.MatrixMultiply(graph, r2, self.weight3)
        r3 = nn.ReLU(graph, m3)
        m4 = nn.MatrixMultiply(graph, r3, self.weight4)
        r4 = nn.ReLU(graph, m4)
        m5 = nn.MatrixMultiply(graph, r4, self.weight5)


        if y is not None:
            "* YOUR CODE HERE *"
            input_y = nn.Input(graph, y)

            #use softmaxloss not squareloss
            loss = nn.SoftmaxLoss(graph, m5, input_y)

            return graph

        else:
            "* YOUR CODE HERE *"
            return graph.get_output(m5)