import numpy as np

import backend

class Perceptron(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.get_data_and_monitor = backend.make_get_data_and_monitor_perceptron()


        #create and store a weight vector represented as a numpy array of zeros
        #get_weight will return this vector

        self.vec = np.zeros(dimensions) #creating a vector with the given dimension
        # self.storage = [] # creating an empty array for storing vectors
        # self.storage.append(self.vec) 



        "*** YOUR CODE HERE ***"

    def get_weights(self):
        """
        Return the current weights of the perceptron.

        Returns: a numpy array with D elements, where D is the value of the
            `dimensions` parameter passed to Perceptron.__init__
        """
        return self.vec

        "*** YOUR CODE HERE ***"

    def predict(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """

        dotproduct = np.dot(self.vec, x)
        print("dotproduct", dotproduct)

        if dotproduct >= 0:
            return 1
        else:
            return -1

        "*** YOUR CODE HERE ***"

    def update(self, x, y):
        """
        Update the weights of the perceptron based on a single example.
            x is a numpy array with D elements, where D is the value of the
                `dimensions`  parameter passed to Perceptron.__init__
            y is either 1 or -1

        Returns:
            True if the perceptron weights have changed, False otherwise
        """
        print(self.vec)
        if self.predict(x) == y: 
            return False
        else:
            self.vec += np.dot(y, x) 
            print("updated vector", self.vec)
            return True 


        "*** YOUR CODE HERE ***"

    def train(self):
        """
        Train the perceptron until convergence.

        To iterate through all of the data points once (a single epoch), you can
        do:
            for x, y in self.get_data_and_monitor(self):
                ...

        get_data_and_monitor yields data points one at a time. It also takes the
        perceptron as an argument so that it can monitor performance and display
        graphics in between yielding data points.
        """

        epochsConverged = False

        while epochsConverged == False:
            convergenceList = []

            for x, y in self.get_data_and_monitor(self):
                updates = self.update(x,y)
                
                if updates == False:
                    convergenceList.append(True)
                else:
                    convergenceList.append(False)

            if all(convergenceList) == True:
                epochsConverged = True






        "*** YOUR CODE HERE ***"
