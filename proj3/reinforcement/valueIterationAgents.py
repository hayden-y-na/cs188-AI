# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        allStates = self.mdp.getStates()
        for count in range(self.iterations):
          oldDictionary = util.Counter()
          for state in allStates:
            if not self.mdp.isTerminal(state):
              bestAction = self.computeActionFromValues(state)
              qVal = self.computeQValueFromValues(state, bestAction)
              oldDictionary[state] = qVal
          self.values = oldDictionary.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        transition = self.mdp.getTransitionStatesAndProbs(state,action)
        if self.mdp.isTerminal(state) == False:
          for nextState, probability in transition:
            q_value += probability * (self.mdp.getReward(state,action,nextState) + (self.discount*self.getValue(nextState)))
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actionsList = self.mdp.getPossibleActions(state)
        
        #if it causes any problems, try to use isTerminal(self,state) from mdp
        if (actionsList is None):
          return None
        
        #savedMax = (value, action)
        savedMax = (-(float('inf')), None)
        for action in actionsList:
          qVal = self.computeQValueFromValues(state, action)
          if qVal >= savedMax[0]:
            savedMax = (qVal, action)
        
        return savedMax[1]
     


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        cureentStateNumber = 0
        currentState = None #setting the current state
        for state in allStates: #setting all values in each state as 0 
          self.values[state] = 0
          
        for iterator in range(self.iterations):
          cureentStateNumber = iterator%len(allStates) 
          currentState = allStates[cureentStateNumber]
          maxVal = -(float('inf'))
          
          if not self.mdp.isTerminal(currentState):
            bestAction = self.computeActionFromValues(currentState)
            qVal = self.computeQValueFromValues(currentState, bestAction)
            self.values[currentState] = qVal

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue()
        states = self.mdp.getStates()
        self.values = util.Counter()

        predecessors = dict()
        for state in states:
          predecessors[state] = set()

        for state in states:
          if self.mdp.isTerminal(state) == False:
            for action in self.mdp.getPossibleActions(state):
              for transitionSP in self.mdp.getTransitionStatesAndProbs(state, action):
                if transitionSP[1] > 0:
                  predecessors[transitionSP[0]].add(state)
            bestAction = self.computeActionFromValues(state)
            difference = abs(self.values[state] - self.computeQValueFromValues(state, bestAction))
            pq.push(state, -difference)

        for iteration in range(self.iterations):
          if pq.isEmpty() == False:
            state = pq.pop()
            bestAction = self.computeActionFromValues(state)
            self.values[state] = self.computeQValueFromValues(state, bestAction)

            for predecessor in predecessors[state]:
              bestAction = self.computeActionFromValues(predecessor)
              difference = abs(self.values[predecessor] - self.computeQValueFromValues(predecessor, bestAction))
              if difference > self.theta:
                pq.update(predecessor, -difference)





