# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        foodList = successorGameState.getFood().asList()
        
        for foodPosition in foodList:
          foodSocresList = []
          foodSocresList.append(manhattanDistance(newPos, foodPosition))
          foodNearst = min(foodSocresList)
          foodScores = 10/foodNearst
          
          
          
        ghostStates = []
        for ghostState in newGhostStates:
          ghostStates.append(ghostState.getPosition())
        
        if ghostStates: 
          ghostDistances = []
          for ghostCoordinate in ghostStates:
            ghostDistances.append(manhattanDistance(newPos, ghostCoordinate))
            ghostNearst = min(ghostDistances)
            
            if ghostNearst == 0:
              return successorGameState.isLose()
            else:
              successorGameState.isWin()
              
        heuristic = successorGameState.getScore() + foodScores/ghostNearst

        #ghostDistance = []
      
            #ghostDistance.append(newPos, )
            #ghostClosest = min()
        
          
          
        


        "*** YOUR CODE HERE ***"
        return heuristic
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

#    def maxAgent(state, depth, alpha, beta):
#      
#      if state.isWin(): #base case1
#        return self.evaluationFunction(state)
#      if state.isLose(): #base case2
#        return self.evaluationFunction(state)
#      if depth == self.depth: #base case3
#        return self.evaluationFunction(state)
#      
#      score = -9999999 #setting a huge negative value for comparing reason
#      for move in legalMoves:
#        agentState = state.generateSuccessor(0,move)
#        score = max(score, minAgent(agentState,depth,1,alpha,beta)) #getting the max score comparing with the minimum scores from its successor
#        if score > beta:
#          alpha = max(alpha, score)
#        else:
#          return score
#          
#        return score
#      
#      def minAgent(state,depth,agentIndex,alpha,beta):
#        
#        if state.isWin(): #base case1
#          return self.evaluationFunction(state)
#        if state.isLose(): #base case2
#          return self.evaluationFunction(state)
#        if depth == self.depth: #base case3
#          return self.evaluationFunction(state) 
#          
#        score = "9999999" #setting score as a huge positive value to compare below
#        
#        legalMoves = state.getLegalActions()
#        totalNumberAgents = state.getNumAgents()
#        
#        if agentIndex == totalNumberAgents-1: #when there is only one ghost left
#          for move in legalMoves:
#            agentState = state.generateSuccessor(agentIndex,move) 
#            score = min(score, maxAgent(agentState, depth+1, alpha, beta)) #get the max score for pacman
#            
#            if score > alpha:
#              beta = min(score, beta) #setting beta as the minimum value
#            else:
#              return score # no need to check its successors
#        else: # when there are more than 1 ghosts
#          for move in legalMoves:
#            agentState = state.generateSucessor(agentIndex,move) 
#            score = min(score, minAgent(agentState, depth, agentIndex+1, alpha, beta))
#            
#            if score > alpha:
#              beta = min(score,beta)
#            else:
#              return score
#        return score
#    
#             
#  
#
#    def getAction(self, gameState):
#        """
#          Returns the minimax action using self.depth and self.evaluationFunction
#        """
#        
#        
#        legalMoves = gameState.getLegalActions() #get all possible states in list
#        currentMove = Directions.STOP #set current Move as STOP
#        score = -9999999 #setting a huge negative value
#        alpha = 9999999
#        beta = -9999999
#        
#        for move in legalMoves:
#          agentState = gameState.generateSuccessor(0,move) 
#          compareScore = (agentState, 0, 1, alpha, beta)
#          if compareScore > score:
#            score = compareScore
#            currentMove = move #setting this move for return purpose
#          alpha = max(score, alpha)
#        return currentMove
#        
#        
#        util.raiseNotDefined()
    def getAction(self, gameState):
   
    #util.raiseNotDefined()
      return self.AlphaBeta(gameState, 1, 0, -sys.maxint, sys.maxint)

    def AlphaBeta(self, gameState, currentDepth, agentIndex, alpha, beta):
      if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    
      legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action!='Stop']
    
      # update next depth
      nextIndex = agentIndex + 1
      nextDepth = currentDepth
      if nextIndex >= gameState.getNumAgents():
          nextIndex = 0
          nextDepth += 1
    
      if agentIndex == 0 and currentDepth == 1: # pacman first move
        results = [self.AlphaBeta( gameState.generateSuccessor(agentIndex, action) , nextDepth, nextIndex, alpha, beta) for action in legalMoves]
        bestMove = max(results)
        bestIndices = [index for index in range(len(results)) if results[index] == bestMove]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]
    
      if agentIndex == 0:
        bestMove = -sys.maxint
        for action in legalMoves:
          bestMove = max(bestMove,self.AlphaBeta( gameState.generateSuccessor(agentIndex, action) , nextDepth, nextIndex, alpha, beta))
          if bestMove >= beta:
            return bestMove
          alpha = max(alpha, bestMove)
        return bestMove
      else:
        bestMove = sys.maxint
        for action in legalMoves:
          bestMove = min(bestMove,self.AlphaBeta( gameState.generateSuccessor(agentIndex, action) , nextDepth, nextIndex, alpha, beta))
          if alpha >= bestMove:
            return bestMove
          beta = min(beta, bestMove)
        return bestMove
    
    
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    stateLayout = currentGameState.getWalls()
    
    
    
    currentPosition 
    
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

