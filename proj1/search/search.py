# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import searchAgents

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())
    # util.raiseNotDefined()

    def remove(fromFringe):
        return fromFringe.pop()

    def getCurrent(fringeNode):
        return fringeNode[0]
    
    def getParent(fringeNode):
        return fringeNode[1]

    def getState(fringeNode):
        return getCurrent(fringeNode)[0]

    def getAction(fringeNode):
        return getCurrent(fringeNode)[1]

    def getStepCost(fringeNode):
        return getCurrent(fringeNode)[2]

    def updatePath(dictionary, key, value):
        dictionary[key] = value

    def findPath(dictionary, goalLocation):
        path = []
        currentLocation = goalLocation
        while currentLocation!=start:
            path.append(dictionary[currentLocation][0])
            currentLocation = dictionary[currentLocation][2]
        path.reverse()
        return path

    #pathMap contains a dictionary of currentNode:parentPath_info
    #parentPath_info contains: (nodeAction, nodeStepCost, parentNode)
    pathMap = {}

    #closed stores information of visited nodes
    #only location of these visited nodes is relevant to store
    closed = []

    #fringe stores information as a tuple of (currentNodeInfo, parentNode)
    #currentNodeInfo: (currentNode, Action, StepCost), or the retVal of searchAgents.getSuccessors()
    #parentNode: (x,y), coordinates of the parentNode
    fringe = util.Stack()

    #note: startInfo matches format of return values from getSuccessors()
    #fringe invariant of (someNode, action, stepCost) is therefore preserved
    start = problem.getStartState()
    startNode = (start, None, 0)
    startInfo = (startNode, None)
    fringe.push(startInfo)

    while True:
        if fringe.isEmpty():
            return None

        node = remove(fringe)
        if problem.isGoalState(getState(node)):
            print "This is the goal node of DFS ", getState(node)
            # print "reached isGoalState()"
            return findPath(pathMap, getState(node))

        if getState(node) not in closed:
            # print "new time accessed"
            # print "node", node
            # print "state", getState(node)
            closed.append(getState(node))
            updatePath(pathMap, getState(node), (getAction(node), getStepCost(node), getParent(node)))

            childNodeList = problem.getSuccessors(getState(node))
            for childNode in childNodeList:
                fringe.push((childNode, getState(node)))
                #note: in case goal was found in childNode, it is important to update that in pathMap Dictionary
                if problem.isGoalState(childNode[0]):
                    #because goalStateNode is stupid
                    updatePath(pathMap, childNode[0], (childNode[1], childNode[2], getState(node)))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    def remove(fromFringe):
        return fromFringe.pop()

    def getCurrent(fringeNode):
        return fringeNode[0]
    
    def getParent(fringeNode):
        return fringeNode[1]

    def getState(fringeNode):
        return getCurrent(fringeNode)[0]

    def getAction(fringeNode):
        return getCurrent(fringeNode)[1]

    def getStepCost(fringeNode):
        return getCurrent(fringeNode)[2]

    def updatePath(dictionary, key, value):
        dictionary[key] = value

    def findPath(dictionary, goalLocation):
        path = []
        currentLocation = goalLocation
        while currentLocation!=start:
            path.append(dictionary[currentLocation][0])
            currentLocation = dictionary[currentLocation][2]
        path.reverse()
        return path

    #pathMap contains a dictionary of currentNode:parentPath_info
    #parentPath_info contains: (nodeAction, nodeStepCost, parentNode)
    pathMap = {}

    #closed stores information of visited nodes
    #only location of these visited nodes is relevant to store
    closed = []

    #fringe stores information as a tuple of (currentNodeInfo, parentNode)
    #currentNodeInfo: (currentNode, Action, StepCost), or the retVal of searchAgents.getSuccessors()
    #parentNode: (x,y), coordinates of the parentNode
    fringe = util.Queue()

    #note: startInfo matches format of return values from getSuccessors()
    #fringe invariant of (someNode, action, stepCost) is therefore preserved
    start = problem.getStartState()
    startNode = (start, None, 0)
    startInfo = (startNode, None)
    fringe.push(startInfo)

    while True:
        if fringe.isEmpty():
            return None

        node = remove(fringe)
        if problem.isGoalState(getState(node)):
            #print "This is the goal node of BFS ", getState(node)
            # print "reached isGoalState()"
            return findPath(pathMap, getState(node))

        if getState(node) not in closed:
            # print "new time accessed"
            # print "node", node
            # print "state", getState(node)
            closed.append(getState(node))
            updatePath(pathMap, getState(node), (getAction(node), getStepCost(node), getParent(node)))

            childNodeList = problem.getSuccessors(getState(node))
            for childNode in childNodeList:
                fringe.push((childNode, getState(node)))
                #note: in case goal was found in childNode, it is important to update that in pathMap Dictionary
                if problem.isGoalState(childNode[0]):
                    #because goalStateNode is stupid
                    updatePath(pathMap, childNode[0], (childNode[1], childNode[2], getState(node)))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    def remove(fromFringe):
        return fromFringe.pop()

    def getCurrent(fringeNode):
        return fringeNode[0]

    def getParent(fringeNode):
        return fringeNode[1]

    def getSum(fringeNode):
        return fringeNode[2]

    def getState(fringeNode):
        return getCurrent(fringeNode)[0]

    def getAction(fringeNode):
        return getCurrent(fringeNode)[1]

    def getStepCost(fringeNode):
        return getCurrent(fringeNode)[2]

    def updatePath(dictionary, key, value):
        dictionary[key] = value

    def findPath(dictionary, goalLocation):
        path = []
        currentLocation = goalLocation
        while currentLocation!=start:
            path.append(dictionary[currentLocation][0])
            currentLocation = dictionary[currentLocation][2]
        path.reverse()
        return path

    #pathMap contains a dictionary of currentNode:parentPath_info
    #parentPath_info contains: (nodeAction, nodeStepCost, parentNode, totalCost)
    #NOTE: included FOURTH element in the tuple: totalCost!
    pathMap = {}

    #closed stores information of visited nodes
    #only location of these visited nodes is relevant to store
    closed = []
    
    #fringe stores information as a tuple of (currentNodeInfo, parentNode)
    #currentNodeInfo: (currentNode, Action, StepCost), or the retVal of searchAgents.getSuccessors()
    #parentNode: (x,y), coordinates of the parentNode
    fringe = util.PriorityQueue()

    #note: startInfo matches format of return values from getSuccessors()
    #fringe invariant of (someNode, action, stepCost) is therefore preserved
    start = problem.getStartState()
    startNode = (start, None, 0)
    startInfo = (startNode, None, 0)
    #example structure: ((currentNode, Action, stepCost), parentNode, sumCost)
    #example structure: ((start, None, 0), None, 0)
    fringe.push(startInfo, 0)

    while True:
        if fringe.isEmpty():
            return None

        node = remove(fringe)
#        print "this is updated SUMCOST of the node", getStepCost(node)
        if problem.isGoalState(getState(node)):
            # print "reached isGoalState()"
            return findPath(pathMap, getState(node))

        if getState(node) not in closed:
#            print "new time accessed"
#            print "node", node
#            print "state", getState(node)
            closed.append(getState(node))
            updatePath(pathMap, getState(node), (getAction(node), getStepCost(node), getParent(node)))

            childNodeList = problem.getSuccessors(getState(node))
            for childNode in childNodeList:
                sumCost = getSum(node) + childNode[2]
                fringe.update((childNode, getState(node), sumCost), sumCost)
                #note: in case goal was found in childNode, it is important to update that in pathMap Dictionary
                if problem.isGoalState(childNode[0]):
                    #because goalStateNode is stupid
                    #Note: need to make this pretty
                    if childNode[0] in pathMap:
                        compareCost = pathMap[childNode[0]][1]
                        if compareCost >= sumCost:
                            updatePath(pathMap, childNode[0], (childNode[1], childNode[2], getState(node)))
                    else:
                        updatePath(pathMap, childNode[0], (childNode[1], childNode[2], getState(node)))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def remove(fromFringe):
        return fromFringe.pop()

    #fringeNode: (currentNodeInformation, parentNode, totalCost)
    def getCurrent(fringeNode):
        return fringeNode[0]

    #fringeNode: (currentNodeInformation, parentNode, totalCost)
    def getParent(fringeNode):
        return fringeNode[1]

    #fringeNode: (currentNodeInformation, parentNode, totalCost)
    def getSum(fringeNode):
        return fringeNode[2]
        
    #NOTE: these functions will only work on a freshly popped node from the fringe
    #NOTHING else.
    def getState(fringeNode):
        #getCurrent gets (currentNode, action, stepcost)
        #so getting 0 of that gets the currentNode or state
        return getCurrent(fringeNode)[0]
        
    def getAction(fringeNode):
        #so getting 1 of that gets the action
        return getCurrent(fringeNode)[1]

    def getStepCost(fringeNode):
        #so getting 2 of that gets the stepCost
        return getCurrent(fringeNode)[2]

    #dictionary = pathMap
    #key = currentNode
    #value = (action, stepCost, parentNode, sumCost)
    def updatePath(dictionary, key, value):
        dictionary[key] = value

    def findPath(dictionary, goalLocation):
        path = []
        currentLocation = goalLocation
        while currentLocation!=start:
            path.append(dictionary[currentLocation][0])
            currentLocation = dictionary[currentLocation][2]
        path.reverse()
        return path

    #pathMap contains a dictionary of currentNode:parentPath_info
    #parentPath_info contains: (nodeAction, nodeStepCost, parentNode, sumCost)
    #NOTE: included FOURTH element in the tuple: totalCost!
    pathMap = {}

    #closed stores information of visited nodes
    #only location of these visited nodes is relevant to store
    closed = []
    
    #fringe stores information as a tuple of (currentNodeInfo, parentNode)
    #currentNodeInfo: (currentNode, Action, StepCost), or the retVal of searchAgents.getSuccessors()
    #parentNode: (x,y), coordinates of the parentNode
    fringe = util.PriorityQueue()

    #note: startInfo matches format of return values from getSuccessors()
    #fringe invariant of (someNode, action, stepCost) is therefore preserved
    start = problem.getStartState()
    startNode = (start, None, 0)
    startInfo = (startNode, None, 0)
    #example structure: ((currentNode, Action, stepCost), parentNode, sumCost)
    #example structure: ((start, None, 0), None, 0)
    fringe.push(startInfo, 0)

    while True:
        if fringe.isEmpty():
            return None

        node = remove(fringe)
#        print "this is updated SUMCOST of the node", getStepCost(node)
        if problem.isGoalState(getState(node)):
            # print "reached isGoalState()"
            return findPath(pathMap, getState(node))

        if getState(node) not in closed:
            closed.append(getState(node))
            updatePath(pathMap, getState(node), (getAction(node), getStepCost(node), getParent(node), getSum(node)))

            childNodeList = problem.getSuccessors(getState(node))
            for childNode in childNodeList:
                sumCost = getSum(node) + childNode[2]
                sumHeuristicCost = sumCost + heuristic(childNode[0], problem)
                fringe.update((childNode, getState(node), sumCost), sumHeuristicCost)
                #note: in case goal was found in childNode, it is important to update that in pathMap Dictionary
                if problem.isGoalState(childNode[0]):
                    #because goalStateNode is stupid
                    #Note: need to make this pretty
                    if childNode[0] in pathMap:
                        compareCost = pathMap[childNode[0]][3]#comparing the sumCost
                        if compareCost >= sumCost:
                            updatePath(pathMap, childNode[0], (childNode[1], childNode[2], getState(node), sumCost))
                    else:
                        updatePath(pathMap, childNode[0], (childNode[1], childNode[2], getState(node), sumCost))#need to include the getSum cost/value in here


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
