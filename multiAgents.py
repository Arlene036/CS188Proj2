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
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # print('gameState:', gameState)
        # print('gameState.getLegalActions():', gameState.getLegalActions()) # ['Stop', 'East', 'North']

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # print('GameState.generatePacmanSuccessor(action):', successorGameState)

        newPos = successorGameState.getPacmanPosition()
        # print('GameState.getPacmanPosition():', newPos) # tuple 形如 (1, 1)

        newFood = successorGameState.getFood()
        # print('GameState.getFood().asList():', newFood.asList()) # 形如 [(1, 5), (1, 7), (2, 6), (2, 8), (3, 1), (3, 3), (3, 5), (3, 7)]

        newGhostStates = successorGameState.getGhostStates()
        # print('GameState.getGhostStates():', newGhostStates) # [<game.AgentState object at 0x0000020D474FEB00>]
        # print('GameState.getGhostStates() length:', len(newGhostStates)) # 1
        # print('GameState.getGhostStates()[0]', newGhostStates[0]) # Ghost: (x,y)=(2, 7), Stop
        # print('GameState.getGhostStates()[0].getPosition()', newGhostStates[0].getPosition()) # (2, 7)
        # print('GameState.getGhostStates()[0].getDirection()', newGhostStates[0].getDirection()) # Stop/East/North/..

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print('ghostState.scaredTimer:', newScaredTimes) # 返回一个list的形式 [0]

        "*** YOUR CODE HERE ***"
        "consider: food, ghost, scared_ghost, score"
        score = successorGameState.getScore()

        # food: 1) dis to the nearest food; 2) if newScaredTimes > nearest ghost distance, reduce the weight of food
        foodList = newFood.asList()
        foodDistance = [manhattanDistance(newPos, food) for food in foodList]
        nearestGhostDistance = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        weight_food = 1
        weight_reduced_food = 0.8

        if len(foodDistance) > 0:
            nearestFoodDistance = min(foodDistance)
            if newScaredTimes[0] > nearestGhostDistance:
                score += weight_reduced_food / (0.1+nearestFoodDistance)
            else:
                score += weight_food / (0.1+nearestFoodDistance)

        # ghost: 1) whether ghost is scared; 2) dis to the nearest ghost
        weight_ghost = 1
        if newScaredTimes[0] > nearestGhostDistance:
            score += weight_ghost / (0.1+nearestGhostDistance)
        else:
            score -= weight_ghost / (0.1+nearestGhostDistance)
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        def minimax(node, depth, agentIndex):
            if depth == 0 or node.isWin() or node.isLose():
                return self.evaluationFunction(node)

            legalActions = node.getLegalActions(agentIndex)

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                value = float('-inf')
                for action in legalActions:
                    successor = node.generateSuccessor(agentIndex, action)
                    value = max(value, minimax(successor, depth, agentIndex + 1))
                return value
            else:  # Ghosts' turns (minimizing players)
                value = float('inf')
                for action in legalActions:
                    successor = node.generateSuccessor(agentIndex, action)
                    if agentIndex == node.getNumAgents() - 1:  # Last ghost's turn, increment depth
                        value = min(value, minimax(successor, depth - 1, 0))
                    else:
                        value = min(value, minimax(successor, depth, agentIndex + 1))
                return value

        legalActions = gameState.getLegalActions(0)  # Pacman's legal actions
        bestAction = None
        bestValue = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, self.depth, 1)  # Start with the first ghost
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(node, depth, alpha, beta, agentIndex):
            if depth == 0 or node.isWin() or node.isLose():
                return self.evaluationFunction(node)

            legalActions = node.getLegalActions(agentIndex)

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                value = float('-inf')
                for action in legalActions:
                    successor = node.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(successor, depth, alpha, beta, agentIndex + 1))
                    if value > beta:
                        return value  # Beta cutoff
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts' turns (minimizing players)
                value = float('inf')
                for action in legalActions:
                    successor = node.generateSuccessor(agentIndex, action)
                    if agentIndex == node.getNumAgents() - 1:  # Last ghost's turn, increment depth
                        value = min(value, alphabeta(successor, depth - 1, alpha, beta, 0))
                    else:
                        value = min(value, alphabeta(successor, depth, alpha, beta, agentIndex + 1))
                    if value < alpha:
                        return value  # Alpha cutoff
                    beta = min(beta, value)
                return value

        legalActions = gameState.getLegalActions(0)  # Pacman's legal actions
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, self.depth, alpha, beta, 1)  # Start with the first ghost
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(node, depth, agentIndex):
            if depth == 0 or node.isWin() or node.isLose():
                return self.evaluationFunction(node)

            legalActions = node.getLegalActions(agentIndex)
            numActions = len(legalActions)

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                value = float('-inf')
                for action in legalActions:
                    successor = node.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(successor, depth, agentIndex + 1))
                return value
            else:  # Ghosts' turns (averaging players)
                value = 0
                for action in legalActions:
                    successor = node.generateSuccessor(agentIndex, action)
                    if agentIndex == node.getNumAgents() - 1:  # Last ghost's turn, increment depth
                        value += expectimax(successor, depth - 1, 0)
                    else:
                        value += expectimax(successor, depth, agentIndex + 1)
                return value / numActions  # Average over all ghost actions

        legalActions = gameState.getLegalActions(0)  # Pacman's legal actions
        bestAction = None
        bestValue = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, self.depth, 1)  # Start with the first ghost
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Consider mainly two factors, food and ghost.
    score is high if we are near to food; if ghost is scared, we focus more on eating ghost other than eating food
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    score = currentGameState.getScore()

    # food: 1) dis to the nearest food; 2) if ScaredTimes > nearest ghost distance, reduce the weight of food
    foodList = Food.asList()
    foodDistance = [manhattanDistance(Pos, food) for food in foodList]
    nearestGhostDistance = min([manhattanDistance(Pos, ghost.getPosition()) for ghost in GhostStates])
    weight_food = 1
    weight_reduced_food = 0.8

    if len(foodDistance) > 0:
        nearestFoodDistance = min(foodDistance)
        if ScaredTimes[0] > nearestGhostDistance:
            score += weight_reduced_food / (0.1 + nearestFoodDistance)
        else:
            score += weight_food / (0.1 + nearestFoodDistance)

    # ghost: 1) whether ghost is scared; 2) dis to the nearest ghost
    weight_ghost = 1
    if ScaredTimes[0] > nearestGhostDistance:
        score += weight_ghost / (0.1 + nearestGhostDistance)
    else:
        score -= weight_ghost / (0.1 + nearestGhostDistance)
    return score

# Abbreviation
better = betterEvaluationFunction
