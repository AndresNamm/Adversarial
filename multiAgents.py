import math
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import pdb
import sys
import math
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
        #ReflexAgent.counter+=1
        #print( ReflexAgent.counter )

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
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates() # GhostPosition
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Getting the shortest distance through all the elements starting with next position

        unvisited = []
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y] == True:
                    unvisited.append((x,y))

        pos = newPos
        totalDistanceDots = 0.00000000001
        for i in unvisited:
            ## Find closest index to position
            distance=99999999
            index = -1
            for i in range(len(unvisited)):
                if manhattanDistance(pos,unvisited[i])<distance:
                    index = i
                    distance=manhattanDistance(pos,unvisited[i])
            totalDistanceDots+=distance
            ## Update next position and remove this element from unvister
            pos = unvisited[index]
            unvisited.pop(index)


        totalDistanceGhosts=0.00000000001
        for i in newGhostStates:
            totalDistanceGhosts+=manhattanDistance(newPos,i.getPosition())

        # print("Total distance to Ghost is " , totalDistanceGhosts)
        #print("Total distance through Dots is " , totalDistanceDots)
        # print("Division is " , totalDistanceGhosts/totalDistanceDots)

        ## Start escape when
        totalDistanceGhosts = min(3,totalDistanceGhosts)
        #return  totalDistanceGhosts + 1/totalDistanceDots + successorGameState.getScore()
        # Average 1200
        # totalDistanceGhosts , the bigger the better  from 1...2n
        # totalDistanceDots 0....1  shitty
        # scoreFunction  I guess some relation betwween time, dots eaten + utility if terminal state
        return totalDistanceGhosts*1/totalDistanceDots + successorGameState.getScore()










        # Get Shortest Distance Through food
        # Get Closest Boogie Monster



        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """

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
    def miniMax(self, gameState, cnt=0):
        terminal, utility = valueState(self,gameState, cnt)
        if terminal: return utility  # This Code construct makes no difference for minimax whther utility is a heuristic or real utiliyt
        states = [gameState.generateSuccessor(cnt%gameState.getNumAgents(), action) for action in gameState.getLegalActions(cnt%gameState.getNumAgents())]
        if cnt%gameState.getNumAgents()  == 0:
            return max(self.miniMax(state, cnt + 1) for state in states)
        else:
            return min(self.miniMax(state, cnt + 1) for state in states)

    def getAction(self, gameState):

        legalActions=gameState.getLegalActions()
        succGameState= [gameState.generateSuccessor(0,action) for action in legalActions]
        ans = [ self.miniMax(gameState,1) for gameState in succGameState]
        bestScore = max(ans)
        bestIndices = [ index for index in range(len(ans)) if ans[index]==bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalActions[chosenIndex]

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
        # run minimax, it returns the values of the states

        "*** YOUR CODE HERE ***"

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        legalActions=gameState.getLegalActions()
        succGameState= [gameState.generateSuccessor(0,action) for action in legalActions]
        #ans = [ self.miniMax(gameState,cnt=1) for gameState in succGameState]
        maximum=-sys.maxsize
        ans = []
        for state in succGameState:
            v = self.miniMax(state,alpha=maximum, beta=sys.maxsize,cnt=1)
            ans.append(v)
            # No need to check beta here, because it wont be updated in max, because max sets the minimum bound but not the max bound
            if v > maximum: maximum=v

        bestScore=max(ans)
        bestIndices = [index for index in range(len(ans)) if ans[index] == bestScore ]
        chosenIndex = random.choice(bestIndices)

        return legalActions[chosenIndex]

    def miniMax(self, gameState, alpha=-sys.maxsize, beta=sys.maxsize, cnt=0):
        terminal, utility = valueState(self,gameState, cnt)
        if terminal: return utility # This Code construct makes no difference for minimax whther utility is a heuristic or real utiliyt
        #if gameState.isWin() or gameState.isLose(): return gameState.getScore()
        succActions =[ action for action in gameState.getLegalActions(cnt % gameState.getNumAgents())]
        if cnt % gameState.getNumAgents() == 0: # => We are at a maximizer
            v=-sys.maxsize
            for action in succActions:
                state=gameState.generateSuccessor(cnt % gameState.getNumAgents(),action)
                v = max(v, self.miniMax(state,alpha=alpha,beta=beta,cnt=cnt+1))
                # print("Maximizer at level ", cnt)
                # print(alpha)
                # print(v)
                if v > beta:
                    break
                alpha=max(alpha,v)
            return v
        else:
            v=sys.maxsize
            for action in succActions:
                state = gameState.generateSuccessor(cnt % gameState.getNumAgents(), action)
                v = min(v,self.miniMax(state,alpha=alpha,beta=beta,cnt=cnt+1))

                if v < alpha:
                    break
                beta=min(v,beta)
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def miniMax(self, gameState, cnt=0):
        terminal, utility = valueState(self,gameState, cnt)
        if terminal: return utility  # This Code construct makes no difference for minimax whther utility is a heuristic or real utiliyt
        legalActions=[action for action in gameState.getLegalActions(cnt % gameState.getNumAgents())]
        states = [gameState.generateSuccessor(cnt%gameState.getNumAgents(), action) for action in legalActions]
        if cnt%gameState.getNumAgents()  == 0:
            #return sum( [self.miniMax(state, cnt + 1) for state in states] )/ len(legalActions)
            return max(self.miniMax(state, cnt + 1) for state in states)
        else:
            return sum([self.miniMax(state, cnt + 1) for state in states]) / len(legalActions)
            # bestIndices = [index for index in range(len(states))]
            # chosenIndex = random.choice(bestIndices)
            # return self.miniMax(states[chosenIndex], cnt + 1)


    def getAction(self, gameState):

        legalActions=gameState.getLegalActions()
        succGameState= [gameState.generateSuccessor(0,action) for action in legalActions]
        ans = [ self.miniMax(gameState,1) for gameState in succGameState]
        bestScore = max(ans)
        bestIndices = [ index for index in range(len(ans)) if ans[index]==bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalActions[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    prevFood = currentGameState.getFood()
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()  # GhostPosition
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Getting the shortest distance through all the elements starting with next position

    unvisited = []
    for x in range(newFood.width):
        for y in range(newFood.height):
            if newFood[x][y] == True:
                unvisited.append((x, y))

    pos = newPos
    totalDistanceDots = 0.00000000001
    for i in unvisited:
        ## Find closest index to position
        distance = 99999999
        index = -1
        for i in range(len(unvisited)):
            if manhattanDistance(pos, unvisited[i]) < distance:
                index = i
                distance = manhattanDistance(pos, unvisited[i])
        totalDistanceDots += distance
        ## Update next position and remove this element from unvister
        pos = unvisited[index]
        unvisited.pop(index)

    totalDistanceGhosts = 0.00000000001
    for i in newGhostStates:
        totalDistanceGhosts += manhattanDistance(newPos, i.getPosition())

    # print("Total distance to Ghost is " , totalDistanceGhosts)
    # print("Total distance through Dots is " , totalDistanceDots)
    # print("Division is " , totalDistanceGhosts/totalDistanceDots)

    ## Start escape when
    totalDistanceGhosts = min(5, totalDistanceGhosts)
    # return  totalDistanceGhosts + 1/totalDistanceDots + successorGameState.getScore()
    # Average 1200
    # totalDistanceGhosts , the bigger the better  from 1...2n
    # totalDistanceDots 0....1  shitty
    # scoreFunction  I guess some relation betwween time, dots eaten + utility if terminal state
    return totalDistanceGhosts / totalDistanceDots + successorGameState.getScore()



def valueState(agent,gameState, cnt):
    if gameState.isWin() or gameState.isLose():

        return [True,gameState.getScore()]
    # cnt is assumed to start from 0 , therefore the second move is 1, third 2, thus the condition following
    if math.floor(cnt / gameState.getNumAgents()) >=  agent.depth:
        #print(cnt, " ",gameState.getNumAgents()," ", cnt/gameState.getNumAgents())
        return [True,agent.evaluationFunction(gameState)] # Different agents have different evaluationFunctions
    return [False,-sys.maxsize]


# Abbreviation
better = betterEvaluationFunction

