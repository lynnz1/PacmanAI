# myTeam.py
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


# AGENTS AVAILABLE:
# =========================================
#  - RLAgent - OffensiveAgent : Offensive agent which uses re-inforcement learning (Q learning) techiniques
#  - OffensiveAstar (USED) : Offence agent which uses A* search to solve the problem 
#  
#  - QLearningDefence : Defensive Agent which uses Q learning technique
#  - DefensiveAgent (USED):  Defensive agent with Evaluation Function 
#  - DefenceAgent :  Defensive agents with Heuristic function (used in preliminary submission)
#  - MctsDefensiveAgent : Defensive agent with Monte Carlo Trees 
# 
#  - DefensiveAgentWithOffensive Agent : Decision tree agents which switches between Deffence from Offence agents
#  - OffenceAndDefence (USED) : Decision tree agents which switches between Offence and Defence agents 


# RL Imports 
import random, time, util
from captureAgents import CaptureAgent
from game import Directions, Actions
from copy import deepcopy

#MONTECARLO IMPORTS 
import distanceCalculator
import sys
import game
import math 
from util import nearestPoint 

from capture import SIGHT_RANGE, GameState

TIME_LIMIT = 0.92


#################
# Team creation #
#################
# def createTeam(firstIndex, secondIndex, isRed,
#                first = 'OffenceAndDefence', second = 'QLearningDefence'):

# def createTeam(firstIndex, secondIndex, isRed,
#                first = 'OffenceAndDefence', second = 'DefensiveAgentWithOffensive'):

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffenceAndDefence', second = 'OffenceAndDefence'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class RLAgent(CaptureAgent):
    def __init__(self, _index):
        super().__init__(_index)

        self.epsilon = 0.0
        self.alpha = 0.0
        self.discount = 0.9
        self.num_train = 0
        self.episode = 0
        self.reward_train = 0.0
        self.reward_test = 0.0
        self.previousActions = []

        self.weights = {
            'in_deadend': -999,
            'in-danger': 0,
            'eat_capsule': 300,
            'dist_ghost': 5,
            'dist_food': -3,
            'dist_capsule': -5,
            'eat_food': 250,
            'food_to_base': 400,
            'stop': -100,
            'dist_entry': -3,
            'defense': -2,
            'attack': 1,
            'dist_border': -3,
            'eaten_by_ghost': -10000,
        }

    # helper functions
    def getRewards(self, gameState):
        return gameState.getScore()

    def getLegalActions(self, gameState):
        return gameState.getLegalActions(self.index)

    def startEpisode(self):
        self.reward_e = 0.0
        self.lastAction = None
        self.lastGameState = None

    def observeTransition(self, gameState, action, nextGameState, reward):
        self.agentSync(gameState, action, nextGameState, reward)
        self.reward_e = self.reward_e + reward

    def terminateEpis(self):

        if self.episode < self.num_train:
            self.reward_train += self.reward_e
        else:
            self.reward_test += self.reward_e

        self.episode = self.episode + 1

        if self.episode >= self.num_train:
            self.alpha = 0.0
            self.epsilon = 0.0

    def isTest(self):
        return not self.isTrain()

    def isTrain(self):
        return self.episode < self.num_train

    ################################
    #          value-iteration        #
    ################################
    def getFeatures(self, gameState, action):
        features = util.Counter()
        return features

    def getWeights(self):
        return self.weights

    def agentSync(self, currentGameState, passedAction, tdxedGameState, rewardValue):
        featuresCounter = self.getFeatures(currentGameState, passedAction)
        for feature, weight in featuresCounter.items():
            deficit = (rewardValue + self.discount * self.getMaxQvalue(tdxedGameState)) - self.getFeatures(
                currentGameState, passedAction) * self.getWeights()
            self.weights[feature] = self.weights[feature] + self.alpha * deficit * weight

    def takeAction(self, state, action):
        self.lastAction = action
        self.lastGameState = state
        self.previousActions.append(action)

    def getMaxQvalue(self, gameState, action):
        value_list = []
        actions = self.getLegalActions(gameState)
        maximum = 0.0
        if len(actions) > 0:
            for action in actions:
                weights = self.getWeights()
                features = self.getFeatures(gameState, action)
                value_list.append(features * weights)
                maximum = max(value_list)
            return maximum

    def getActionByQvalue(self, gameState):
        bestActionsList = []
        actions = self.getLegalActions(gameState)
        maxValue = self.getMaxQvalue(gameState)

        for action in actions:
            if self.getFeatures(gameState, action) * self.getWeights() == maxValue:
                bestActionsList.append(action)

        if len(bestActionsList) > 0:
            return random.choice(bestActionsList)
        elif len(actions) > 0:
            return random.choice(actions)
        else:
            return None

    def getPolicy(self, gameState):
        return self.getActionByQvalue(gameState)

    ####################
    #  Agent Specification  #
    ####################

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.startEpisode()
        self.startPos = gameState.getAgentPosition(self.index)

    def chooseAction(self, gameState):

        loopActsReg = self.ifLoop(gameState)
        legalActsReg = self.getLegalActions(gameState)

        if (loopActsReg in legalActsReg) and (loopActsReg is not None):
            legalActsReg.remove(loopActsReg)
            return random.choice(legalActsReg)

        if len(legalActsReg) == 0:
            return None

        if util.flipCoin(self.epsilon):
            self.takeAction(gameState, random.choice(legalActsReg))
        else:
            self.takeAction(gameState, self.getPolicy(gameState))

    def getSuccessor(self, gameState, action):

        position = gameState.generateSuccessor(self.index, action).getAgentState(self.index).getPosition()
        if position != util.nearestPoint(position):
            return gameState.generateSuccessor(self.index, action).generateSuccessor(self.index, action)
        else:
            return gameState.generateSuccessor(self.index, action)

    def getEnemyState(self, gameState):
        statesList = []
        for x in self.getOpponents(gameState):
            statesList.append(gameState.getAgentState(x))
            return statesList

    def getMyState(self, gameState):
        return gameState.getAgentState(self.index)

    def getMyPosition(self, gameState):
        return self.getMyState(gameState).getPosition()

    def ifLoop(self, gameState):
        lengthCheck = len(self.previousActions)

        if lengthCheck < 4:
            return None

        if self.previousActions[-3] == self.previousActions[-4] and self.previousActions[-1] == self.previousActions[-2] \
                and self.previousActions[-2] != self.previousActions[-4] and self.previousActions[-1] != \
                self.previousActions[-3]:
            loopActions = self.previousActions[-2]
            self.previousActions = []
            return loopActions
        else:
            return None

    def final(self, gameState):

        # terminal state

        reward = self.getRewards(gameState)
        self.observeTransition(self.lastGameState, self.lastAction, gameState, reward)
        self.terminateEpis()



################################################################################
#   OFFENCE AGENT
################################################################################

# Offensive agent which uses reinforcement learning (Q learning)
class OffensiveAgent(RLAgent):

    def valleyEnds(self, position):
        return position in self.deadEnds

    def getFeatures(self, gameState, action):
        sucessorState = self.getSuccessor(gameState, action)
        NewState = self.getMyState(sucessorState)

        uniqueFeaturesCounter = util.Counter()

        stateOfNow = self.getMyState(gameState)
        newPosition = NewState.getPosition()
        enemyList = self.getEnemyState(gameState)
        walls = gameState.getWalls()
        foods = self.getFood(gameState)
        capsules = self.getCapsules(gameState)

        x, y = newPosition
        x = int(x)
        y = int(y)

        if action == Directions.STOP:
            uniqueFeaturesCounter['stop'] = 1

        ghostAgentList = []

        for ghost in enemyList:
            if (ghost.getPosition() is not None) and (not ghost.isPacman) and (ghost.scaredTimer < 1):
                ghostAgentList.append(ghost)

        if len(ghostAgentList) > 0:
            dist_min = 100000
            for ghost in ghostAgentList:
                dist = self.getMazeDistance(newPosition, ghost.getPosition())
                if dist < 5:
                    if dist_min > dist:
                        dist_min = dist
                        self.closestGhost = ghost

        if not NewState.isPacman:
            maxScore = -99999
            food_list = foods.asList()
            for entry in self.entries:
                if self.closestGhost:
                    distToGhost = self.getMazeDistance(entry, self.closestGhost.getPosition())
                else:
                    distToGhost = 0

                if len(food_list) > 0:
                    list_food = []
                    for food in food_list:
                        list_food.append(self.getMazeDistance(entry, food))

                    dist_food = min(list_food)

                    score = distToGhost - dist_food
                else:
                    score = distToGhost

                if score > maxScore:
                    maxScore = score
                    self.bestEntry = entry

            uniqueFeaturesCounter['dist_entry'] = float(self.getMazeDistance(self.bestEntry, newPosition)) / (
                        walls.height * walls.width)

            inDanger = False
            if self.closestGhost and self.getMazeDistance(self.closestGhost.getPosition(),
                                                          stateOfNow.getPosition()) <= 2:
                inDanger = True

            if stateOfNow.isPacman and newPosition == self.startPos:
                uniqueFeaturesCounter['eaten_by_ghost'] = 1

            if stateOfNow.isPacman and newPosition != self.startPos and stateOfNow.numCarrying > 0:
                uniqueFeaturesCounter['food_to_base'] = 1

            if stateOfNow.isPacman and (not inDanger) and (
                    newPosition != self.startPos) and stateOfNow.numCarrying == 0:
                uniqueFeaturesCounter['defense'] = 1


        else:
            if (not stateOfNow.isPacman) and NewState.isPacman and self.getMazeDistance(self.bestEntry,
                                                                                        stateOfNow.getPosition()) <= 1:
                uniqueFeaturesCounter['attack'] = 1

            closestGhost = None

            ghostAgentList = []

            for ghost in enemyList:
                if ghost.getPosition() is not None and not ghost.isPacman:
                    ghostAgentList.append(ghost)
            if len(ghostAgentList) > 0:
                dist_min = 999999
                for ghost in ghostAgentList:
                    dist = self.getMazeDistance(newPosition, ghost.getPosition())
                    if dist <= 5:
                        if dist < dist_min:
                            dist_min = dist
                            closestGhost = ghost

            if closestGhost and self.getMazeDistance(closestGhost.getPosition(),
                                                     newPosition) <= 5 and closestGhost.scaredTimer <= 5:
                distToGhost = self.getMazeDistance(closestGhost.getPosition(), newPosition)
                uniqueFeaturesCounter['in_danger'] = 1
                uniqueFeaturesCounter['dist_ghost'] = float(distToGhost) / (walls.height * walls.width)

                if self.valleyEnds((x, y)):
                    uniqueFeaturesCounter['in_deadend'] = 1

                    uniqueFeaturesCounter['dist_ghost'] = -uniqueFeaturesCounter['dist_ghost']

                if distToGhost <= 1:
                    uniqueFeaturesCounter['eaten_by_ghost'] = 1

                dist_border_list = []
                for entry in self.entries:
                    dist_border_list.append(self.getMazeDistance(newPosition, entry))
                closest_border = min(dist_border_list)

                if len(capsules) > 0 and int(gameState.data.timeleft) / 4 - 5 >= closest_border:
                    if newPosition in capsules:
                        uniqueFeaturesCounter['eat_capsule'] = 1.0
                    else:
                        capsule_distance = []
                        for capsule in capsules:
                            capsule_distance.append(self.getMazeDistance(capsule, newPosition))
                        distToCapsule = min(capsule_distance)

                        capsule_list = []
                        for capsule in capsules:
                            if self.getMazeDistance(capsule, newPosition) == distToCapsule:
                                capsule_list.append(capsule)
                        best_capsule = capsule_list[0]

                        uniqueFeaturesCounter['dist_capsule'] = float(distToCapsule) / (walls.height * walls.width)
                        if self.lastGameState:
                            LastPosition = self.getMyState(self.lastGameState).getPosition()

                            lastDistToCapsule = self.getMazeDistance(best_capsule, LastPosition)

                            ghostDistToCapsule = self.getMazeDistance(best_capsule, closestGhost.getPosition())

                            # turn off dead end then get capsule
                            if distToCapsule < lastDistToCapsule and self.valleyEnds(best_capsule) and self.valleyEnds(
                                    (x, y)) and ghostDistToCapsule > distToCapsule:
                                uniqueFeaturesCounter['in_deadend'] = 0
                                uniqueFeaturesCounter['dist_ghost'] = -uniqueFeaturesCounter['dist_ghost']
                else:
                    distToHome = self.getMazeDistance(self.startPos, newPosition)
                    uniqueFeaturesCounter['dist_border'] = float(distToHome) / (walls.height * walls.width)

            if self.closestGhost and closestGhost is None and self.getMazeDistance(self.closestGhost.getPosition(),
                                                                                   newPosition) <= 5:
                self.closestGhost = None

            if len(capsules) > 0:
                if newPosition in capsules:
                    uniqueFeaturesCounter['eat_capsule'] = 1.0

            if uniqueFeaturesCounter['dist_ghost'] == 0.0:
                uniqueFeaturesCounter['dist_ghost'] = 4.0 / (walls.height * walls.width)

            if uniqueFeaturesCounter['in_danger'] != 1:
                border_list = []
                food_list = foods.asList()
                for entry in self.entries:
                    border_list.append(self.getMazeDistance(newPosition, entry))
                closest_border = min(border_list)

                if int(gameState.data.timeleft) / 4 - 5 < closest_border or len(food_list) <= 2:
                    distToHome = self.getMazeDistance(self.startPos, newPosition)
                    uniqueFeaturesCounter['dist_border'] = float(distToHome) / (walls.height * walls.width)
                else:
                    maxScore = -99999
                    bestFood = None
                    if len(food_list) > 0:
                        for food in food_list:
                            if self.closestGhost:
                                distToGhost = self.getMazeDistance(food, self.closestGhost.getPosition())
                            else:
                                distToGhost = 0

                            dist_food = self.getMazeDistance(newPosition, food)
                            score = distToGhost - dist_food
                            if score > maxScore:
                                bestFood = food
                                maxScore = score

                    if bestFood is not None:
                        dist_food = self.getMazeDistance(newPosition, bestFood)
                    else:
                        dist_food = None
                    # check condition
                    addDist_food = (bestFood is not None) and (
                                dist_food < closest_border or NewState.numCarrying == 0) and (
                                               stateOfNow.numCarrying < int(self.totalFoodNum / 2))

                    if foods[x][y]:
                        uniqueFeaturesCounter['eat_food'] = 1.0
                    if addDist_food:
                        uniqueFeaturesCounter['dist_food'] = float(dist_food) / (walls.height * walls.width)
                    else:
                        uniqueFeaturesCounter['dist_border'] = float(closest_border) / (walls.height * walls.width)

        # uniqueFeaturesCounter.divideAll(10)
        # print(uniqueFeaturesCounter)

        return uniqueFeaturesCounter

    def registerInitialState(self, gameState):
        layout = deepcopy(gameState.data.layout)
        mapY = layout.height
        mapX = layout.width

        RLAgent.registerInitialState(self, gameState)

        if self.red:
            midline_X = int(mapX / 2) - 1
            enemy_X_pos = int(mapX / 2)
            start_x = enemy_X_pos
            end_x = mapX
        else:
            midline_X = int(mapX / 2)
            enemy_X_pos = int(mapX / 2) - 1
            start_x = 0
            end_x = enemy_X_pos
        self.entries = []

        for midline_Y in range(0, mapY):
            if layout.isWall((midline_X, midline_Y)):
                continue
            else:
                self.entries.append((midline_X, midline_Y))

        self.deadEnds = []
        lastLength = len(self.deadEnds)
        while True:
            latestWalls = deepcopy(layout.walls)
            for horizontal in range(start_x, end_x):
                for verticalAxis in range(0, mapY):
                    if layout.walls[horizontal][verticalAxis]:
                        continue
                    else:
                        self.deadEnds.append((horizontal, verticalAxis))
                        latestWalls[horizontal][verticalAxis] = True
            layout.walls = latestWalls
            if lastLength == len(self.deadEnds):
                break
            else:
                lastLength = len(self.deadEnds)

        self.bestEntry = None
        self.closestGhost = None
        self.startPos = gameState.getAgentState(self.index).getPosition()
        self.totalFoodNum = len(self.getFood(gameState).asList())

    def getRewards(self, gameState):
        rewardPoints = 0
        agentPos = self.getMyPosition(gameState)
        x, y = agentPos[0], agentPos[1]

        agentPreviousPos = self.getMyPosition(self.lastGameState)
        previousCapsulesMatrix = self.getCapsules(self.lastGameState)
        previousFoodsMatrix = self.getFood(self.lastGameState)

        agentPreviousState = self.getMyState(self.lastGameState)
        agentState = self.getMyState(gameState)

        if previousFoodsMatrix[int(x)][int(y)]:
            rewardPoints = rewardPoints + 1

        if agentPos == agentPreviousPos:
            rewardPoints = rewardPoints - 1

        if agentPos == self.startPos and agentPreviousState.isPacman and not agentState.isPacman:
            rewardPoints = rewardPoints - 100

        if agentPos in previousCapsulesMatrix:
            rewardPoints = rewardPoints + 20

        if agentPreviousState.numCarrying > agentState.numCarrying and agentPos != self.startPos:
            rewardPoints += 40 * agentPreviousState.numCarrying

        return rewardPoints


# Offensive agent which uses A* search
class OffensiveAstar(CaptureAgent):

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.start = gameState.getAgentPosition(self.index)
        self.previousActions = []

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

         # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

    def getNearestGhostDistance(self, gameState):
        currentPosition = gameState.getAgentState(self.index).getPosition()

        opponents = []
        for i in self.getOpponents(gameState):
            opponents.append(gameState.getAgentState(i))

        ghost_opponent = []
        for opponent in opponents:
            if not opponent.isPacman and opponent.getPosition() != None:
                ghost_opponent.append(opponent)

        if ghost_opponent != None and len(ghost_opponent) > 0:
            distances = []
            for g in ghost_opponent:
                if g.configuration != None:
                    distances.append(self.getMazeDistance(
                        currentPosition, g.getPosition()))
            if len(distances) > 0:
                return min(distances)
        return None

    def distanceToBoundary(self, gameState):
        currentPosition = gameState.getAgentState(self.index).getPosition()
        if self.red:
            i =  int(gameState.data.layout.width /2 -1)
        else:
            i = int(gameState.data.layout.width /2 + 1)
        boundaries = [(i,j) for j in  range(gameState.data.layout.height)]
        
        midBoundary = []
        minDistance = []
        for b in boundaries:
            if not gameState.hasWall(b[0],b[1]):
                midBoundary.append(b)
            
        for m in midBoundary:
            minDistance.append(self.getMazeDistance(m,currentPosition))
        return min(minDistance)
        
    def minimumFoodDistance(self, currentPostion, gameState):
        foodAvailable = self.getFood(gameState).asList()
        foodDistance = []
        for food in foodAvailable:
            foodDistance.append(self.getMazeDistance(currentPostion, food))
        # print(min(foodDistance))
        return min(foodDistance)

    def chooseAction(self, gameState):
        current_state = gameState.getAgentState(self.index)
        current_position = current_state.getPosition()

        # find whether the opponent is in the scared state
        scared_time = 0
        for opponent in self.getOpponents(gameState):
            scared = gameState.getAgentState(opponent).scaredTimer
            if scared > 1:
                scared_time = scared
                break

        # find wheather capsule is close by
        isCapsuleClose = False
        for cap in self.getCapsules(gameState):
            if self.getMazeDistance(current_position, self.getCapsules(gameState)[0]) < 6:
                isCapsuleClose = True

        # decision tree
        # come back home if food left is at 2
        if self.getFood(gameState).count() <= 2 or gameState.data.timeleft < self.distanceToBoundary(gameState) + 60:
            problem = SearchEscape(gameState, self, self.index)
            path = self.aStarSearch(problem, gameState, self.offenceHeuristc)
        # if there is capsule exist and is close
        elif len(self.getCapsules(gameState)) > 0 and scared_time < 10 and isCapsuleClose:
            problem = SearchCapsule(gameState, self, self.index)
            path = self.aStarSearch(problem, gameState, self.offenceHeuristc)

        # if there is enermy nearby
        elif self.getNearestGhostDistance(gameState) != None and self.getNearestGhostDistance(gameState) < 5 and scared_time < 5:
            # if capusule is closer than the enermy
            if len(self.getCapsules(gameState)) > 0 and self.getMazeDistance(current_position, self.getCapsules(gameState)[0])\
                 < self.getNearestGhostDistance(gameState):
                problem = SearchCapsule(gameState, self, self.index)
                path = self.aStarSearch(problem, gameState, self.offenceHeuristc)
            # if agent is in the home boundary
            elif (self.red and current_position[0] < (gameState.data.layout.width /2)) \
                or (not self.red and current_position[0] >= (gameState.data.layout.width /2)):
                problem = SearchHome(gameState, self, self.index)
                path = self.aStarSearch(problem, gameState, self.offenceHeuristc)

            else:
                problem = SearchEscape(gameState, self, self.index)
                path = self.aStarSearch(problem, gameState, self.offenceHeuristc)

        # if agent is carrying food and is close to home boundary
        elif current_state.numCarrying > 1 and abs(current_position[0] - (gameState.data.layout.width /2)) < 3:
            if self.minimumFoodDistance(current_position, gameState) < 2:
                problem = SearchFoodCapusule(gameState, self, self.index)
                path = self.aStarSearch(problem, gameState, self.offenceHeuristc)
            else:
                problem = SearchEscape(gameState, self, self.index)
                path = self.aStarSearch(problem, gameState, self.offenceHeuristc)

        elif current_state.numCarrying > 1 and self.getNearestGhostDistance(gameState) != None \
            and abs(current_position[0] - (gameState.data.layout.width /2)) < 5:
            problem = SearchEscape(gameState, self, self.index)
            path = self.aStarSearch(problem, gameState, self.offenceHeuristc)

        # come back home if agent carry more than 5 food
        elif (current_state.numCarrying > 5 and scared_time < 5):
            # if there is food nearby and opponent ghost is not close by
            if self.getNearestGhostDistance(gameState) == None and self.minimumFoodDistance(current_position, gameState) < 3:
                problem = SearchFoodCapusule(gameState, self, self.index)
                path = self.aStarSearch(problem, gameState, self.offenceHeuristc)
            else: 
                problem = SearchEscape(gameState, self, self.index)
                path = self.aStarSearch(problem, gameState, self.offenceHeuristc)
  

        else:
            problem = SearchFoodCapusule(gameState, self, self.index)
            path = self.aStarSearch(problem, gameState, self.offenceHeuristc)

        while len(path) == 0:
            return 'Stop'
        return path[0]
    
    def pathToLastEatenFood(self, gameState, foodPosition):
        problem = SearchLastEatenFood(gameState, self, self.index)
        problem.setGoal(foodPosition)
        path = self.aStarSearch(problem, gameState, self.offenceHeuristc)
        return path


    def nullHeuristic(self, state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest
        goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0

    def offenceHeuristc(self, state, gameState):
        heuristic = 0

        # if there is ghost nearby
        if self.getNearestGhostDistance(gameState) != None:
            enemies = []
            ghost = []

            # detect ghost
            for opponent in self.getOpponents(gameState):
                enemies.append(gameState.getAgentState(opponent))
            for enermy in enemies:
                if not enermy.isPacman and enermy.scaredTimer < 2 and enermy.getPosition() != None:
                    ghost.append(enermy)

            if len(ghost) > 0:
                ghost_positions = []
                for g in ghost:
                    ghost_positions.append(g.getPosition())

                ghost_distances = []
                for gp in ghost_positions:
                    ghost_distances.append(self.getMazeDistance(state, gp))
                min_ghost_distance = min(ghost_distances)
                # give heuristic
                if min_ghost_distance < 6:
                    heuristic = (6-min_ghost_distance) ** 5

        return heuristic

    def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):
        """Search the node that has the lowest combined cost and heuristic first."""
        start_state = problem.getStartState()
        currentNode = {'state': problem.getStartState(),
                       'cost_start': 0,
                       'cost': heuristic(start_state, gameState)}

        # check if it is the goal
        if problem.isGoalState(currentNode['state']):
            return []

        priorityQueue = util.PriorityQueue()
        priorityQueue.push(currentNode, currentNode['cost'])
        pq_exist = set()
        pq_exist.add(currentNode['state'])
        visited = set()
        path = []

        while(not priorityQueue.isEmpty()):
            currentNode = priorityQueue.pop()
            pq_exist.discard(currentNode['state'])

            # check for goal
            if(problem.isGoalState(currentNode['state'])):
                # create path
                while 'parent' in currentNode:
                    path.append(currentNode['action'])
                    currentNode = currentNode['parent']
                # reverse its path
                path.reverse()
                return path

            visited.add(currentNode['state'])

            # for each adjacent node
            for next in problem.getSuccessors(currentNode['state']):
                childNode = {
                    'state': next[0],
                    'action': next[1],
                    'cost_start': next[2] + currentNode['cost_start'],
                    'cost': next[2] + currentNode['cost_start'] + heuristic(next[0], gameState),
                    'parent': currentNode
                }

                # if node hasn't been previously visited
                if childNode['state'] not in visited and childNode['state'] not in pq_exist:
                    priorityQueue.push(childNode, childNode['cost'])
                    pq_exist.add(childNode['state'])
                # if it is in queue update with lowest cost
                elif childNode['state'] in pq_exist:
                    priorityNode = priorityQueue.pop()
                    if priorityNode['cost'] > childNode['cost']:
                        priorityQueue.push(childNode, childNode['cost'])
                    else:
                        priorityQueue.push(priorityNode, priorityNode['cost'])
        return path

################################################################################
#   DEFFENCE AGENT
################################################################################

# defensive agent using Q learning (RL)
# To train the QLearningDefence agent, set 'numTraining' in createTeam to how many times 
# it needs to trained, the 'num_train' parameter inside QLearningDefence also needs to be
# set. Finally, the command to run Pacman is python3 capture.py -r my team.py -b baselineTeam.py -n 101
# if we want it to train 100 times.
class QLearningDefence(CaptureAgent):
    def __init__(self, _index):
        super().__init__(_index)
        # QLearning Parameter Settings
        self.epsilon = 0.8 # Greedy Parameter
        self.alpha = 0.85 # Learning Rate
        # self.gamma = 0.8
        self.discount = 0.9
        
        self.num_train = 10 # Num of trainings to be performed
        self.episode = 0 # Counter for the number of games
        self.reward_train = 0.0
        self.reward_test = 0.0
        self.previousActions = []
        
        self.score = 0
        self.q_value = util.Counter()
        self.weights = {
            'in_deadend': -999,
            'in-danger': 0,
            'eat_capsule': 300,
            'dist_ghost': 5,
            'dist_food': -3,
            'dist_capsule': -5,
            'eat_food': 250,
            'food_to_base': 400,
            'stop': -100,
            'dist_entry': -3,
            'defense': -2,
            'attack': 1,
            'dist_border': -3,
            'eaten_by_ghost': -10000,
        }

    # helper functions
    def getRewards(self, gameState):
        return gameState.getScore()

    def getLegalActions(self, gameState):
        return gameState.getLegalActions(self.index)


    def startEpisode(self):
        self.reward_e = 0.0
        self.lastAction = None
        self.lastGameState = None


    def observeTransition(self, gameState, action, nextGameState, reward):
        self.agentSync(gameState, action, nextGameState, reward)
        self.reward_e = self.reward_e + reward

 
    def terminateEpis(self):
            
        if self.episode < self.num_train:
            self.reward_train += self.reward_e
        else:
            self.reward_test += self.reward_e

        self.episode = self.episode + 1

        if self.episode >= self.num_train:
            self.alpha = 0.0  
            self.epsilon = 0.0   
        

    def isTest(self):
        return not self.isTrain()


    def isTrain(self):
        return self.episode < self.num_train

        

    ################################
    #          value-iteration        #
    ################################
    def getFeatures(self, gameState, action):
        features = util.Counter()
        return features

    def getWeights(self):
        return self.weights

    def agentSync(self, currentGameState, passedAction, tdxedGameState, rewardValue):
        featuresCounter = self.getFeatures(currentGameState, passedAction)
        for feature, weight in featuresCounter.items():
            deficit = (rewardValue + self.discount * self.fetchValue(tdxedGameState)) - self.getQValue(currentGameState, passedAction)
            self.weights[feature] = self.weights[feature] + self.alpha * deficit * weight

    def takeAction(self, state, action):
        self.lastAction = action
        self.lastGameState = state
        self.previousActions.append(action)


    def fetchValue(self, gameState):
        return self.getMaxQvalue(gameState)


    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights()
        return features * weights

    def getMaxQvalue(self, gameState):
        value_list = []
        actions = self.getLegalActions(gameState)
        if len(actions) > 0:
            for action in actions:
                value_list.append(self.getQValue(gameState,action))
                maximum = max(value_list)
            return maximum
        return 0.0

    def getActionByQvalue(self, gameState):
        bestActionsList = []
        actions = self.getLegalActions(gameState)
        maxValue = self.getMaxQvalue(gameState)
        
        for action in actions:
            if self.getQValue(gameState, action) == maxValue:
                bestActionsList.append(action)

        if len(bestActionsList) > 0:
            return random.choice(bestActionsList)
        elif len(actions) > 0:
            return random.choice(actions)
        else:
            return None

    def getPolicy(self, gameState):
        return self.getActionByQvalue(gameState)


   
    ####################
    #  Agent Specification  #
    ####################

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.startEpisode()
        self.startPos = gameState.getAgentPosition(self.index)

    # update Q value
    # Update Q Value using the Bellman equation
    def updateQ(self, state, action, reward, qmax):
        q = self.getQValue(state,action)
        self.q_value[(state,action)] = q + self.alpha*(reward + self.discount*qmax - q)

    def chooseAction(self, gameState):
       
        loopActsReg = self.ifLoop(gameState)
        legalActsReg = self.getLegalActions(gameState)

        # update Q-value
        reward = gameState.getScore()-self.score
        if self.lastGameState is not None:
            # Get the maximum QValue
            max_q = self.getMaxQvalue(gameState)
            # Update QValue table with the best score
            self.updateQ(self.lastGameState, self.lastAction, reward, max_q)
        
        if (loopActsReg in legalActsReg) and (loopActsReg is not None):
            legalActsReg.remove(loopActsReg)
            return random.choice(legalActsReg)

        if len(legalActsReg) == 0:
            return None

        # Greedy Search 
        if util.flipCoin(self.epsilon):
            toBeAct = random.choice(legalActsReg)
        else:
            toBeAct = self.getPolicy(gameState)

        self.takeAction(gameState, toBeAct)
        self.score = gameState.getScore()
        return toBeAct
    
    def getSuccessor(self, gameState, action):

        successorState = gameState.generateSuccessor(self.index, action)
        position = successorState.getAgentState(self.index).getPosition()
        if position != util.nearestPoint(position):
            return successorState.generateSuccessor(self.index, action)
        else:
            return successorState


    def getEnemyState(self, gameState):
        statesList = []
        for x in self.getOpponents(gameState):
            statesList.append(gameState.getAgentState(x))
            return statesList

    
    def getMyState(self, gameState):
        return gameState.getAgentState(self.index)


    def getMyPosition(self, gameState):
        return self.getMyState(gameState).getPosition()

    def ifLoop(self, gameState):
        lengthCheck = len(self.previousActions)

        if lengthCheck < 4:
            return None

        if self.previousActions[-3] == self.previousActions[-4] and self.previousActions[-1] == self.previousActions[-2]\
                and self.previousActions[-2] != self.previousActions[-4] and self.previousActions[-1] != self.previousActions[-3] :
            loopActions = self.previousActions[-2]
            self.previousActions = []
            return loopActions
        else:
            return None

    def final(self, gameState):
    
         # terminal state
       
        reward = self.getRewards(gameState)
        self.observeTransition(self.lastGameState, self.lastAction, gameState, reward)
        self.terminateEpis()

# QLearning Defence Agent


# Defensive agent with Evaluation Function
class DefensiveAgent(CaptureAgent):
    def takeAction(self, gameState, action):
        self.lastAction = action
        self.lastGameState = gameState


    def sum_cal(self, x, y):
        tot = 0
        for i in self.entries:
            tot += self.getMazeDistance((x, y), i)
        return tot


    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.entries = []
        walls = gameState.getWalls()
        mapWidth = walls.width

        mapHeight = walls.height
        if self.red:
            midline_X = math.floor(mapWidth / 2) - 1
        else:
            midline_X = math.floor(mapWidth / 2)
        for midline_Y in range(0, mapHeight):
            if not walls[midline_X][midline_Y]:
                temp_loc = (midline_X, midline_Y)
                self.entries.append(temp_loc)

        minDist_to_entries = 10000000
        self.guardPoint = (0, 0)

        if self.red:
            start_x = 0
        else:
            start_x=midline_X

        if self.red:
            end_x = midline_X + 1
        else:
            end_x=mapWidth

        for x in range(start_x, end_x):
            for y in range(mapHeight):
                if not walls[x][y]:
                    dist_to_entries = self.sum_cal(x, y)
                    now_loc = (x, y)
                    if dist_to_entries < minDist_to_entries:
                        minDist_to_entries = dist_to_entries
                        self.guardPoint = now_loc
        self.lastGameState = None
        self.lastAction = None
        self.target = None


    def chooseAction(self, gameState):

        actionList = gameState.getLegalActions(self.index)
        values = []
        for i in actionList:
            values.append(self.evaluate(gameState, i))
        peakRes = max(values)

        optAct = []
        min_len = min(len(actionList), len(values))
        for x in range(min_len):
            if values[x] == peakRes:
                optAct.append(actionList[x])

        select_action = random.choice(optAct)
        self.takeAction(gameState, select_action)
        return select_action

    def getSuccessor(self, gameState, action):

        nextState = gameState.generateSuccessor(self.index, action)
        position = nextState.getAgentState(self.index).getPosition()
        if position == util.nearestPoint(position):
            return nextState
        else:
            return nextState.generateSuccessor(self.index, action)

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        return self.getFeatures(gameState, action) * self.getWeights(gameState, action)


    def min_dist(self, pos, invaders):
        temp_list = []
        for i in invaders:
            temp_list.append(self.getMazeDistance(pos, i.getPosition()))
        return min(temp_list)



    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # compute when we defense / offence
        features['act_defense'] = 1
        if myState.isPacman:
            features['act_defense'] = 0

        # calsulate distance to invaders
        enemies = []
        for i in self.getOpponents(successor):
            enemies.append(successor.getAgentState(i))

        invaders = []
        for enemy in enemies :
            if enemy.getPosition() is None:
                continue
            if enemy.isPacman:
                invaders.append(enemy)

        features['num_invader'] = len(invaders)
        if len(invaders) > 0:
            self.target = None
            dist = self.min_dist(myPos, invaders)
            agentState = gameState.getAgentState(self.index)
            if dist > 3:
                features['dist_invader'] = dist
            elif agentState.scaredTimer <= 0:
                features['dist_invader'] = dist
            else:
                features['dist_invader'] = -dist
        else:
            if self.lastGameState:
                lastFoods = self.getFoodYouAreDefending(self.lastGameState).asList()
                currentFood = self.getFoodYouAreDefending(gameState).asList()
                difference = set(lastFoods).difference(currentFood)

                if len(difference) > 0:
                    # Find the food that is closest to the missing food
                    missingFood = difference.pop()
                    nextFood = None
                    dist_min = 1000000
                    for food in currentFood:
                        if dist_min > self.getMazeDistance(food, missingFood):
                            dist_min = self.getMazeDistance(food, missingFood)
                            nextFood = food
                    self.target = nextFood

            if self.target is None:
                features['dist_protectPoint'] = self.getMazeDistance(myPos, self.guardPoint)
            else:
                features['dist_target'] = self.getMazeDistance(myPos, self.target)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features

    def getWeights(self, gameState, action):
        return {
            'num_invader': -1000,
            'stop': -100,
            'reverse': -2,
            'act_defense': 100,
            'dist_invader': -10,
            'dist_protectPoint': -10,
            'dist_target': -10,
        }

# Defensive agent with Heuristic Function used in preliminary submission
class DefenceAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        width, height = gameState.data.layout.width, gameState.data.layout.height
        self.exploration = {}
        for x in range(width):
            for y in range(height):
                if not gameState.hasWall(x, y) and self.red == gameState.isRed((x, y)):
                    self.exploration[(x, y)] = 0

    def evaluate(self, gameState, action, scared):
        gameState = GameState(gameState)

        enemyIndices = gameState.getBlueTeamIndices(
        ) if self.red else gameState.getRedTeamIndices()
        enemyPos = [gameState.getAgentPosition(
            ind) for ind in enemyIndices if gameState.getAgentPosition(ind) is not None]

        gameState = gameState.generateSuccessor(self.index, action)
        if gameState.getAgentState(self.index).isPacman:
            return -1e9

        if (not enemyPos) and action == 'Stop':
            return 0.

        self_pos = gameState.getAgentPosition(self.index)
        score = 0.
        if self.red:
            foodPos = gameState.getRedFood().asList()
        else:
            foodPos = gameState.getBlueFood().asList()
        for pos in enemyPos:
            distance = self.distancer.getDistance(self_pos, pos)
            if not scared:
                score += .9 ** distance * 1000.
            else:
                score -= .9 ** distance * 1000.
        for pos in foodPos:
            distance = self.distancer.getDistance(self_pos, pos)
            score += .9 ** distance * 10.
        for pos in self.exploration:
            distance = self.distancer.getDistance(self_pos, pos)
            score += .9 ** distance * self.exploration[pos]
        return score

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        gameState = GameState(gameState)

        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        scared = gameState.getAgentState(self.index).scaredTimer > 0
        values = [self.evaluate(gameState, a, scared) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        bestAction = random.choice(bestActions)
        successor = self.getSuccessor(gameState, bestAction)
        successor_pos = successor.getAgentPosition(self.index)
        for pos in self.exploration:
            if util.manhattanDistance(successor_pos, pos) <= SIGHT_RANGE:
                self.exploration[pos] = 0
            else:
                self.exploration[pos] += 1
                self.exploration[pos] = min(10, self.exploration[pos])
        return bestAction

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

###### MCT Implementation: 


#########################
# MCT Node #
#########################
class MCTNode:
    def __init__(self, gameState, simulationStep= 6):
        
        self.children = None
        self.parent = None
        self.nodeVisitedCount = 0
        self.simluationStep = simulationStep
        self.gameState = gameState
        self.reward = 0.0
        
        

    def __str__(self):
        str = 'reward: %f, visited: %d times' %(self.reward, self.nodeVisitedCount)
        return str

    def __eq__(self, other):
        if other == None: 
          return False
        if not self.simluationStep == other.simluationStep: 
          return False
        if not self.gameState == other.gameState: 
          return False
        if not self.nodeVisitedCount == other.nodeVisitedCount: 
          return False
        if not self.reward == other.reward: 
          return False
        if not self.children == other.children: 
          return False
        if not self.parent == other.parent: 
          return False
        return True

    def makeParent(self, parentNode):
        self.parent = parentNode

    def addChildren(self, childNode):
        
        childNode.makeParent(self)
        if self.children is not None:
            self.children.append(childNode)
        else:
            self.children = [childNode]

    def nodeCounterUpdate(self, passedReward):
        self.nodeVisitedCount += 1
        self.reward = self.reward + passedReward
        

    def nodeVisitedBoolStatus(self):
        return self.nodeVisitedCount > 0

    def nodeAverageReward(self):
        vs = self.reward/float(self.nodeVisitedCount)
        return vs

    def getPrevState(self):
        if self.parent is not None:
            return self.parent.gameState
        else:
            return None

#####################
# MCT Defensive Agent #
#####################
class MctsDefensiveAgent(CaptureAgent):



    def evaluate(self, gameState, action):
        weights = self.getWeights(gameState, action)
        features = self.getFeatures(gameState, action)
        
        return features * weights

    def getFeatures(self, gameState, action):
        
        
        successor = self.getSuccessor(gameState, action)
        features = util.Counter()
        reverse = self.fetchCurrToPrevStateAction(gameState)
        myState = successor.getAgentState(self.index)
        
        invaderDistance = self.enemyPacmanDistance(successor)
   
        if action == Directions.STOP:
            features['stop'] = 1
       
        if not myState.isPacman:
            features['onDefense'] = 1

        if action == reverse:
            features['reverse'] = 1

        if len(invaderDistance) > 0:
            features['numInvaders'] = len(invaderDistance)
            features['invaderDistance'] = min(invaderDistance)

        features['foodDefending'] = self.mySideFoodCount(successor)
        features['distanceToBoundary'] = self.agentDistanceFromBoundary(successor)

        return features

    def getWeights(self, gameState, action):
        
        weights = util.Counter()
        weights['invaderDistance'] = -10
        weights['stop'] = -20
        weights['onDefense'] = 100
        weights['reverse'] = -20
       
      

        if len(self.currAgentToEnemyGhostDistance(gameState)) == 0:
            weights['distanceToStart'] = 10
            weights['distanceToBoundary'] = -10
            

        else:
            weights['foodDefending'] = 50
            weights['numInvaders'] = -1000
            
            

        return weights


    def getSuccessor(self, gameState, action):
        
        listSuccessor = gameState.generateSuccessor(self.index, action)
        sucessorPosition = listSuccessor.getAgentState(self.index).getPosition()
        if sucessorPosition != nearestPoint(sucessorPosition):
            return listSuccessor.generateSuccessor(self.index, action)
        else:
            return listSuccessor

    def successorsStates(self, gameState):
        
        ListLegalActions = gameState.getLegalActions(self.index)
        ListLegalActions.remove(Directions.STOP)
        successorsStatesList = [self.getSuccessor(gameState, unitAction) for unitAction in ListLegalActions]
        return successorsStatesList


    def registerInitialState(self, gameState):
        

        self.start = gameState.getAgentPosition(self.index)
        self.midHorizontal = 0

        if self.red:
          self.midHorizontal =  gameState.data.layout.width//2-1
        else:
          self.midHorizontal =  gameState.data.layout.width//2

        # self.midX = gameState.data.layout.width//2-1 if self.red else gameState.data.layout.width//2

        self.boundary = [(self.midHorizontal ,y) for y in range(gameState.data.layout.height) if not gameState.hasWall(self.midHorizontal, y)]

        CaptureAgent.registerInitialState(self, gameState)


    def chooseAction(self, gameState):
        rootNode = self.mctSearch(gameState)
        rootPath = self.findPath(rootNode)
        futureNode = rootPath[1]
        futureState = futureNode.gameState
        chosenAction = futureState.getAgentState(self.index).configuration.direction
        return chosenAction


    
    ##################### Helper Functions #
 


    def enemyPacmanDistance(self, gameState):
        
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        
        agentPosition = gameState.getAgentPosition(self.index)

        invadingAgentsList = [i.getPosition() for i in enemies if i.isPacman and i.getPosition() != None]
        if len(invadingAgentsList) >0:
            enemyDist = [self.getMazeDistance(agentPosition, invaderAgent) for invaderAgent in invadingAgentsList]
            return enemyDist
        else:
            return []


    def mySideFoodCount(self, gameState):
        
        return self.getFoodYouAreDefending(gameState).count()

    def agentDistanceFromBoundary(self, gameState):
        
        agentPosition = gameState.getAgentPosition(self.index)
        boundaryDistance = min([self.getMazeDistance(agentPosition, point) for point in self.boundary])
        return boundaryDistance

    def agentDistanceToStart(self, gameState):
        
        agentPosition = gameState.getAgentPosition(self.index)
        distanceFromStart = self.getMazeDistance(agentPosition,self.start)
        return distanceFromStart



    def getEnemyScaredTimer(self, gameState):
        
        enemiesList = []

        for i in self.getOpponents(gameState):
          enemiesList.append(gameState.getAgentState(i))

        enemyGhostsList = []

        for a in enemiesList:
          if not a.isPacman and a.getPosition() != None:
            enemyGhostsList.append(a)

        localTimeLeft = []
        
        for ghost in enemyGhostsList:
            localTimeLeft.append(ghost.scaredTimer)
        if len(localTimeLeft) > 0:
            return min(localTimeLeft)
        else:
            return 0

       
    def currAgentToEnemyGhostDistance(self, gameState):
        
        enemiesList = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        currentPosition = gameState.getAgentPosition(self.index)

        ghostList = [a.getPosition() for a in enemiesList if not a.isPacman and a.getPosition() != None]
        if len(ghostList) > 0:
            distances = [self.getMazeDistance(currentPosition, ghost) for ghost in ghostList]
            return distances
        else:
            return []


    def currentAgentDown(self, previousState, currentState):
        previousPosition = previousState.getAgentPosition(self.index)
        currentPosition = currentState.getAgentPosition(self.index)

        distance = self.getMazeDistance(currentPosition, previousPosition)
        if currentPosition is self.start and distance > 10:
            return True
        else:
            return False


    def fetchPrevToCurrStateAction (self, gameState):
        return gameState.getAgentState(self.index).configuration.direction

    def fetchCurrToPrevStateAction(self,gameState):
        return Directions.REVERSE[self.fetchPrevToCurrStateAction(gameState)]

  
    ############# #  UCT = UCB + MCT functions #
   
   
    def mctSearch(self, gameState):
        
        initTime = time.time()
        rootNode = MCTNode(gameState)

        
        while True:
            # if the time limit expires, stop doing iterations
            currentLoppTime = time.time()
            if currentLoppTime - initTime > TIME_LIMIT:
                break

            currentNode = self.MctSelect(rootNode)
            if currentNode.nodeVisitedBoolStatus() or currentNode == rootNode:
                currentNode = self.MctExpand(currentNode)
            currentReward = self.MctSimulate(currentNode)
            self.MctBackPropagate(currentNode, currentReward)
            
        return rootNode


    def MctSimulate(self, currentNode, discountFactor = 0.9):
       
        sigmaRewards = 0
        stepCount = currentNode.simluationStep
        currentState = currentNode.gameState

        while stepCount > 0:
            actions = currentState.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            reverseActions = self.fetchCurrToPrevStateAction(currentState)

            if (reverseActions in actions) and (len(actions) > 1):
                actions.remove(reverseActions)

            exponent = currentNode.simluationStep - stepCount
            nextAction = random.choice(actions)
            

            sigmaRewards += discountFactor**exponent * self.evaluate(currentState, nextAction)
            successorState = self.getSuccessor(currentState, nextAction)
            currentState = successorState
            stepCount = stepCount - 1
        return sigmaRewards

    def MctBackPropagate(self, currentNode, R):
        
        while currentNode is not None:
            currentNode.nodeCounterUpdate(R)
            currentNode = currentNode.parent


    def MctSelect(self, currentNode):
        while currentNode.children is not None:
            children = currentNode.children
             
            ucbValues=[]

            for child in children:
              ucbValues.append(self.upperConfidenceBound(child))
            
            maxUcbValue = max(ucbValues)

            candidateNodes=[]
            for child, value in zip(children, ucbValues):
              if value==maxUcbValue:
                candidateNodes.append(child)

            currentNode = random.choice(candidateNodes)
        return currentNode

    def MctExpand(self, currentNode):
        
        successorNodes = self.successorsStates(currentNode.gameState)
        for successorNode in successorNodes:
            child = MCTNode(successorNode)
            currentNode.addChildren(child)
        currentNode = random.choice(currentNode.children)
        return currentNode


    def findPath(self,currentNode):
        path = [currentNode]
        while currentNode.children is not None:
            children = currentNode.children
            
            ucbValues= []

            for child in children:
              ucbValues.append(self.upperConfidenceBound(child))
            maxUcbValue = max(ucbValues)


            candidateNodes = []
 
            for child, value in zip(children, ucbValues):
              if value == maxUcbValue:
                candidateNodes.append(child)


            currentNode = random.choice(candidateNodes)
            path.append(currentNode)
        return path

    def upperConfidenceBound(self, currentNode ):
        
        if currentNode.nodeVisitedBoolStatus():
            exploitationFactor = currentNode.nodeAverageReward()
            explorationFactor = math.sqrt( (2 * math.log(currentNode.parent.nodeVisitedCount)) / currentNode.nodeVisitedCount) ** (1.41)
            upperConfidenceBound = exploitationFactor + explorationFactor
        else:
            upperConfidenceBound = math.inf # default value if the node is not visited
        return upperConfidenceBound




################################################################################
#   SWITCH AGENT (THIS AGENT SWITCHES AGENTS INTO ATTACK AND DEFENCE MODE)
################################################################################

# Switch agent for deffensive agent, which switches from defensive to offensive agent
# when its own capsule was eaten
class DefensiveAgentWithOffensive(CaptureAgent):
    def __init__(self, _index):
        super().__init__(_index)
        self.defense = DefensiveAgent(_index)
        self.attack = OffensiveAstar(_index)
        
    def registerInitialState(self, gameState):
        self.defense.registerInitialState(gameState)
        
        self.attack.registerInitialState(gameState)
        

    def chooseAction(self, gameState):
        myState = gameState.getAgentState(self.index)

        if myState.scaredTimer > 0:
            return self.attack.chooseAction(gameState)
        
        else:
            return self.defense.chooseAction(gameState)


# Switch agent that switches between offence and defence agent during the game
DEFENCE = 1
ATTACK = 0
class OffenceAndDefence(CaptureAgent):
    def __init__(self, _index):
        super().__init__(_index)
        self.defense = DefensiveAgent(_index)
        self.attack = OffensiveAstar(_index)
        self.mode = ATTACK
        self.myOtherAgent = None
        if _index >= 2 :
            self.mode = DEFENCE
        self.prevPositions = []
        self.prevMyFood= []
        self.lastFood= None
        self.mid = []
        self.start = None
        self.foodStart = 0
        
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.red = gameState.isOnRedTeam(self.index)
        self.registerTeam(self.getTeam(gameState))
        self.start = gameState.getAgentState(self.index).getPosition()

        if self.red:
            if self.index == 0:
                self.myOtherAgent = 2
            else:
                self.myOtherAgent = 0
        else:
            if self.index == 1:
                self.myOtherAgent = 3
            else:
                self.myOtherAgent = 1

        if self.red:
            self.mid =  [index for index in range(0, int(gameState.data.layout.width /2))]
        else:
            self.mid =  [index for index in range(int(gameState.data.layout.width/2), gameState.data.layout.width)]
        self.prevMyFood = self.getFoodYouAreDefending(gameState).asList()
        self.foodStart = len(self.prevMyFood)
        self.defense.registerInitialState(gameState)
        self.attack.registerInitialState(gameState)

    # Calculate nearest ghost distance
    def getNearestGhostDistance(self, currentPosition, gameState):
        opponents = []
        for i in self.getOpponents(gameState):
            opponents.append(gameState.getAgentState(i))

        ghost_opponent = []
        for opponent in opponents:
            if not opponent.isPacman and opponent.getPosition() != None:
                ghost_opponent.append(opponent)

        if ghost_opponent != None and len(ghost_opponent) > 0:
            distances = []
            for g in ghost_opponent:
                if g.configuration != None:
                    distances.append(self.getMazeDistance(
                        currentPosition, g.getPosition()))
            if len(distances) > 0:
                return min(distances)
        return None
                
    # Calculate nearest pacman distance
    def getNearestPacmanDistance(self, position, gameState):
        opponents = []
        for i in self.getOpponents(gameState):
            opponents.append(gameState.getAgentState(i))

        pacman_opponent = []
        for opponent in opponents:
            if opponent.isPacman and opponent.getPosition() != None:
                pacman_opponent.append(opponent)

        if pacman_opponent != None and len(pacman_opponent) > 0:
            distances = []
            for g in pacman_opponent:
                if g.configuration != None:
                    distances.append(self.getMazeDistance(
                        position, g.getPosition()))
            if len(distances) > 0:
                return min(distances)
        return None
    
    # Caculate nearest food distance
    def getNearestFoodDistance(self, position, gameState):
        food = self.getFood(gameState).asList()
        distance = [self.getMazeDistance(position, f) for f in food]
        return min(distance)

    # Check whther food is eaten at a previous step
    def lastEatenFood(self, gameState):
        currentFoodLeft = self.getFoodYouAreDefending(gameState).asList()
        if len(currentFoodLeft) == len(self.prevMyFood):
            return None
        
        for food in self.prevMyFood:
            if not food not in currentFoodLeft:
                return food

    
    def chooseAction(self, gameState):
        myState = gameState.getAgentState(self.index)
        otherState = gameState.getAgentState(self.myOtherAgent)
        myPosition = gameState.getAgentState(self.index).getPosition()
        otherPosition = gameState.getAgentState(self.myOtherAgent).getPosition()
        eatenFood =self.lastEatenFood(gameState)
        if eatenFood:
            self.lastFood = eatenFood
        self.prevMyFood = self.getFoodYouAreDefending(gameState).asList()
        enemyState = [gameState.getAgentState(index) for index in self.getOpponents(gameState)]
        enemyPacman = 0
        for enemy in enemyState:
            if enemy.isPacman :
                enemyPacman += 1

        self.mode = ATTACK

        # Defend if there is pacman nearby
        myNearestPacmanDistance = self.getNearestPacmanDistance(myPosition, gameState)
        if myNearestPacmanDistance:
            otherNearestPacmanDistance =  self.getNearestPacmanDistance(otherPosition, gameState)

            if otherNearestPacmanDistance and  myNearestPacmanDistance < otherNearestPacmanDistance:
                self.mode = DEFENCE
            elif otherNearestPacmanDistance and  myNearestPacmanDistance == otherNearestPacmanDistance and self.index < 2:
                self.mode = DEFENCE
            elif not otherNearestPacmanDistance:
                self.mode = DEFENCE

        # Attack if all opponents are invading and you are losing
        elif self.getScore(gameState) <= 0 and enemyPacman == 2:
            self.mode = ATTACK 

        # Defend if enemy is invading the territory and the agent is close by
        elif self.lastFood:
            for enemy in enemyState:
                if enemy.isPacman :
                    myDistance = self.getMazeDistance(myPosition, self.lastFood)
                    if myDistance < self.getMazeDistance(otherPosition, self.lastFood):
                        if myState.isPacman : 
                            path  = self.attack.pathToLastEatenFood(gameState, self.lastFood)
                            if len(path) > 0: return path[0]
                        self.mode = DEFENCE
                    elif myDistance == self.getMazeDistance(otherPosition, self.lastFood) and self.index < 2:
                        if myState.isPacman : 
                            path  = self.attack.pathToLastEatenFood(gameState, self.lastFood)
                            if len(path) > 0: return path[0]
                        self.mode = DEFENCE

        # If winning, defend when
        if self.getScore(gameState) > 1:
            # all opponents invading
            if enemyPacman == 2:
                if self.lastFood:
                    myDistance = self.getMazeDistance(myPosition, self.lastFood)
                    if myState.isPacman  and myDistance < self.getMazeDistance(self.start, self.lastFood): 
                        path  = self.attack.pathToLastEatenFood(gameState, self.lastFood)
                        if len(path) > 0: return path[0]
                self.mode = DEFENCE
            # an opponent is invading and eating more than the score
            if enemyPacman == 1 and self.getScore(gameState) < (self.foodStart - len(self.prevMyFood)):
                if self.lastFood:
                    myDistance = self.getMazeDistance(myPosition, self.lastFood)
                    if myState.isPacman  and myDistance < self.getMazeDistance(self.start, self.lastFood): 
                        path  = self.attack.pathToLastEatenFood(gameState, self.lastFood)
                        if len(path) > 0: return path[0]
                self.mode = DEFENCE
            # no invader and the agent's index is 0 or 1
            if enemyPacman == 0 and self.index < 2:
                if myState.isPacman and self.lastFood: 
                    path  = self.attack.pathToLastEatenFood(gameState, self.lastFood)
                    if len(path) > 0: return path[0]
                self.mode = DEFENCE


        if self.mode == ATTACK:
            return self.attack.chooseAction(gameState)
        elif self.mode == DEFENCE:
            return self.defense.chooseAction(gameState)




######################################################################################
#   THIS IS A HELPER CLASS FOR DEFININING MULTIPLE OFFENCE PROBLEM FOR OffensiveAstar agent
######################################################################################

# REFERENCED FROM : searchAgent.py in project 1


class PositionSearchProblem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.
    The state space consists of (x,y) positions in a pacman game.
    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.
        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None:
            self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                # @UndefinedVariable
                if 'drawExpandedCells' in dir(__main__._display):
                    __main__._display.drawExpandedCells(
                        self._visitedlist)  # @UndefinedVariable

            return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
          As noted in search.py:
              For a given state, this should return a list of triples,
          (successor, action, stepCost), where 'successor' is a
          successor to the current state, 'action' is the action
          required to get there, and 'stepCost' is the incremental
          cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None:
            return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += self.costFn((x, y))
        return cost


class SearchFoodCapusule(PositionSearchProblem):
    """
     The goal state is to find all the food
    """

    def __init__(self, gameState, agent, agentIndex=0, costFn=lambda x: 1):
        self.walls = gameState.getWalls()
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.costFn = costFn
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

        # Store the food for later reference
        self.food = agent.getFood(gameState).asList()
        self.capsule = [agent.getCapsules(gameState)]

        self.goal = self.food + self.capsule

    def isGoalState(self, state):
        # the goal state is the position of food or capsule
        return state in self.goal


class SearchEscape(PositionSearchProblem):
    """
    Used to escape from enermy
    """

    def __init__(self, gameState, agent, agentIndex=0, costFn=lambda x: 1):
        self.walls = gameState.getWalls()
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.costFn = costFn
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

        self.my_boundary = []

        if agent.red:
            end = int( (gameState.data.layout.width /2 ) - 1 )


        else:
            end = int(gameState.data.layout.width / 2 )


        b = [(end, j) for j in range(gameState.data.layout.height)]
        for pos in b:
            if not gameState.hasWall(pos[0],pos[1]):
                self.my_boundary.append(pos)

        self.capsule = agent.getCapsules(gameState)

        self.goal = self.my_boundary + [self.capsule]

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        # the goal state is the boudary of home or the positon of capsule
        return state in self.goal

class SearchHome(PositionSearchProblem):
    """
    Used to escape from enermy
    """

    def __init__(self, gameState, agent, agentIndex=0, costFn=lambda x: 1):
        self.walls = gameState.getWalls()
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.costFn = costFn
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

        self.my_boundary = []

        if agent.red:
            end = 1
        else:
            end = gameState.data.layout.width - 2


        self.my_boundary = [(end, j) for j in range(gameState.data.layout.height)]

        self.goal = self.my_boundary 

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        # the goal state is the boudary of home or the positon of capsule
        return state in self.goal


class SearchCapsule(PositionSearchProblem):
    """
    Used to search capsule
    """

    def __init__(self, gameState, agent, agentIndex=0, costFn=lambda x: 1):
        self.walls = gameState.getWalls()
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.costFn = costFn
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

        self.capsule = agent.getCapsules(gameState)
        self.goal = self.capsule

    def isGoalState(self, state):
        # the goal state is the location of capsule
        return state in self.goal

class SearchLastEatenFood(PositionSearchProblem):
    """
    Used to search capsule
    """

    def __init__(self, gameState, agent, agentIndex=0, costFn=lambda x: 1):
        self.walls = gameState.getWalls()
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.costFn = costFn
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

        self.goal = []

    def setGoal(self, state):
        self.goal = [state]

    def isGoalState(self, state):
        # the goal state is the location of capsule
        return state in self.goal
