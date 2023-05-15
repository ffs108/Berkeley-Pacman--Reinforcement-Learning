# qlearningAgents.py
# ------------------
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


from game import *
from featureExtractors import *
from learningAgents import ReinforcementAgent
import util
import random


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter() #same as valIter agent for qVals

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        stateActionPair = (state, action)
        if stateActionPair not in self.values: #if never been
            return 0.0
        else:
            return self.values[stateActionPair]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #the legal action func returns an iterable so we can check if no legal actions
        if len(self.getLegalActions(state)) == 0:
            return 0.0
        qvals = []
        for action in self.getLegalActions(state):
            qvals.append(self.getQValue(state, action))
        return max(qvals) #we find what the max qVal for all legal acts if any
        

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        if len(self.getLegalActions(state)) == 0:
            return None
        actionVals = list()
        curMax = self.getQValue(state, self.getLegalActions(state)[0])
        #print(curMax)
        # we iterate through all qVals in legal acts to find the max
        for action in self.getLegalActions(state):
            curVal = self.getQValue(state,action)
            if curVal > curMax:
                curMax = curVal
        # we iterate through all legal acts again to find which actions lead to them
        for action in self.getLegalActions(state):
            if self.getQValue(state,action) == curMax:
                actionVals.append(action)
        # we randomly choose from the pool of options
        randomChoice = random.choice(actionVals)
        return randomChoice

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return None
        # we take the best policy action
        action = self.getPolicy(state)
        # OR if the epsilon hits True we choose at random from all legal actions
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        curVal = self.getQValue(state,action)
        nextVal = self.getValue(nextState) #minor spelling err 1 - me 0
        # https://www.baeldung.com/cs/epsilon-greedy-q-learning
        # q(s,a) = q(s,a) + α * [reward + γ * maxQ(s{t+1},a) - q(s,a)]
        newQ = curVal + self.alpha * (reward + self.discount * nextVal - curVal)
        #print(newQ)
        self.values[(state, action)] = newQ

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        Qsa = self.weights.__mul__(self.featExtractor.getFeatures(state,action)) #dot product operand for Counter()
        return Qsa

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        curVal = self.getQValue(state,action)
        nextVal = self.getValue(nextState)
        #q(s,a) = reward + γ * maxQ(s{t+1},a) - q(s,a)
        diff = (reward + self.discount * nextVal) - curVal
        for feature in self.featExtractor.getFeatures(state,action):
            wi = self.weights[feature]
            fi = self.featExtractor.getFeatures(state,action)[feature]
            # wi = wi + α * difference * fi(s,a)
            weightUpdate = wi + self.alpha * diff * fi
            self.weights[feature] = weightUpdate


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            #print(self.weights)
            # print('this is the weight: ', end='')
            # print(self.weights, end='\n')
            pass
