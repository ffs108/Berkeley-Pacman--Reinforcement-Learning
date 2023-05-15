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


import util
from learningAgents import ValueEstimationAgent


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
        "*** YOUR CODE HERE ***"
        # V(s) = max( Σ T(s,a,s') * [R(s,a,s') + γ * V(s')] )
        for curIter in range(self.iterations): #the iter was mentioned this would be the loop
            # I errorneaously had the policy variable set before the looping starts for each iteration
            # this cause policy (or as in the book the util vector) is something that is carried from state to state
            # but should not be reflective through each iteration
            policy = util.Counter() # synthetic π -> this has to be state to actions
            for state in self.mdp.getStates():
                # need to check if terminal state
                if self.mdp.isTerminal(state):
                    continue
                # walking back from OH it makes sense to delegate the loop through the possible actions here to the actionFromVal func
                # it does practically the same behavior as the loop I previously had here and the return action leads to a variale I can
                # use to then call QValueFromAction()
                maxAct = self.computeActionFromValues(state)
                # The bellman update  Σ T(s,a,s') * [R(s,a,s') + γ * V(s')]  happens here aswell so while I could
                #  have recalculated these steps here in the valiter func, reusing of the functions was more optimal
                maxQ = self.computeQValueFromValues(state, maxAct)
                # now the email suggestion made previously makes more and more sense because now we have the synthetic policy of a state
                # matching to the optimal Q value which through more iterations should become more and more accurate while converging 
                policy[state] = maxQ
        #return policy is more of an update to the self.values class attribute & completely got rid ot the argmax() here like suggested
            self.values = policy





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
        # Q(s,a) = R(s,a) + γ Σ P(s' | s,a) * V(s') - https://artint.info/2e/html/ArtInt2e.Ch9.S5.SS2.html
        # QValueFromValues(state,action) = getReward(state,action,nextState) + γ * Σ Transition(state,action) * self.values[state]
        Qval = 0 # this is our returning Q val
        #need to go through the transition states for the parameter state from mdp.py know this is list [nextState,prob]
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
            #start the Σ here - we will continue summing into retval
            reward = self.mdp.getReward(state, action, nextState)
            Qval += reward + self.discount * prob * self.getValue(nextState)
        return Qval
            



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # V* = maxQval*(s, a) -- use to get optimal Q val over all states and return assoc actions
        if len(self.mdp.getPossibleActions(state)) == 0 or self.mdp.isTerminal(state):
            return None
        qVals = util.Counter() # this is going to be similar to the argmax stuff in valIter need actions
        for action in self.mdp.getPossibleActions(state):
            # I had a loop here going through all the transition states and probabilites but I do the same
            # in computeQValuesFromValues so I just called it to avoid redundancy
            optimQ = self.computeQValueFromValues(state, action)
            qVals[action] = optimQ
        mostQAction = (qVals.sortedKeys())[0] #sorted keys returns a list of keys sorted by their values index zero should be highest key
        return mostQAction



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent): #extends class above so funcs still avail
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
        # self.mdp.getStates() is array
        for curIter in range(self.iterations):
            index = (curIter % len(self.mdp.getStates())) #this should ensure that each iteration only focuses on one state & loops back when needed
            state = self.mdp.getStates()[index] 
            if self.mdp.isTerminal(state):
                continue
            maxAct = self.computeActionFromValues(state)
            maxQ = self.computeQValueFromValues(state, maxAct)
            self.values[state] = maxQ


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
        #compute predecessors
        predecessors = self.calcPredecessors()
        #empty priority queue
        priQueue = util.PriorityQueue()
        # for each state push into pQueue the abs(of the diff between thr curval and highest qval)
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            curVal = self.values[state]
            maxAct = self.computeActionFromValues(state)
            maxQ = self.computeQValueFromValues(state, maxAct)
            diff = abs(curVal - maxQ)
            priQueue.push(state, -diff) #negative cause min heap
        #for iteration 
        for curIter in range(self.iterations):
            #if queue empty terminate
            if priQueue.isEmpty():
                break
            state = priQueue.pop()
            if self.mdp.isTerminal(state):
                continue
            #if popped state is not terminal update the self.values val
            maxAct = self.computeActionFromValues(state)
            maxQ = self.computeQValueFromValues(state, maxAct)
            self.values[state] = maxQ
            #for each predecessor of this state
            for pred in predecessors[state]:
                curPredVal = self.values[pred]
                maxPredAct = self.computeActionFromValues(pred)
                maxPredQ = self.computeQValueFromValues(pred, maxPredAct)
                #absolute value of the difference between the current value of p in self.values and the highest Q-value
                predDiff = abs(curPredVal - maxPredQ)
                # If diff > theta, push p into the priority queue with priority -diff 
                if predDiff > self.theta:
                    priQueue.update(pred, -predDiff)


    def calcPredecessors(self):
        #so for every state - its predecessors are states that have a nonzero prob of reaching said state via an action
        #one state can have multiple predecessors so will use list to store later in the counter()
        predecessorMap = util.Counter()
        #predecessorMap = {x: set() for x in predecessorMap.keys()}#dict comprehension https://stackoverflow.com/questions/49358963/how-do-i-initialize-a-dictionary-with-a-list-of-keys-and-values-as-empty-sets-in
        for state in self.mdp.getStates():
            predecessorMap[state] = set()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            #what actions can lead to current state
            for action in self.mdp.getPossibleActions(state):
                #loop to see all states that have a nonzero probability of reaching s by taking some action
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state,action):
                    if prob != 0 and not self.mdp.isTerminal(nextState):
                        predecessorMap[nextState].add(state)
        # for key in predecessorMap.keys():
        #     arrayFormat = predecessorMap[key].split(',')
        #     predecessorMap[key] = set(arrayFormat) #make set to rid of duplicates
        return predecessorMap







