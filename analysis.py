# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.


def question2():
    answerDiscount = 0.9
    answerNoise = 0 #chosen var
    return answerDiscount, answerNoise


# answerDiscount - the bigger this is the more a reward is valued deeper into the iteration cycle
# answerNoise - Just like q2 showed the smaller the value the more the intended path is followed the higher the more likelihood that a random action is done
# answerLivingReward - just like name implies, with a larger value, the longer agent stays alive the more reward it receives

def question3a():
    answerDiscount = 0.6 #value the longterm but not too much to not go to 10
    answerNoise = 0 #do not go where not intend no exploratory paths
    answerLivingReward = -1 #risk the ledge no regard for life
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


# cannot get this but is possible according to output
# ive tried a lot of different combinations for the 3 variables and I am not sure why it is not working
# ive played around with variables that are close to what I currently have but this is the closest to the
# solution I have made so far 
def question3b():
    # answerDiscount = .871
    # answerNoise = 0.5
    # answerLivingReward = -0.5
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'



def question3c():
    answerDiscount = 1 # have a longer term sol in mind
    answerNoise = 0 #avoid disttractions
    answerLivingReward = -1 #risk cliff dont care how long alive
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3d():
    answerDiscount = 1 #care significantly about longer term sol since we want 2nd exit
    answerNoise = 0.5 # slightly care for path noise
    answerLivingReward = 0 # care about cliff but dont want to stay alive for ever
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3e():
    answerDiscount = 0 #indifferent about 
    answerNoise = 0 # indifference in path noise
    answerLivingReward = 10 # want to definitely stay alive indefinitely
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question8():
    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE'
    #return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
