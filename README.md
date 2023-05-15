See the original project instructions here: http://ai.berkeley.edu/reinforcement.html

This project focuses on the build up to a robust Q-learning model. This problem set also focused on a Gridworld which visualizes the reinforcement learning steps. Underlying these approaches is also a Markov Decision Process model.

Commands of note:

  Value Iteration:
                   * python gridworld.py -a value -i 100 -k 10
                   * python gridworld.py -a value -i 5

  Bridge Crossing Analysis:
                   * python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2
                   
  Q-Learning with ε-Greedy:
                   * python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
                   * python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 1
                   
  Q-Learning and Pacman: 
                   * python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
                   
  Approximate Q-Learning:
                   * python pacman.py -p ApproximateQAgent -x 2000 -n 2010 -l smallGrid
                   * python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
