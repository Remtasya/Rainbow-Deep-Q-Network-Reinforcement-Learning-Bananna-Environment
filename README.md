# DRLND-project-1-navigation
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


### Summary
This is the repository for the 1st project of the Deep reinforcement learning nanodegree program. The requiremrnt is to train an agent to navigate the Bananna Enviroment with an average score greater than 13 over 100 episodes. This repository provides the code to acheive this in 1000 episodes by using a DQN (Deep-Q Network) with the modifications of Double DQN, Prioritised experience replay, and Duel DQN.

### Enviroment
This Unity enviroment requires an agent to navigate a large square in order to collect banannas.

![Trained Agent][image1]

The task is episodic with no set termination (although we will terminate after 500 timesteps).

#### State space
A state is represented by a vector of 37 dimensions, which contains infomation about the agent such as its velocity and forward-ray based object detection.

#### Action space
There are four possible actions - move up, move down, turn left, turn right.

#### Reward
Collecting a yellow bananna provides +1 reward while collecting a blue bananna provides -1 reward.

### Dependencies
In order to run this code you will require:

1.  Python 3 with the packages in the following repository: https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation, including pytorch.

2.  The ml-agents package, which can be the installed following the following instructions: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

3.  The Bananna Unity enviroment specific to your operating system, which can be found here: https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation. After cloning this enviroment please replace the Bananna Folder with the one appropriate to your operating system, as well as change it's path when loaded at the begining of the script

### How to run the repository


#### Watching a random agent
To confirm the enviroment is set up correctly I recommend running the random_agent.ipynb notebook to observe a randomly-acting agent.

#### How to train
To run the code from scratch simply open the train_agent.ipynb notebook and run the code.

#### How to test
To test a pre-trained agent (I've included one in this repository) simply open the test_agent.ipynb notebook and run the code.

### What files are included

#### ipynb files
As stated above train_agent.ipynb and test_agent.ipynb are intuitive files that are all that's required to walk you through training or tested this agent. If however you would like to change the code (such as to specify a different model architecture, or hyperparameter selection) then you may find the following descriptions useful:

### report.pdf
This describes the implementation in detail beyond the scope of this readme. read this file if you'd like to know more about: the model architecture, the DQN algorithm itself and the hyperparameters used, the modifications made such as Duel DQN and prioritised replay, or the suggestions for further work.

### model.py
This is a simple python script that specifies the pytorch model architecture used. For this project the architecture is quite straightforward, a simple feed-forward neural network with linear layers. Added complexity however comes from the Duel-DQN implementation, which causes the computational graph to fork into state values and state-action values before recombining.

### dqn_agent.py
This file contains all of the functions required for the agent to store experience, sample and learn from it, and select actions in the enviroment. There is also a lot of extra complexity in this coode due to the prioritised experience replay and double DQN implementations.

### Agent design and implementation

DQN

#### Double DQN
Note that we're selecting the action with the maximum estimated value in Q update rule. Since these estimates are likely to be noisey, this max is likely to overestimate the value of the state value. One small trick that helps this a little bit is to use a slightly different model to select versus evaluate the maximum action value. In this implementation we use the learned network to select the best action, and then use the Q-target network to evaulate the value of said action.

#### Prioritised Experience replay
Note that although all state-action-reward tuples are saved in the experience buffer, some of these may be more valuable for learning that others. For example an agent may have plenty of experiences with the starting state but relatively little which more rare states. In this algorithm we use how 'surprising' an observed state-action value is as a measure of how 'useful' learning from it is. More formally we use the absolute value of the TD-update.

#### Duel DQN
Note that under the state-action-reward Q learning paradim each timestep contributes to the learning of only one state-action pair. This is despite the fact that for many states all of the action values are likely to be very similar, and learning from one action value ought to be transfered to others. This motivates the idea of the Duel DQN. It works by using a deep neural network artitecture that forces the Q values to be learned as the sum of the state value and the action advantage (which represents how much better one action is over another in a given state). Note however that the equation Qsa = Vs + Asa has too many degrees of freedom, and so we instead use Qsa = Vs + (Asa - max_a(Asa)), using the fact that an optimal policy will always choose the best action, leading to it having 0 advantage.








Additions that might improve the algorithm further
