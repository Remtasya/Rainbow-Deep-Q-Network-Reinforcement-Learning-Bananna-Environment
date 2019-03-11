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

Python 3 with the packages in the following repository: https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation. The ml-agents package via the installation instructions https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
Bananna Unity enviroment


DQN

#### Double DQN
Note that we're selecting the action with the maximum estimated value in Q update rule. Since these estimates are likely to be noisey, this max is likely to overestimate the value of the state value. One small trick that helps this a little bit is to use a slightly different model to select versus evaluate the maximum action value. In this implementation we use the learned network to select the best action, and then use the Q-target network to evaulate the value of said action.

#### Prioritised Experience replay
Note that although all state-action-reward tuples are saved in the experience buffer, some of these may be more valuable for learning that others. For example an agent may have plenty of experiences with the starting state but relatively little which more rare states. In this algorithm we use how 'surprising' an observed state-action value is as a measure of how 'useful' learning from it is. More formally we use the absolute value of the TD-update.

#### Duel DQN
Note that under the state-action-reward Q learning paradim each timestep contributes to the learning of only one state-action pair. This is despite the fact that for many states all of the action values are likely to be very similar, and learning from one action value ought to be transfered to others. This motivates the idea of the Duel DQN. It works by using a deep neural network artitecture that forces the Q values to be learned as the sum of the state value and the action advantage (which represents how much better one action is over another in a given state). Note however that the equation Qsa = Vs + Asa has too many degrees of freedom, and so we instead use Qsa = Vs + (Asa - max_a(Asa)), using the fact that an optimal policy will always choose the best action, leading to it having 0 advantage.








Additions that might improve the algorithm further
