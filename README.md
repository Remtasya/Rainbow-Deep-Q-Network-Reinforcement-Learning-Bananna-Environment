# DRLND-project-1-navigation
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif=250x "Trained Agent"
[image2]: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aef2add_dqn/dqn.png=250x "DQN diagram"
[image3]: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b3814f1_screen-shot-2018-06-30-at-6.40.09-pm/screen-shot-2018-06-30-at-6.40.09-pm.png=250x "Rainbow"

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

#### report.pdf
This describes the implementation in detail beyond the scope of this readme. read this file if you'd like to know more about: the model architecture, the DQN algorithm itself and the hyperparameters used, the modifications made such as Duel DQN and prioritised replay, or the suggestions for further work.

#### model.py
This is a simple python script that specifies the pytorch model architecture used. For this project the architecture is quite straightforward, a simple feed-forward neural network with linear layers. Added complexity however comes from the Duel-DQN implementation, which causes the computational graph to fork into state values and state-action values before recombining.

#### dqn_agent.py
This file contains all of the functions required for the agent to store experience, sample and learn from it, and select actions in the enviroment. There is also a lot of extra complexity in this coode due to the prioritised experience replay and double DQN implementations.

### Agent design and implementation

**Details of the agent design can also be found in the report.pdf, but a summary with references is provided here:**

The algorithm used is based on the DQN algorithm described in this paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

![DQN diagram][image2]

DQN (Deep-Q-Networks) is an invovate approach in reinforcement learning that effectively combines two seperate fields:

#### Q-Learning
In Reinforcement learning, the goal is to have an agent learn how to navigate a new enviroment with the goal of maximising cummulative rewards. One approach to this end is Q-learning, where the agent tries to learn the dynamics of the enviroment indirectly by focusing on estimating the value of each state-action pair in the enviroment. This is acheived over the course of training, using it's experiences to produce and improve these estimates - as the agent encounters state-action pairs more often it becomes more confident in its estimate of their value. 

#### Deep Learning
Famous in computer vision and natural language processing, deep learning uses machine learning to make predictions by leveraging vast amounts of training data and a flexible architecture that is able to generalise to previously unseen examples. In DQN we leverage this power for the purpose of predicting Q values, and use the agents experiences within the enviroment as a reusable form of training data. This proves to be a powerful combination thanks to Deep learning's ability to generalise given sufficent data and flexibility.

In addition to vanilla DQN we also make use of the following modifications:

#### Double DQN
Note that when updating our Q-values we assume the agent selects the action with the maximum estimated value in the next timestep. However since these action-value estimates are likely to be noisey, taking the max is likely to overestimate their true value. One small trick that reduces this problem a little bit is to use a slightly different model to select versus evaluate the maximum action value. In this implementation we use the current Q-network to select the best action, and then use the Q-target network to evaulate the value of said action, which is essentially a lagged version of the Q-network with weights updated less frequently.

Read more: https://arxiv.org/abs/1509.06461

#### Prioritised Experience replay
In order to produce training data we store all state-action-reward tuples as experiences and then sample them randomly each time we update the model. Note though that some of these may be more valuable for learning that others. For example an agent may have plenty of experiences from the starting state but relatively little from more rare states. In this modification we use how 'surprising' an observed state-action value is as a measure of how 'useful' learning from it is, which formally is the absolute difference between the value we observed and what our model predicted is should have been.

Read more: https://arxiv.org/abs/1511.05952

#### Duel DQN
Note that under the state-action-reward Q learning paradim each timestep contributes to the learning of only one state-action pair. This is despite the fact that for many states the various action values are likely to be very similar, and learning from one action value ought to be transfered to others, since they all arise from the same state. This motivates the idea of the Duel DQN. It works by using an architecture that forces the Q values to be learned as the sum of the state value and the action advantage (which represents how much better one action is over another in a given state). Note however that the equation Qsa = Vs + Asa has too many degrees of freedom to be learned without restrictions, and so we instead use Qsa = Vs + (Asa - max_a(Asa)), using the fact that an optimal policy will always choose the best action, and thereby using it as a benchmark of having 0 advantage.

Read more: https://arxiv.org/abs/1511.06581

### Further additions
Additions that might improve the algorithm further are the other 3 modifications of the Rainbow implementation, which acheives state-of-the-art-performance in DQNs.

![Rainbow][image3]

**Namely these are:**

Learning from multi-step bootstrap targets -  https://arxiv.org/abs/1602.01783
Distributional DQN - https://arxiv.org/abs/1707.06887
Noisy DQN - https://arxiv.org/abs/1706.10295
