# Report

This report is designed to cover the following in more detail than the readme:
1.  Theoretical DQN Agent Design
2.  Implementation, Hyperparameters, and Performance
3.  Ideas for Future Improvements

## Theoretical DQN Agent Design

The algorithm used is based on the DQN algorithm described in this paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

<img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aef2add_dqn/dqn.png" alt="DQN diagram" width="550"/>

DQN (Deep-Q-Networks) is an innovative approach in reinforcement learning that effectively combines two seperate fields:

### Q-Learning

In Reinforcement learning, the goal is to have an agent learn how to navigate a new enviroment with the goal of maximising cummulative rewards. One approach to this end is Q-learning, where the agent tries to learn the dynamics of the enviroment indirectly by focusing on estimating the value of each state-action pair in the enviroment. This is acheived over the course of training, using it's experiences to produce and improve these estimates - as the agent encounters state-action pairs more often it becomes more confident in its estimate of their value. 

### Deep Learning

Famous in computer vision and natural language processing, deep learning uses machine learning to make predictions by leveraging vast amounts of training data and a flexible architecture that is able to generalise to previously unseen examples. In DQN we leverage this power for the purpose of predicting Q values, and use the agents experiences within the enviroment as a reusable form of training data. This proves to be a powerful combination thanks to Deep learning's ability to generalise given sufficent data and flexibility.

In addition to vanilla DQN we also make use of the following modifications:

### Double DQN

Note that when updating our Q-values we assume the agent selects the action with the maximum estimated value in the next timestep. However since these action-value estimates are likely to be noisey, taking the max is likely to overestimate their true value. One small trick that reduces this problem a little bit is to use a slightly different model to select versus evaluate the maximum action value. In this implementation we use the current Q-network to select the best action, and then use the Q-target network to evaulate the value of said action, which is essentially a lagged version of the Q-network with weights updated less frequently.

Read more: https://arxiv.org/abs/1509.06461

### Prioritised Experience replay

In order to produce training data we store all state-action-reward tuples as experiences and then sample them randomly each time we update the model. Note though that some of these may be more valuable for learning that others. For example an agent may have plenty of experiences from the starting state but relatively little from more rare states. In this modification we use how 'surprising' an observed state-action value is as a measure of how 'useful' learning from it is, which formally is the absolute difference between the value we observed and what our model predicted is should have been.

Read more: https://arxiv.org/abs/1511.05952

### Duel DQN

Note that under the state-action-reward Q learning paradim each timestep contributes to the learning of only one state-action pair. This is despite the fact that for many states the various action values are likely to be very similar, and learning from one action value ought to be transfered to others, since they all arise from the same state. This motivates the idea of the Duel DQN. It works by using an architecture that forces the Q values to be learned as the sum of the state value and the action advantage (which represents how much better one action is over another in a given state). Note however that the equation Qsa = Vs + Asa has too many degrees of freedom to be learned without restrictions, and so we instead use Qsa = Vs + (Asa - max_a(Asa)), using the fact that an optimal policy will always choose the best action, and thereby using it as a benchmark of having 0 advantage.

Read more: https://arxiv.org/abs/1511.06581

## Implementation and Empirical Results

After ~550 episodes the agent was about to 'solve' the enviroment by attaining an average reward over 100 episodes greater than 13.0.

A plot of score over time is shown below:

<img src="https://github.com/Remtasya/DRLND-project-1-navigation/blob/master/project_images/Bananna_project_results.PNG" alt="Rainbow" width="400"/>

### Hyperparameters
#### Several Hyperparameters were used in this implementation which will be described below:


**n_episodes (int): maximum number of training episodes**
the model was found to converge after ~1000 episodes.

**max_t (int): maximum number of timesteps per episode**
this is useful for envioments that permit infinite exploration as it helps reset the enviroment.

**eps_start (float): starting value of epsilon, for epsilon-greedy action selection**
to encourage early exploration of the state-action space we're using an epsilon-greedy apporach with epsilon starting at 1.

**eps_end (float): minimum value of epsilon**
Epsilon should never reach 0 or else in the limit we might not explore all state-action pairs, so this sets a lower bound.

**eps_decay (float): multiplicative factor (per episode) for decreasing epsilon**
we want epsilon to decay over the course of training as the agent transitions from exploration to exploitation.

**Double_DQN (bool): whether to implement Double_DQN modification**
Non-functional currently, leave as True

**Priority_Replay (bool): whether to implement Priority_Replay modification**
Non-functional currently, leave as True

**Duel_DQN (bool): whether to implement Duel_DQN modification**
Non-functional currently, leave as True

**Priority_Replay_Paras (list of e,a,b floats):**
These determine the parameters of the priority modification. e adds some priority to all experiences to prevent over-fitting to a subset of the state-action space. Priorities are raised to the power of a, where a value close to 0 encourages uniform sampling whereas close to 1 emphasises priorities. Lastly the update rule itself becomes biased due to non-uniform sampling, as so must be corrected, with b controling the degree of correction, which is increased over the course of training.

**GAMMA (float): discount rate**
Close to 1 will cause the agent to value all future rewards equally, while close to 0 will cause the agent to prioritise more immediate rewards. Unlike most hyperparameters, this will not only effect convergence but also the optimal policy converged to. For example if an agent must choose between collecting 1 bananna and then waiting 20 timeseteps versus collecting 2 banannas after 20 timesteps, then the optimal policy depends on the reard discount rate. Close to 1 is often best so I chose 0.99.


**LR (float): model hyperparameter - learning rate**
This determines how large the model weight updates are after each learning step. Too large and instability is caused, while too small and the model may never converge. I chose the small 5e-4, since we can increase epsiodes until we reach convergence. 

**BATCH_SIZE (int): model hyperparameter - number of experiences sampled for a model minibatch**
Too low will cause learning instability and poor convergence, too high can cause convergence to local optima. I chose 64 as a default.


**BUFFER_SIZE (int): replay buffer size**
this is the size of the experience buffer, which when exceeded will drop old experiences. This is mainly limited by your available RAM - if you experience issues with RAM try lowering it


*TAU (float): how closely the target-network should track the current network** 
After every learning step the target-network weights are updated closer to the current network, so that the target-network weights are a moving average over time of the current network past weights. i chose a relatively small value (1e-3) although haven't experimented with tuning it.

**UPDATE_EVERY (int): how often to update the network**
How many steps should pass before an update of the current network takes place. I chose every 4 timesteps.

## Ideas for Future Improvements

Additions that might improve the algorithm further are the other 3 modifications of the Rainbow implementation, which acheives state-of-the-art-performance in DQNs.

<img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b3814f1_screen-shot-2018-06-30-at-6.40.09-pm/screen-shot-2018-06-30-at-6.40.09-pm.png" alt="Rainbow" width="400"/>

**Namely these are:**

1.  Learning from multi-step bootstrap targets -  https://arxiv.org/abs/1602.01783
2.  Distributional DQN - https://arxiv.org/abs/1707.06887
3.  Noisy DQN - https://arxiv.org/abs/1706.10295

