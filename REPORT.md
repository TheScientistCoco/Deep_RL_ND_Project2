# Project 2: Continuous Control
Yu Tao

## Overview

In this project, a reinforcement learning (RL) agent, a double-jointed arm, was trained to reach target locations.

For this project, the Unity [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment was used. In this environment, a double-jointed arm can move to target locations. A reward of **+0.1** is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of **33** variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between **-1 and 1**.

For this project, you can choose either one of the two separate versions of the Unity environment. The first version contains **a single agent**, the second version contains **20 identical agents**, each with its own copy of the environment. The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience. This task is episodic, and in order to solve the environment, your agent(s) must get an average score of **+30** over **100** consecutive episodes.

## Learning Algorithm

### Deep Deterministic Policy Gradient (DDPG)
This project implements an off-policy method called **Deep Deterministic Policy Gradient**, the details can be found in [this paper](https://arxiv.org/pdf/1509.02971.pdf), written by researchers at Google Deepmind. The DDPG algorithm belongs to the actor-critic methods, that use deep function approximators to learn policies in high-dimensional, continuous action spaces.

The implementation details can be found in the **DDPG_agent.py** file in this repository.

### Actor-Critic Method
Actor-critic methods leverage the strengths of both policy-based and value-based methods. Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents are also more stable than value-based agents, while requiring fewer training samples than policy-based agents.

The implementation details can be found in the **model.py** file in this repository.

### Hyperparameters

The DDPG used the following hyperparameters (details in ddpg_agent.py)

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

### Model Architecture

The model architecture is as follows (details in model.py):

For the actor part, it consists of 3 fully connnected layers:
```
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```

For the critic part, it consists of 3 fully connnected layers:
```

```

The number of the input units of the neural network is 37, corresponding to the state space dimension. The number of the output nodes of the neural network is 4, corresponding to the action space dimension. I built a DDPG with 2 fully-connected (FC) layers with 64 nodes, each followed by a ReLu activation function. The network used the Adam optimizer, and the learning rate was set to 0.0005, with a batch size of 64.


### Plot of Rewards

![DQN score](./images/Score.png)

This model solved the environment in **489** episodes, which meets the requirement that the agent is able to receive an average reward (over 100 episodes) of at least +13. The final model is saved in 'checkpoint.pth'.

## Ideas for Future Work

To improve the performance of the agent, there are several ideas to modify the deep Q-Learning algorithm we have used:
-	**Double DQN**: Deep Q-Learning tends to [overestimate](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf) the action values. Double Q-Learning has been shown to work well in practice to help with [this](https://arxiv.org/abs/1509.06461).
-	**Prioritized Experience Replay**: Deep Q-Learning samples experience transitions uniformly from a replay memory. [Prioritized experienced replay](https://arxiv.org/abs/1511.05952) is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.
-	**Dueling DQN**: Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values for each action. However, by replacing the traditional Deep Q-Network (DQN) architecture with a [dueling architecture](https://arxiv.org/abs/1511.06581), we can assess the value of each state, without having to learn the effect of each action.

Besides, we can also train the agent directly from its observed raw pixels of the environment instead of using the 37 dimensional states. In this case, we can add a series of [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) to extract the spatial features from the pixels. DeepMind already leveraged such method to build the Deep Q-Learning algorithm that learned to play Atari video games.
