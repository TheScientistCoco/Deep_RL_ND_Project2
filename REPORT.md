# Project 1: Navigation
Yu Tao

## Overview

In this project, a reinforcement learning (RL) agent was trained to navigate (and collect bananas!) in a large, square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

The environment is similar to the [Unity's Banana Collector environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector). The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

## Learning Algorithm

### Deep Q-Networks

This project implemented a value-based method called [Deep Q-Networks](https://en.wikipedia.org/wiki/Q-learning). A DQN, or Deep Q-Network, represents the action-value function in a Q-Learning framework as a neural network.

![DQN algorithm](./images/DQN_algorithm.png)

Deep RL use non-linear function approximators (deep neural network) to calculate the action values based directly on observation (state) from the environment. RL is notoriously unstable when neural networks are used to represent the action values (weights oscillate and diverge due to the high correlation between actions and states). Deep Q-Learning algorithm addressed these instabilities by using two key features: (1) Experience Replay; (2) Fixed Q-Targets.

For detailed information on DQN, please look at the original [Deep Q-Learning algorithm paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

### Hyperparameters

The DQN used the following hyperparameters (details in dqn_agent.py)

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size 
GAMMA = 0.995           # discount factor 
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

### Model Architecture

The model architecture is as follows (details in model.py):

![DQN model](./images/Model.png)

The number of the input units of the neural network is 37, corresponding to the state space dimension. The number of the output nodes of the neural network is 4, corresponding to the action space dimension. I built a DQN with 2 fully-connected (FC) layers with 64 nodes, each followed by a ReLu activation function. The network used the Adam optimizer, and the learning rate was set to 0.0005, with a batch size of 64.


### Plot of Rewards

![DQN score](./images/Score.png)

This model solved the environment in **489** episodes, which meets the requirement that the agent is able to receive an average reward (over 100 episodes) of at least +13. The final model is saved in 'checkpoint.pth'.

## Ideas for Future Work

To improve the performance of the agent, there are several ideas to modify the deep Q-Learning algorithm we have used:
-	**Double DQN**: Deep Q-Learning tends to [overestimate](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf) the action values. Double Q-Learning has been shown to work well in practice to help with [this](https://arxiv.org/abs/1509.06461).
-	**Prioritized Experience Replay**: Deep Q-Learning samples experience transitions uniformly from a replay memory. [Prioritized experienced replay](https://arxiv.org/abs/1511.05952) is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.
-	**Dueling DQN**: Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values for each action. However, by replacing the traditional Deep Q-Network (DQN) architecture with a [dueling architecture](https://arxiv.org/abs/1511.06581), we can assess the value of each state, without having to learn the effect of each action.

Besides, we can also train the agent directly from its observed raw pixels of the environment instead of using the 37 dimensional states. In this case, we can add a series of [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) to extract the spatial features from the pixels. DeepMind already leveraged such method to build the Deep Q-Learning algorithm that learned to play Atari video games.
