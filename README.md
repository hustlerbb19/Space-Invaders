# Overview
The following notebook walks through development of a Double Deep Q-Learning agent to play [Atari Space Invaders](https://gym.openai.com/envs/SpaceInvaders-v0/), taking inspiration from RL tutorials from Jon Krohn's [Deep Learning Illustrated](https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/cartpole_dqn.ipynb) and Aurélien Géron's [Hands On Machine Learning with Scikit-Learn and Tensorflow](https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb). Learning has been scaled to a single machine, trained for 200,000 frames using a memory buffer of 60,000 experiences. Model training was performed ever three frames (per [guidance from the DeepMind team](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), with ε-greedy learning over 20,000 decay steps, or roughly a third of total training. Additional modifications include using Huber loss as objective function for model training, implementatoin of "double" Q-learning (with online model weights copied to training model every 2,000 frames for 100 copy-events total), and inclusion of "spawn" and "respawn" skips over periods of user non-playability at the start of each game epsisode and user regeneration after loss of life (40 and 20 frames, respectively).
## Training
To train our RL agent, we pass a series of game 

To process our image observations, we'll crop our image to the playable field of the Space Invaders game (roughly the space between the bottom of the agent-controlled spaceship and the mid-score area where the "mothership" will appear, downsample our image by 2x (to dimension 92 x 80, convert our observed pixel values to grey-scale, and scale values to [0,1].


Our model architecture is inspired by the [DeepMind Nature publication](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). It consists of three hidden CNN layers (afirst 32-filter CNN layer of 8x8 convolutions passed over our images at stride 4, a 64-filter CNN layer of 4x4 convolutions scanning at stride 2, and lastly second 64-filter CNN layer of 3x3 convolutions proceeding at stride 1) followed by a fully connected 512-node ReLU activation layer, and finally  6-output layer (mapping to the 6 playable actions  of the SpaceInvaders game.

![ANN Arhcitecture](https://raw.githubusercontent.com/hustlerbb19/Space-Invaders/master/DDQN_model_output/ANN_architecture.png)


![img](https://raw.githubusercontent.com/hustlerbb19/Space-Invaders/master/ddqn_model_output/Clipped%20Learning%20Curve.png)
