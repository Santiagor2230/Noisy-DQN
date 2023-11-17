# Noisy-DQN
Noisy DQN with Prioritized Experience Replay 

# Requirements
gym == 0.17.3

pytorch-lightining == 1.6.0

torch == 2.0.1

# Collab installations
!apt-get install -y xvfb

!pip install pygame gym==0.17.3 pytorch-lightning==1.6.0 pyvirtualdisplay

!pip install git+https://github.com/JY251/stable-baselines3.git

!pip install git+https://github.com/GrupoTuring/PyGame-Learning-Environment

!pip install git+https://github.com/lusob/gym-ple


# Description
Noisy DQN is a continuation of double DQN with Priotirized Experience Replay, in this instance we modify the weights of the neural network by allowing them to have some sort of noise when training this allows us for the neural network to implictly explore the environment by choosing actions that may not give the highest reward making this approach stochastic and not deterministic.

# Game
Catcher

# Architecture
Noisy Double DQN with Priotirized Experience Replay

# optimizer
AdamW

# Loss
smooth L1 loss function

# Video Result



https://github.com/Santiagor2230/Noisy-DQN/assets/52907423/09e6f4a1-32ef-4c00-923f-7fef0124630e

