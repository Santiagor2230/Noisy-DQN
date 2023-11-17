from trainings import DeepQLearning
import torch
from pytorch_lightning import Trainer
from display import display_video

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

model = DeepQLearning(
    'Catcher-v0',
    lr = 0.0005,
    sigma = 0.5,
    hidden_size=512,
    samples_per_epoch=1_000,
    a_last_episode = 1_200,
    b_last_episode=1_200
)

trainer = Trainer(
    gpus=num_gpus,
    max_epochs = 2_000,
    log_every_n_steps=1
)

trainer.fit(model)


env = model.env
policy = model.policy
q_net = model.q_net.cuda()
frames = []

for episode in range(10):
  done = False
  obs = env.reset()
  while not done:
    frames.append(env.render(mode='rgb_array'))
    action = policy(obs, q_net)
    obs, _, done, _ = env.step(action)
    

display_video(frames)