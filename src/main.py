import gym
import torch
import numpy as np
from Clases.SessionGenerator import SessionGenerator
from Clases.MlpAgent import MlpAgent

#Create environment
env = gym.make("CartPole-v0")

#Set actor
n_actions = env.action_space.n
agent = MlpAgent(len(env.reset()), 20, n_actions)

if torch.cuda.is_available():
  agent = agent.cuda()

#training loop
n_samples = 10
percentile = 70

bce = torch.nn.BCELoss()
optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)

for i in range(50):
    #sample sessions
    sessions = [SessionGenerator.generate_session(agent, env, n_actions) for _ in range(n_samples)]
    batch_states, batch_actions, batch_rewards = map(np.array,zip(*sessions))

    #choose threshold on rewards
    threshold = np.percentile(batch_rewards,percentile)
    elite_states = torch.tensor(np.concatenate(batch_states[batch_rewards>=threshold])).float()
    
    if torch.cuda.is_available():
      elite_states = elite_states.cuda()

    elite_actions = torch.tensor(np.concatenate(batch_actions[batch_rewards>=threshold]))

    elite_actions = elite_actions.reshape(elite_actions.shape[0], 1)
    one_hot_target = (elite_actions == torch.arange(n_actions).reshape(1, n_actions)).float()
    
    if torch.cuda.is_available():
      one_hot_target = one_hot_target.cuda()

    #report progress
    print("epoch %i \tmean reward=%.2f\tthreshold=%.2f"%(i,batch_rewards.mean(),threshold))
    y_pred = agent.forward(elite_states)
    loss = bce(y_pred, one_hot_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Show our agent in action
SessionGenerator.generate_session(agent, env, n_actions, True)