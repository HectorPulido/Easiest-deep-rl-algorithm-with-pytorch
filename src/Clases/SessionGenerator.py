import gym
import torch
import numpy as np

class SessionGenerator():

    @staticmethod
    def generate_session(agent, env, n_actions, render=False):
        states, actions, total_reward = [], [], 0

        s = env.reset()
        while True:
            state = torch.tensor(s).float()
            if torch.cuda.is_available():
                state = state.cuda()
            proba = agent.forward(state)
            proba = proba.cpu().detach().numpy()

            a = np.random.choice(n_actions, p=proba)
            states.append(s)
            actions.append(a)

            if render:
                env.render()
            s, r, done, _ = env.step(a)

            total_reward += r
            if done:
                break

        return states, actions, total_reward
