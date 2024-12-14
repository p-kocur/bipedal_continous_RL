import gymnasium as gym

import torch.nn as nn
import torch
from torch.distributions.normal import Normal
import random
import numpy as np

class NNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.simple_nn = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier_p = nn.Sequential(
            nn.Linear(32, 4)
        )
        
        log_std = -0.5 * np.ones(4, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
    def forward(self, s):
        x_1 = self.simple_nn(s)
        return Normal(self.classifier_p(x_1), torch.exp(self.log_std))
    
class NNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.simple_nn = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier_v = nn.Sequential(
            nn.Linear(32, 1))
        
    def forward(self, s):
        x_1 = self.simple_nn(s)
        return self.classifier_v(x_1)
    
def actor_loss(p, adv, distribution, eps):
    loss = -torch.min(p*adv, torch.clamp(p, 1-eps, 1+eps)*adv).mean()
    entropy_loss = -distribution.entropy().mean()
    return loss + entropy_loss * 0.01

def critic_loss(y, v):
    loss = (y-v)**2
    return loss

def ppo(n_episodes=12000, gamma=0.99, Ne=4, eps=0.2):
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array").unwrapped
    actor = NNActor()
    critic = NNCritic()
    a_optimizer = torch.optim.Adam(actor.parameters())
    c_optimizer = torch.optim.Adam(critic.parameters())
    
    for i in range(n_episodes):
        print(i)
        if i % 100 == 0:
            torch.save(actor.state_dict(), "one_body_problem/models/model_ppo.pth")
            play()
            
        s_t, _ = env.reset(seed=123) 
        s_t = torch.tensor(s_t, dtype=torch.float32)
        
        for j in range(1600):
            distribution = actor(s_t)
            actions = distribution.sample()
            a_log_prob = distribution.log_prob(actions).sum(axis=-1)
            v_s_t = critic(s_t)
                
            s_t_1, r, terminated, _, _ = env.step(actions.detach().numpy())
            s_t_1 = torch.tensor(s_t_1, dtype=torch.float32)
            if j > 1598:
                r -= 100
                terminated = True
                
            
            actor_beta = NNActor()
            
            actor_beta.load_state_dict(actor.state_dict())
            for e in range(Ne):
                if terminated:
                    adv = r - v_s_t
                    y = r
                else:
                    with torch.no_grad():
                        v_s_t_1 = critic(s_t_1)
                    adv = r + gamma*v_s_t_1 - v_s_t
                    y = r + gamma*v_s_t_1
                
                a_optimizer.zero_grad()
                with torch.no_grad():
                    b_distribution = actor_beta(s_t)
                p = torch.exp(a_log_prob - b_distribution.log_prob(actions).sum(axis=-1).detach())
                a_loss = actor_loss(p, adv.detach(), distribution, eps)
                a_loss.backward()
                a_optimizer.step()
                    
                c_optimizer.zero_grad()
                c_loss = critic_loss(y, v_s_t)
                c_loss.backward()
                c_optimizer.step()
                
                distribution = actor(s_t)
                a_log_prob = distribution.log_prob(actions).sum(axis=-1)
                v_s_t = critic(s_t)
            
            if terminated:
                break
            
            s_t = s_t_1.clone()
     
    torch.save(actor.state_dict(), "one_body_problem/models/model_ppo.pth")
    
    
def play(render="rgb_array"):
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode=render).unwrapped
    r_sum = 0
    for _ in range(10):
        s_t, _ = env.reset(seed=123)
        s_t = torch.tensor(s_t, dtype=torch.float32)
        net = NNActor()
        net.load_state_dict(torch.load("one_body_problem/models/model_ppo.pth", weights_only=True))
        terminated = False
        
        for _ in range(2000):
            actions = net(s_t).sample()
            s_t_1, r, terminated, _, _ = env.step(actions.detach().numpy())
            r_sum += r
            if terminated:
                break
            s_t = torch.tensor(s_t_1, dtype=torch.float32)
    print(r_sum/10)

