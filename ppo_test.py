import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
import gymnasium as gym
import numpy as np 



GAMME = 0.99
EPS_CLIP = 0.2
EPOCH = 10
BATCH_SIZE = 128
MAX_EPOCH = 10000
UPDATE_STEPS = 4
LR=3e-4


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.common = nn.Linear(state_size , hidden_size)
        self.hidden_layer = []
        for i in range(4):
            self.hidden_layer.append(nn.Linear(hidden_size, hidden_size))
        
        self.actor  = nn.Linear(hidden_size, action_size)


    def forward(self, x):
        x = F.relu(self.common(x))
        for layer in self.hidden_layer:
            x = F.relu(layer(x))
        
        action_prob = F.softmax(self.actor(x), dim=-1)
        return action_prob

    def predict(self , state):
        action_prob =  self.forward(state)
        dist = Categorical(action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob



class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(Critic, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(16)]
        self.critic = nn.Linear(hidden_size, 1)
    def forward(self, state):
        x = F.gelu(self.input_layer(state))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.critic(x)

    
def compute_returns(rewards,  gamme):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamme * R 
        returns.insert(0 , R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def ppo_update(model, critic , optimizer, optimizer2, state , actions ,
 log_prob_old , returns , advantages , eps_clip, epochs):
    for _ in range(epochs):
        actions_probs = model(state)
        dist = Categorical(actions_probs)
        values = critic(state)
        log_prob_new = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratios = torch.exp(log_prob_new - log_prob_old)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
        critic_loss = F.mse_loss(values.squeeze(), returns)

        optimizer.zero_grad()
        actor_loss = -torch.min(surr1 , surr2).mean() - 0.01 * entropy
        optimizer.step()

        optimizer2.zero_grad()
        critic_loss.backward()
        optimizer2.step()

def visualize_model(model, num_episodes=5):
    """Visualize the trained model performance"""
    env = gym.make('CartPole-v1', render_mode='human')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0
        done = False
        step = 0
        
        print(f"Visualization Episode {episode + 1}")
        
        from numpy.random import choice
        while not done:
            with torch.no_grad():
                action, _ = model.predict(state, deterministic=True)
            state, reward, done, _, _ = env.step(action)
            state = torch.tensor(state, dtype=torch.float32)
            total_reward += reward
            step += 1
            
            env.render()
        
        print(f"Episode {episode + 1} finished with reward: {total_reward}, steps: {step}")
    
    env.close()

def main():

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    model = Actor(state_size , action_size)
    critic = Critic(state_size)

    optimizer = optim.Adam(model.parameters() , lr=LR)
    optimizer2 = optim.Adam(critic.parameters(), lr=LR)
    for episode in range(MAX_EPOCH):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        states, actions , rewards, log_probs_old = [] , [] , [] ,[]
        done = False

        ep_reward = 0

        while not done:
            action, log_prob = model.predict(state.unsqueeze(0))
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs_old.append(log_prob)

            state = torch.tensor(next_state, dtype=torch.float32)
            ep_reward += reward

        returns = compute_returns(rewards , GAMME)
        values = critic(torch.stack(states))
        advantages = returns - values.squeeze().detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states = torch.stack(states)
        actions = torch.tensor(actions)
        log_probs_old = torch.stack(log_probs_old).detach()

        ppo_update(model, critic, optimizer, optimizer2, states , actions, log_probs_old, returns , advantages, EPS_CLIP, EPOCH)

        if episode % 10 == 0:
            print(f'Episode {episode}, Reward: {ep_reward}')

        env.close()


    # from stable_baselines3 import PPO
    # import gymnasium as gym

    # env = gym.make('CartPole-v1')
    # model = PPO('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=10000)

    print("Training completed! Starting visualization...")
    visualize_model(model)

if __name__ == '__main__':
    main()
    




