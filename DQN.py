import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import torch
from collections import deque
import random
import time


class DQN:

    # Constructor for DQN.
    def __init__(self, policy_net, target_net, n_actions, state_size, env, learning_rate=0.1, 
                 gamma=0.9, epsilon=0.1, epsilon_decay=1.0, memory=10000, target_update=128):
        
        self.n_actions = n_actions
        self.state_size = state_size
        self.env = env
        self.memory = deque(maxlen=memory)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.target_update = target_update
        self.update_counter = 0
        
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        
    # Add instance to memory. Didn't need to save that much and could speed it up if removed half of unused.
    def enqueue(self, state, action, reward, next_state, done, truncated):
        self.memory.append((state, action, reward, next_state, done, truncated))
    
    
    # Given a state, random exploration or greedy choice.
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions) # random exploratory
        else:
            state = utils.preprocess(state)
                        
            self.policy_net.eval()
            with torch.no_grad():
                act_values = self.policy_net(state)
            act_values = act_values.cpu().data.numpy()
            
            return np.argmax(act_values) # greedy
    
    
    # Train from the memory.
    # TODO
    def replay(self, batch_size):
        self.policy_net.train()
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, truncated = zip(*batch)
        
        # Most of these are usesless but I don't care anymore.
        states = utils.preprocess(states)[:, 0, :]
        actions = utils.preprocess(actions, s=False).long().unsqueeze(-1)
        rewards = utils.preprocess(rewards, s=False).unsqueeze(-1)
        next_states = utils.preprocess(next_states)[:, 0, :]
        dones = utils.preprocess(dones, s=False).unsqueeze(-1)
        truncated = utils.preprocess(truncated, s=False).unsqueeze(-1)

        # TODO : comment
        current_q_values = self.policy_net(states).gather(1, actions) 
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1).detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    
    # Train the DQN agent
    # TODO
    def train(self, episodes=10000, batch_size=128, logger=100, LOG_episodes=None, LOG_time=None):
        episode_rewards = []
        
        if LOG_episodes is not None:
            LOG_episodes[0] = 0
        if LOG_time is not None:
            LOG_time[0] = time.time()
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            state = np.expand_dims(state, axis=0)
            total_reward = 0
            self.epsilon *= self.epsilon_decay

            for step in range(300):
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = np.expand_dims(next_state, axis=0)
                self.enqueue(state, action, reward, next_state, done, truncated)
                
                state = next_state
                total_reward += reward

                if len(self.memory) > batch_size:
                    self.replay(batch_size)

                if done or truncated:
                    break
                    
            if LOG_episodes is not None:
                LOG_episodes[episode+1] = total_reward
            if LOG_time is not None:
                LOG_time[episode+1] = time.time()

            episode_rewards.append(total_reward)
            
            if episode % logger == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-logger:]):.2f}")
                print(self.epsilon)
                
                if np.mean(episode_rewards[-logger:]) > 195:
                    break
                                    
        return episode_rewards