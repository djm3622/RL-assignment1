import numpy as np
import utils
import time


class Sarsa:
    
    # Constructor for Sarsa.
    def __init__(self, n_actions, state_size, env, bins, learning_rate=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=1.0):
        self.n_actions = n_actions
        self.state_size = state_size
        self.env = env
        self.bins = bins
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.Q = self.init_q_table()
    
    
    # Init Q-table with the zeros, shape = (*shape.bins[0], n_actions).
    def init_q_table(self):
        shape = tuple([self.bins.shape[0] + 1 for _ in range(4)]+[self.n_actions])
        return np.zeros(shape)
    
    
    # Given a state, random exploration or greedy choice.
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions) # random exploratory
        else:
            return np.argmax(self.Q[state]) # greedy
        
    
    # Update the q-table indexed by current state using the next state (given policy) and its q-value (given policy).
    def update_q_value(self, state, action, reward, next_state, stopped):
        current_q = self.Q[state + (action,)]
        next_action = None
        
        if stopped:
            next_q = 0
        else:
            next_action = self.choose_action(next_state)
            next_q = self.Q[next_state + (next_action,)]
            
        # TD-Sarsa equation.
        new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
        self.Q[state + (action,)] = new_q
        
        return next_action
            
    
    # Train the Sarsa instance.
    # Given some episodes and some amount of steps (per each episode), 
    # simulate environment, get discrete state, get next action according to policy,
    # update the q-table index.
    def train(self, n_episodes=1000, logger=1000, LOG_episodes=None, LOG_time=None):
        episode_rewards = []
        
        if LOG_episodes is not None:
            LOG_episodes[0] = 0
        if LOG_time is not None:
            LOG_time[0] = time.time()
        
        for episode in range(n_episodes):
            state, _ = self.env.reset() # reset env
            state = utils.discretize_state(state, self.bins)
            action = self.choose_action(state)
            total_reward = 0
            self.epsilon *= self.epsilon_decay
            
            for step in range(300):
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = utils.discretize_state(next_state, self.bins)
                                
                total_reward += reward
                
                next_action = self.update_q_value(state, action, reward, next_state, done or truncated)
                
                if done or truncated:
                    break
                
                state = next_state
                action = next_action
            
            episode_rewards.append(total_reward)
            
            if LOG_episodes is not None:
                LOG_episodes[episode+1] = total_reward
            if LOG_time is not None:
                LOG_time[episode+1] = time.time()
            
            if episode % logger == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-logger:]):.2f}")          
        
        return episode_rewards