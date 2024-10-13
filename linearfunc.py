import numpy as np
import utils
import time

class LinearApproximation:
    
    # Constructor for Linear Function Approximation.
    def __init__(self, n_actions, state_size, env, learning_rate=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=1.0):
        self.n_actions = n_actions
        self.state_size = state_size
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.weights = self.init_weights()
        
        
    # state -> cart_position, cart_velocity, pole_angle, pole_angular_velocity
    # Extracts a random assortment of features from the given state.
    def feature_extractor(self, state):
        norms = np.array([2.4, 4, np.radians(12), 5, 2.4**2, 4**2, np.radians(12)**2, 5**2, 2.4*np.radians(12), 4*5])
    
        features = np.array([
            state[0], state[1], state[2], state[3],
            state[0] ** 2, state[1] ** 2, state[2] ** 2, state[3] ** 2,    
            state[0] * state[2], state[1] * state[3]                          
        ])
            
        return features / norms

    # Init random weights for features, per action.
    def init_weights(self):
        return np.zeros((self.n_actions, self.feature_extractor(np.zeros(self.state_size)).shape[0]))

    
    # Approximate the Q-value
    def get_q_value(self, state, action):
        features = self.feature_extractor(state)
        return np.dot(self.weights[action], features)

    
    # Given a state, random exploration or greedy choice.
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)   # random exploratory
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
            return np.argmax(q_values)   # greedy 

        
    # Get the estimate for q-value. Use the TD-error to incrementally adjust the weights.
    # Uses the next optimal estimated action in the TD-error. 
    def update_weights(self, state, action, reward, next_state, stopped):
        current_q = self.get_q_value(state, action)
        
        if stopped:
            target = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in range(self.n_actions)]
            target = reward + self.gamma * np.max(next_q_values)
        
        features = self.feature_extractor(state)
        self.weights[action] += self.lr * (target - current_q) * features

        
    # Train the Linear-func instance.
    # Given some episodes and some amount of steps (per each episode), 
    # simulate environment, get continous state, compute action that maximizes value of next state,
    # update the weights of the non-linear features.
    def train(self, episodes=1000, max_steps=300, logger=1000, LOG_episodes=None, LOG_time=None):
        episode_rewards = []
        
        if LOG_episodes is not None:
            LOG_episodes[0] = 0
        if LOG_time is not None:
            LOG_time[0] = time.time()
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            self.epsilon *= self.epsilon_decay
            
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                total_reward += reward
                
                self.update_weights(state, action, reward, next_state, done or truncated)
                
                if done or truncated:
                    break
                
                state = next_state
            
            episode_rewards.append(total_reward)
            
            if LOG_episodes is not None:
                LOG_episodes[episode+1] = total_reward
            if LOG_time is not None:
                LOG_time[episode+1] = time.time()
                
            if episode % logger == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-logger:]):.2f}")
                
                if np.mean(episode_rewards[-logger:]) > 195:
                    break
                        
        return episode_rewards