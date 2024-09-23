import numpy as np
import utils


class QLearning:

    # Constructor for Q-Learning.
    def __init__(self, n_actions, state_size, env, bins, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.state_size = state_size
        self.env = env
        self.bins = bins
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = self.init_q_table()
    
    
    # Init Q-table with the zeros, shape = (*shape.bins[i], n_actions).
    def init_q_table(self):
        shape = tuple([self.bins.shape[0] + 1 for _ in range(4)]+[self.n_actions])
        return np.zeros(shape)
    
    
    # Given a state, random exploration or greedy choice.
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions) # random exploratory
        else:
            return np.argmax(self.Q[state]) # greedy 
    
    
    # Uses the max q-value to update q-table indexed by current state with maximum value from the next state.
    # DOES NOT make use of exploratory actions.
    def update_q_value(self, state, action, reward, next_state, stopped):
        current_q = self.Q[state + (action,)]
        
        if stopped:
            max_future_q = 0
        else:
            max_future_q = np.max(self.Q[next_state])
            
        # TD-QLearning equation.
        new_q = current_q + self.lr * (reward + (self.gamma * max_future_q) - current_q)
        self.Q[state + (action,)] = new_q
    
    
    # Train the Q-Learning instance.
    # Given some episodes and some amount of steps (per each episode), 
    # simulate environment, get discrete state, get action that maximizes value of next state,
    # update the q-table index.
    def train(self, n_episodes=1000, max_steps=300, logger=1000):
        episode_rewards = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            state = utils.discretize_state(state, self.bins)
            total_reward = 0
            
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_state = utils.discretize_state(next_state, self.bins)
                total_reward += reward
                
                self.update_q_value(state, action, reward, next_state, done or truncated)
                
                if done or truncated:
                    break
                
                state = next_state
            
            episode_rewards.append(total_reward)
            
            # TODO: Log average return per episode.
            if episode % logger == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-logger:]):.2f}")
                
            # TODO: Break and log when completely solved. Add to logger the convergence episode.
        
        return episode_rewards