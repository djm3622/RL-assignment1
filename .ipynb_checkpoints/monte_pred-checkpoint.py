import numpy as np
import utils


class MonteCarloQTable:
    
    # Constructor for Monte Carlo Q-Table instance.
    def __init__(self, n_actions, state_size, env, bins, gamma=0.9, epsilon=0.1, first_visit=True):
        self.n_actions = n_actions
        self.state_size = state_size
        self.env = env
        self.bins = bins
        self.gamma = gamma
        self.epsilon = epsilon
        self.first_visit = first_visit
        
        self.Q = self.init_q_table()
        self.returns = {}  # NEED A SEPERATE DICT FOR RETURNS!!!!
    
    
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
    
    
    # Get a single episode and log all (state, action[that led there], reward) pairs.
    def generate_episode(self, max_steps=300):
        episode = []
        state, _ = self.env.reset()
        state = utils.discretize_state(state, self.bins)
        
        for _ in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            next_state = utils.discretize_state(next_state, self.bins)
            episode.append((state, action, reward))
            
            if done or truncated:
                break
            
            state = next_state
        
        return episode
    
    
    # Update the q table according to the MC method. Given an episode, 
    # go through all (state, action, reward) pairs, (rewardless state/action pairs for f-v-MC),
    # add to reward dict, get the mean of all the rewards from it.
    # This function implements both f-v-MC and e-v-MC.
    def update_q_table(self, episode):
        G = 0
        episodes_no_r = [(s, a) for (s, a, _) in episode] # list for rewardless state/action pairs
        
        for ep in range(len(episode)-1, 0-1, -1):
            state, action, reward = episode[ep]
            G = self.gamma * G + reward
            
            # First-visit MC : update only if its the first occurrence of the state-action pair
            if self.first_visit:
                if (state, action) not in episodes_no_r[:ep]: # make sure this is the first visit
                    self.compute_update_q_table(state, action, G)
                    
            # Every-visit MC : update at every occurrence of the state-action pair   
            else:
                self.compute_update_q_table(state, action, G)
    
    
    # Helper to carry out the actual update of the q-table.
    def compute_update_q_table(self, state, action, G):
        if (state, action) not in self.returns:
            self.returns[(state, action)] = [] # returns lookup for multi-episode
        self.returns[(state, action)].append(G)
        self.Q[state + (action,)] = np.mean(self.returns[(state, action)])
    
    
    # Generate episode data, update q-table from it, accumulate reward for output.
    def train(self, n_episodes=1000, logger=1000):
        episode_rewards = []
        
        for episode in range(n_episodes):
            episode_data = self.generate_episode()
            self.update_q_table(episode_data)
            
            episode_rewards.append(sum(r for _, _, r in episode_data))
            
            # TODO: Log average return per episode.
            if episode % logger == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-logger:]):.2f}")
                
            # TODO: Break and log when completely solved. Add to logger the convergence episode.
        
        return self.Q, episode_rewards