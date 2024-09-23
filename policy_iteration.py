import numpy as np
import utils
import copy
from IPython import display

class PolicyIteration:    
    
    def __init__(self, n_actions, state_size, env, bins, gamma=0.1):
        """
        """
        self.state_size = state_size
        self.env = env
        self.bins = bins
        self.gamma = gamma
        # the state values and policy share the same hash table
        self.V, self.policy = self.init_tables(bins.shape[0])
    
    
    def init_tables(self, bins_size):
        """
        """
        V = np.zeros((bins_size+1, bins_size+1, bins_size+1, bins_size+1))
        policy = np.random.randint(0, 1+1, size=(bins_size+1, bins_size+1, bins_size+1, bins_size+1))

        return V, policy
    
    
    def policy_evaluation(self, theta=1e-3):
        """
        Add more exploration and exploitation.
        """
        while True:
            curr_state, _ = self.env.reset()
            curr_state = tuple(utils.to_discrete(curr_state[None], self.bins)[0]) 
            #print(curr_state)
            delta = 0
            time = 1
            
            while True:
                v = 0
                action = self.policy[curr_state]
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_state = tuple(utils.to_discrete(next_state[None], self.bins)[0])
                
                if done or trunc:
                    break
                    
                v += reward * time + self.gamma * self.V[next_state]
                delta += abs(self.V[curr_state] - v)
                self.V[curr_state] = v
                curr_state = next_state
                time += 1

            #print(time)
            #print(delta)
            if delta < theta:
                break
                    
    
    def policy_improvemnt(self, theta=1e-3):
        """
        """
        temp_policy = copy.deepcopy(self.policy)
        stable = 0
        while True:
            curr_state, _ = self.env.reset()
            curr_state = tuple(utils.to_discrete(curr_state[None], self.bins)[0]) 
            delta = 0
            accumulated = 0
            time = 1
            
            while True:
                v = 0
                accumulated_r = 0
                action = self.policy[curr_state]
                alt_action = 1 if action == 0 else 0
                next_state, reward, done, trunc, _ = self.env.step(alt_action)
                next_state = tuple(utils.to_discrete(next_state[None], self.bins)[0])
                accumulated_r += reward
                
                if done or trunc:
                    break
                    
                v += reward*time + self.gamma * self.V[next_state]
                if self.V[curr_state] < v and temp_policy[curr_state] != alt_action:
                    temp_policy[curr_state] = alt_action
                    stable += 1
                curr_state = next_state
                time += 1
                
            accumulated = max(accumulated, accumulated_r)
            if delta < theta:
                break
                
        self.policy = temp_policy
        return stable, accumulated
    
    
    def run(self):
        """
        """
        stable = 1
        while stable != 0:
            # Evaluation
            V = self.policy_evaluation()

            # Imporvement
            stable, accumulated = self.policy_improvemnt()

            print(stable)

        #return policy, V