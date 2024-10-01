import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch


# device to use
device = None

# Gets gym environment.
def get_env():
    return gym.make('CartPole-v0')


# Collect state samples by interacting with the environment.
def get_state_distribution(env):
    state_samples = []

    n_episodes = 100  
    max_steps = 200  

    for episode in range(n_episodes):
        state, obs = env.reset()

        for _ in range(max_steps):
            state_samples.append(state)  
            action = env.action_space.sample()  # random action
            next_state, _, done, trunc, _ = env.step(action)
            state = next_state

            if done or trunc:
                break
    return np.array(state_samples)
   

# View distrbution of observed data.
def plot_distribution(observation_data, nbins, file):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    variables = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity']

    for i in range(4):
        obs_var = observation_data[:, i]

        ax[i%2][int(i/2)].hist(obs_var, bins=nbins, color='black')
        ax[i%2][int(i/2)].set_title(variables[i])
        
    plt.savefig(file)
    

# Dynamically set bins based on observed data.
def get_bins_dynamic(observation_data, nbins):
    bins = np.zeros((nbins, 4))

    for i in range(4):
        _, bins[:, i] = np.histogram(observation_data[:, i], bins=nbins-1)
        
    return bins


# TODO: Set bins uniformily based on observation space.
def get_bins_uniform(ranges, nbins):
    pass


# Gets the discretize version of the continous state.
def to_discrete(observation_data, bins):
    binned_data = np.zeros((observation_data.shape[0], 4), dtype=int)
    
    for i in range(4):
        binned_data[:, i] = np.digitize(observation_data[:, i] 
                                        if len(observation_data.shape) > 1 
                                        else observation_data[i][None], 
                                        bins[:, i]) 
        
    return binned_data


# Wrapper++ for the to_discrete function. Removes ugly code.
def discretize_state(state, bins):
    return tuple(to_discrete(state[None], bins)[0])


# Function to preprocess data for DQN, supports batched and unbatched.
def preprocess(state, s=True):
    state = torch.FloatTensor(np.array(state))
    
    # potentially remove this and simply add group_norm to model
    if s:
        state = state / torch.FloatTensor([2.4, 4, np.radians(12), 5]).unsqueeze(0).unsqueeze(0)
        
    return state.to(device)


# load trained model from disk.
def load(file, model):
    model.load_state_dict(torch.load(file))


# Save trained model to disk.
def save(file, model):
    torch.save(model.state_dict(), file)