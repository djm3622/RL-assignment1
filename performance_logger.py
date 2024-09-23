import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display


# Log to store runs in. 
# Data should be inserted LOG['type_run-id'] = [item, item ...]
LOG = {}


# TODO: flush log to disk
def flush_LOG(location):
    pass


# TODO: flush log to disk
def read_LOG(location):
    pass


# Visualization of the 2 state observations.
def visualize_q_values_2D(Q, dim1, dim2):    
    avg_q_values = np.mean(np.max(Q, axis=-1), axis=(dim1, dim2))
    display.display(display.HTML(f'<h3>Average Max Q-Values</h3>'))
    display.display(display.HTML(f'<p>Min: {np.min(avg_q_values):.2f}, Max: {np.max(avg_q_values):.2f}</p>'))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_q_values, cmap='viridis', origin='lower')
    plt.colorbar(im)
    plt.title('Average Max Q-Values')
    plt.xlabel(f'Discretized Dimension {dim1}')
    plt.ylabel(f'Discretized Dimension {dim2}')
    plt.show()
    

# TODO: Plot 3 dimensions.
def visualize_q_values_3D(Q, dim1, dim2, dim3):    
    pass

    
# Plot the training progress using the episode rewards. Uses a moving average to provide smotheness.
def plot_training_progress(episode_rewards, obj):
    conv = 100
    moving_average = np.ones(conv) / conv
    convolved = np.convolve(episode_rewards, moving_average, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, convolved.shape[0]), convolved)
    plt.title(f'{obj} Training Progress (Moving Average: {conv})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    
    
# TODO: wrapper to time a function execution
def timeit(func):
    pass
    
    
# TODO: function for stability?
def __stability__():
    pass


# TODO: function for stability?
def __efficiency__():
    pass