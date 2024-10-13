import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display


# Reference for variable names.
variables = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity']

# Log to store runs in. 
# Data should be inserted LOG['type_run-id'] = [item, item ...]
LOG = {}


# Flush log to disk.
def flush_LOG(location):
    np.savez(location, **LOG)


# Read log from disk.
def read_LOG(location):
    return np.load(location)


# Visualization of the 2 state observations.
def visualize_q_values_2D(Q, dim1, dim2, file):    
    all_dims = set([0, 1, 2, 3]) 
    used_dims = set([dim1, dim2])
    unused_dim = tuple(all_dims - used_dims)
    avg_q_values = np.mean(np.max(Q, axis=-1), axis=unused_dim)
    
    display.display(display.HTML(f'<h3>Max Q-Values (2D)</h3>'))
    display.display(display.HTML(f'<p>Min: {np.min(avg_q_values):.2f}, Max: {np.max(avg_q_values):.2f}</p>'))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_q_values, cmap='viridis', origin='lower')
    plt.colorbar(im)
    plt.title('Max Q-Values (2D)')
    plt.xlabel(f'{variables[dim1]}')
    plt.ylabel(f'{variables[dim2]}')
    plt.savefig(file)

# Visualization of the 3 state observations.
def visualize_q_values_3D(Q, dim1, dim2, dim3, file, threshold=0.0):
    all_dims = set([0, 1, 2, 3]) 
    used_dims = set([dim1, dim2, dim3])
    unused_dim = tuple(all_dims - used_dims)
    max_q_values = np.mean(np.max(Q, axis=-1), axis=unused_dim) 
    
    display.display(display.HTML(f'<h3>Max Q-Values (3D)</h3>'))
    display.display(display.HTML(f'<p>Min: {np.min(max_q_values):.2f}, Max: {np.max(max_q_values):.2f}</p>'))
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    shape = max_q_values.shape
    
    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    z = np.arange(0, shape[2])
    X, Y, Z = np.meshgrid(x, y, z)
    
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    C = max_q_values.flatten()
    
    mask = np.abs(C) > threshold
    X_filtered = X[mask]
    Y_filtered = Y[mask]
    Z_filtered = Z[mask]
    C_filtered = C[mask]
    
    scatter = ax.scatter(X_filtered, Y_filtered, Z_filtered, c=C_filtered, cmap='viridis', s=50, alpha=0.5)
    
    ax.set_xlabel(f'{variables[dim1]}')
    ax.set_ylabel(f'{variables[dim2]}')
    ax.set_zlabel(f'{variables[dim3]}')
    ax.set_title('Max Q-Values (3D)')
    
    fig.colorbar(scatter, label='Max Q-Value')
    plt.savefig(file)

    
# Plot the training progress using the episode rewards. Uses a moving average to provide smotheness.
def plot_training_progress(episode_rewards, obj, conv, file):
    moving_average = np.ones(conv) / conv
    convolved = np.convolve(episode_rewards, moving_average, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, convolved.shape[0]), convolved)
    plt.title(f'{obj} Training Progress (Moving Average: {conv})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(file)
    
# Plot the time usage using the logged time.
def plot_time_progress(time, obj, conv, file):
    time = np.array([time[t] - time[t-1] for t in range(1, time.shape[0])])
    moving_average = np.ones(conv) / conv
    convolved = np.convolve(time, moving_average, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, convolved.shape[0]), convolved)
    plt.title(f'{obj} Training Time (Moving Average: {conv})')
    plt.xlabel('Episode')
    plt.ylabel('Time (s)')
    plt.savefig(file)
    

# Plot simple bar charts to visual the weights.
def visualize_weights(weights, features, title, file):
    fig, ax = plt.subplots(figsize=(12, 6))
        
    x = np.arange(len(features))
    width = 0.35

    ax.bar(x - width/2, weights[0], width, label='Action 0 (Left)')
    ax.bar(x + width/2, weights[1], width, label='Action 1 (Right)')

    ax.set_ylabel('Weight Value')
    ax.set_title(f'{title} Feature Weights')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(file)

# TODO: function for stability?
def __stability__():
    pass


# TODO: function for stability?
def __efficiency__():
    pass