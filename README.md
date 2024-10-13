Author: David Millard 

# Basic Usage
Each agent is initialized with a constructor and then trained with their builtin `train` function.

    agent = Agent(...)
    rewards = agent.train(...)

Each `train` supports logging the rewards and time. To log these metrics you must pass in seperate log arrays and the builtin train function will update them inplace. 

# Constructors
The following constructors have their arguments in order of position. 

### Q-Learning/Sarsa
| Variable | Required | Default Value | Type |
|----------|----------|----------|----------|
| Action Size | YES | NA | int |
| State Size | YES | NA | int |
| Environment | YES | NA | Gym-Object |
| Bins | YES | NA | Numpy Array |
| Learning Rate | NO | 0.1 | float |
| Gamma | NO | 0.9 | float | float |
| Epsilon | NO | 0.1 | float |
| Epsilon Decay | NO | 0.9 | float | 

### Monte-Carlo Q-Table
| Variable | Required | Default Value | Type |
|----------|----------|----------|----------|
| Action Size | YES | NA | int |
| State Size | YES | NA | int |
| Environment | YES | NA | Gym-Object |
| Bins | YES | NA | Numpy Array |
| Gamma | NO | 0.9 | float | float |
| Epsilon | NO | 0.1 | float |
| Epsilon Decay | NO | 0.9 | float | 
| First Visit | NO | True | bool | 

### Linear Function Approximation w/ and w/o MC
| Variable | Required | Default Value | Type |
|----------|----------|----------|----------|
| Action Size | YES | NA | int |
| State Size | YES | NA | int |
| Environment | YES | NA | Gym-Object |
| Learning Rate | NO | 0.1 | float |
| Gamma | NO | 0.9 | float | float |
| Epsilon | NO | 0.1 | float |
| Epsilon Decay | NO | 1.0 | float | 

### DQN
| Variable | Required | Default Value | Type |
|----------|----------|----------|----------|
| Policy Network | YES | NA | PyTorch (nn.Module) |
| Target Network | YES | NA | PyTorch (nn.Module) |
| Action Size | YES | NA | int |
| State Size | YES | NA | int |
| Environment | YES | NA | Gym-Object |
| Learning Rate | NO | 0.1 | float |
| Gamma | NO | 0.9 | float | float |
| Epsilon | NO | 0.1 | float |
| Epsilon Decay | NO | 1.0 | float | 
| Memory | NO | 10000 | int | 
| Taget Update Freq. | NO | 128 | int | 

# Performance Evaluation
A variable `LOG` is statically accessible and expects Numpy Arrays as entries in a dictionary.

`flush_LOG(location)` := Flushes the `LOG` to the specified location.

`read_LOG(location)` := Reads the `LOG` from the specified location.

`visualize_q_values_2D(Q, dim1, dim2, file)` := Visualizes dimensions of a Q-table. It expects a file name to write the resulting plot to.

`visualize_q_values_3D(Q, dim1, dim2, dim3, file, threshold=0.0` := Visualizes dimensions of a Q-table. For the most part the same as `visualize_q_values_2D`, but also includes thresholding to delete sub 0.0 Q-values from cluttering the plot.

`plot_training_progress(episode_rewards, obj, conv, file)` :=

See the associated juptyer notebooks for more details on how to use this repository.
