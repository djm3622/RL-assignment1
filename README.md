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
This repository offers a file with an array of logging techniques. An exhaustive list is provided below.

```
flush_LOG
```


See the associated juptyer notebooks for more details on how to use this repository.
