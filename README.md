Author: David Millard 

# Basic Usage
Each agent is initialized with a constructor and then trained with their builtin `train` function.

    agent = Agent(...)
    rewards = agent.train(...)

Each `train` supports logging the rewards and time. To log these metrics you must pass in seperate log arrays and the builtin train function will update them inplace. 

# Constructors
## Q-Learning
| Variable | Required | Default Value | Type |
|----------|----------|----------|----------|
| Action Size | YES | NA | int |
| State Size | YES | NA | int |
| Environment | YES | NA | Gym-Object |
| Bins | YES | NA | Numpy Array |
| Learning Rate | NO | 0.1 | float |
| Gamma | NO | 0.9 | float | float |
| Epsilon | NO | 0.1 | float |
| Epsilon Decay | NO | 0.9 | float | float |


See the associated juptyer notebooks for more details on how to use this repository.
