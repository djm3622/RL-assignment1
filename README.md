Author: David Millard 

Usage:
Each agent is initialized with a constructor and then trained with their builtin `train` function.

    agent = Agent(...)
    rewards = agent.train(...)

Each `train` supports logging the rewards and time. To log these metrics you must pass in seperate log arrays and the builtin train function will update them inplace. 

The associated contructors for the agents are as follows:

...


See the associated juptyer notebooks for more details on how to use this repository.