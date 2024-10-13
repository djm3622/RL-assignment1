import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self, state_size, action_size, linear_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, linear_size)
        self.fc2 = nn.Linear(linear_size, linear_size)
        self.fc3 = nn.Linear(linear_size, linear_size)
        self.fc4 = nn.Linear(linear_size, action_size)
        
        self.relu = nn.ReLU()
        # self.batch_norm = nn.BatchNorm1d(linear_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.batch_norm(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)