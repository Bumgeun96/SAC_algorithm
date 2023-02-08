import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.fc1 = nn.Linear(state_size+action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=0, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.mu = nn.Linear(256, action_size)
        self.log_std = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std

    def evaluate(self, state):
        mu, std = self.forward(state)
        # print(mu.size())
        distribution = Normal(0, 1)
        epsilon = distribution.sample().to(device)
        action = torch.tanh(mu+epsilon*std)
        log_prob = Normal(mu, std).log_prob(mu + epsilon * std) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1,keepdim=True)
        return action, log_prob
    
    def get_action(self,state):
        mu,log_std = self.forward(state)
        std = log_std.exp()
        distribution = Normal(0,1)
        epsilon = distribution.sample().to(device)
        action = torch.tanh(mu+epsilon*std).cpu()
        return action
    
    def deterministic_action(self,state):
        mu,log_std = self.forward(state)
        action = torch.tanh(mu).cpu()
        return action