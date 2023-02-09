import torch
import random
import torch.optim as optim
import torch.nn.functional as F

from replay_memory import ReplayMemory
from network import Critic,Actor,ValueNetwork



class Agent():
    def __init__(self, state_size, action_size, random_seed, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.target_entropy = -action_size  # -dim(A)
        self.alpha = torch.tensor([1.0]).to(self.device)
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.config.lr_actor)

        # Actor Network
        self.actor = Actor(state_size, action_size, seed = random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic2 = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.lr_critic)
        
        self.value_network = ValueNetwork(state_size,random_seed).to(self.device)
        self.target_value_network = ValueNetwork(state_size,random_seed).to(self.device)
        self.target_value_network.load_state_dict(self.value_network.state_dict())
        self.target_value_network.eval()
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr = self.config.lr_value)

        # Replay memory
        self.memory = ReplayMemory(action_size, self.config.memory_size, self.config.batch_size, random_seed)


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.config.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.config.gamma)


    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor.get_action(state)[0].detach()
        return action
    
    def deterministic_act(self,state):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor.deterministic_action(state)[0].detach()
        return action

    def learn(self, experiences, gamma=0.99):
        states, actions, rewards, next_states, dones = experiences
        
        # Compute value loss
        action_pred, state_log_pi = self.actor.evaluate(states)
        q1 = self.critic1(states.to(self.device), action_pred.squeeze(0).to(self.device))
        q2 = self.critic2(states.to(self.device), action_pred.squeeze(0).to(self.device))
        q = torch.min(q1, q2)
        with torch.no_grad():
            if self.config.fixed_alpha == None:
                target_value = q- self.alpha * state_log_pi
            else:
                target_value = q - self.config.fixed_alpha * state_log_pi
        value = self.value_network(states)
        value_loss = 0.5*F.mse_loss(value,target_value)
        
        # Compute Q loss
        with torch.no_grad():
            target_q = self.config.reward_scale*rewards + gamma*self.target_value_network(next_states)*(1-dones)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = 0.5*F.mse_loss(q1, target_q)
        critic2_loss = 0.5*F.mse_loss(q2, target_q)

        
        # ---------------------------- update actor ---------------------------- #
        if self.config.fixed_alpha == None:
            alpha = torch.exp(self.log_alpha)
            # Compute alpha loss
            alpha_loss = - (self.log_alpha.cpu() * (state_log_pi.cpu() + self.target_entropy).detach().cpu()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = alpha.to(self.device)
            # Compute actor loss
            actor_loss = (alpha.to(self.device) * state_log_pi.squeeze(0) - q).mean()
        else:
            actor_loss = (self.config.fixed_alpha * state_log_pi.squeeze(0) - q).mean()
        
        # ---------------------------- minimize the loss ---------------------------- #
        # policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.value_network, self.target_value_network, self.config.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)