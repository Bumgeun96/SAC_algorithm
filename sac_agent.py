import torch
import random
import torch.optim as optim
import torch.nn.functional as F

from replay_memory import ReplayMemory
from network import Critic,Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, random_seed, config):
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
        self.alpha = torch.tensor([1.0]).to(device)
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=self.config.lr_actor)

        # Actor Network
        self.actor = Actor(state_size, action_size, seed = random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed).to(device)

        self.critic1_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.lr_critic, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.lr_critic, weight_decay=0)

        # Replay memory
        self.memory = ReplayMemory(action_size, self.config.memory_size, self.config.batch_size, random_seed)


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.config.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.config.gamma)


    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        action = self.actor.get_action(state)[0].detach()
        return action
    
    def deterministic_act(self,state):
        state = torch.from_numpy(state).float().to(device)
        action = self.actor.deterministic_action(state)[0].detach()
        return action

    def learn(self, experiences, gamma=0.99):
        states, actions, rewards, next_states, dones = experiences
        
        next_action, next_state_log_pi = self.actor.evaluate(next_states)
        Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)
        if self.config.fixed_alpha == None:
            Q_targets = rewards + (gamma * (1 - dones) * (Q_target_next - self.alpha * next_state_log_pi.squeeze(0)))
        else:
            Q_targets = rewards + (gamma * (1 - dones) * (Q_target_next - self.config.fixed_alpha * next_state_log_pi.squeeze(0)))
        # Compute critic loss
        q_1 = self.critic1(states, actions)
        q_2 = self.critic2(states, actions)
        critic1_loss = 0.5*F.mse_loss(q_1, Q_targets.detach())
        critic2_loss = 0.5*F.mse_loss(q_2, Q_targets.detach())
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        if self.config.fixed_alpha == None:
            alpha = torch.exp(self.log_alpha)
            # Compute alpha loss
            actions_pred, state_log_pi = self.actor.evaluate(states)
            alpha_loss = - (self.log_alpha.cpu() * (state_log_pi.cpu() + self.target_entropy).detach().cpu()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = alpha.to(device)
            # Compute actor loss
            q1 = self.critic1(states, actions_pred.squeeze(0))
            q2 = self.critic2(states, actions_pred.squeeze(0))
            q = torch.min(q1,q2)
            actor_loss = (alpha.to(device) * state_log_pi.squeeze(0) - q).mean()
        else:
            actions_pred, state_log_pi = self.actor.evaluate(states)
            q1 = self.critic1(states, actions_pred.squeeze(0))
            q2 = self.critic2(states, actions_pred.squeeze(0))
            q = torch.min(q1,q2)
            actor_loss = (self.config.fixed_alpha * state_log_pi.squeeze(0) - q).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target, self.config.tau)
        self.soft_update(self.critic2, self.critic2_target, self.config.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)