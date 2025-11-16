# tax_optimization/reinforcement_learning.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class TaxOptimizationMDP:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.95
        
    def get_state(self, income: np.ndarray, expenses: np.ndarray, 
                 deductions: np.ndarray, month: int) -> np.ndarray:
        state = np.concatenate([
            income / 100000,
            expenses / 100000,
            deductions / 10000,
            [month / 12]
        ])
        return state
    
    def calculate_reward(self, current_tax: float, 
                        baseline_tax: float) -> float:
        tax_savings = baseline_tax - current_tax
        reward = tax_savings
        
        return reward
    
    def is_terminal(self, month: int) -> bool:
        return month >= 12

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class TaxOptimizationAgent:
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.batch_size = 64
        self.target_update_freq = 10
        
        self.mdp = TaxOptimizationMDP(state_dim, action_dim)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, episodes: int, tax_calculator, user_data: Dict):
        episode_rewards = []
        
        for episode in range(episodes):
            total_reward = 0
            month = 1
            
            income = user_data['income_vector']
            expenses = user_data['expense_vector']
            deductions = user_data['deduction_vector']
            
            state = self.mdp.get_state(income, expenses, deductions, month)
            
            while not self.mdp.is_terminal(month):
                action = self.select_action(state, training=True)
                
                strategy = self._action_to_strategy(action)
                current_tax = tax_calculator.calculate_tax_for_strategy(
                    income, expenses, deductions, strategy
                )
                
                baseline_strategy = {'name': 'baseline'}
                baseline_tax = tax_calculator.calculate_tax_for_strategy(
                    income, expenses, deductions, baseline_strategy
                )
                
                reward = self.mdp.calculate_reward(current_tax, baseline_tax)
                
                month += 1
                next_state = self.mdp.get_state(income, expenses, deductions, month)
                done = self.mdp.is_terminal(month)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                loss = self.train_step()
                
                state = next_state
                total_reward += reward
            
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            
            self.decay_epsilon()
            episode_rewards.append(total_reward)
        
        return episode_rewards
    
    def _action_to_strategy(self, action: int) -> Dict:
        strategies = [
            {'name': 'baseline'},
            {'name': 'max_deductions', 'deduction_adjustments': {}},
            {'name': 'bonus_separate', 'income_split': {'year_end_bonus_separate': 0}},
            {'name': 'combined_optimization'}
        ]
        
        return strategies[min(action, len(strategies) - 1)]