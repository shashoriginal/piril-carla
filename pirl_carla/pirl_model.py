import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class PhysicsInformedPolicy(nn.Module):
    """Physics-informed policy network"""
    
    def __init__(self, 
                 state_dim: int = 6,  # [x, y, theta, v, distance_to_target, angle_to_target]
                 action_dim: int = 2,  # [throttle, steering]
                 hidden_dim: int = 256):
        super().__init__()
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Physics prediction network
        self.physics_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Save dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            state: Input state tensor [x, y, theta, v, distance_to_target, angle_to_target]
            
        Returns:
            action: Predicted action [throttle, steering]
            next_state: Predicted next state
        """
        # Get action from policy network
        action = torch.tanh(self.policy_net(state))  # Use tanh for bounded actions
        
        # Predict next state using physics network
        physics_input = torch.cat([state, action], dim=-1)
        next_state = self.physics_net(physics_input)
        
        return action, next_state
    
    def physics_loss(self, 
                    state: torch.Tensor,
                    action: torch.Tensor,
                    next_state: torch.Tensor,
                    L: float = 2.5,
                    dt: float = 0.05) -> torch.Tensor:
        """Compute physics consistency loss based on bicycle model
        
        Args:
            state: Current state [x, y, theta, v, distance_to_target, angle_to_target]
            action: Applied action [throttle, steering]
            next_state: Next state
            L: Wheelbase length
            dt: Timestep
            
        Returns:
            loss: Physics consistency loss
        """
        # Extract state components
        x, y, theta, v = torch.split(state[..., :4], 1, dim=-1)
        throttle, steer = torch.split(action, 1, dim=-1)
        next_x, next_y, next_theta, next_v = torch.split(next_state[..., :4], 1, dim=-1)
        
        # Convert throttle to acceleration (simple model)
        a = throttle * 5.0  # max acceleration of 5 m/s^2
        
        # Bicycle model equations
        x_dot = v * torch.cos(theta)
        y_dot = v * torch.sin(theta)
        theta_dot = (v / L) * torch.tan(steer)
        v_dot = a
        
        # Predicted next state using Euler integration
        pred_x = x + x_dot * dt
        pred_y = y + y_dot * dt
        pred_theta = theta + theta_dot * dt
        pred_v = v + v_dot * dt
        
        # Compute MSE loss between predicted and actual next state positions
        position_loss = F.mse_loss(
            torch.cat([pred_x, pred_y, pred_theta, pred_v], dim=-1),
            torch.cat([next_x, next_y, next_theta, next_v], dim=-1)
        )
        
        return position_loss
    
    def collision_loss(self,
                      state: torch.Tensor,
                      obstacles: torch.Tensor,
                      safe_distance: float = 5.0) -> torch.Tensor:
        """Compute collision avoidance loss
        
        Args:
            state: Current state [x, y, theta, v, distance_to_target, angle_to_target]
            obstacles: Obstacle positions [N, 2] or empty tensor
            safe_distance: Minimum safe distance
            
        Returns:
            loss: Collision avoidance loss
        """
        # If no obstacles, return zero loss
        if obstacles.shape[0] == 0:
            return torch.tensor(0.0, device=state.device)
            
        # Extract position from state
        pos = state[..., :2]  # x, y coordinates
        
        # Compute distances to all obstacles
        distances = torch.norm(pos.unsqueeze(1) - obstacles, dim=-1)
        
        # Compute loss: activate when distance < safe_distance
        loss = F.relu(safe_distance - distances).mean()
        
        return loss

class PhysicsInformedSAC(nn.Module):
    """Physics-informed Soft Actor-Critic"""
    
    def __init__(self,
                 state_dim: int = 6,  # [x, y, theta, v, distance_to_target, angle_to_target]
                 action_dim: int = 2,  # [throttle, steering]
                 hidden_dim: int = 256):
        super().__init__()
        
        # Save dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Policy network
        self.policy = PhysicsInformedPolicy(state_dim, action_dim, hidden_dim)
        
        # Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Target networks
        self.q1_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize target networks with policy network weights
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Entropy temperature parameter
        self.alpha = nn.Parameter(torch.ones(1))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            state: Input state tensor
            
        Returns:
            action: Sampled action
            next_state: Predicted next state
        """
        return self.policy(state)
    
    def compute_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    next_states: torch.Tensor,
                    rewards: torch.Tensor,
                    dones: torch.Tensor,
                    obstacles: torch.Tensor,
                    gamma: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SAC losses with physics constraints
        
        Args:
            states: Batch of states
            actions: Batch of actions
            next_states: Batch of next states
            rewards: Batch of rewards
            dones: Batch of done flags
            obstacles: Batch of obstacle positions
            gamma: Discount factor
            
        Returns:
            policy_loss: Actor loss
            q_loss: Critic loss
            alpha_loss: Temperature loss
        """
        # Physics consistency loss
        physics_loss = self.policy.physics_loss(states, actions, next_states)
        
        # Collision avoidance loss
        collision_loss = self.policy.collision_loss(states, obstacles)
        
        # Sample actions and log probs from current policy
        new_actions, predicted_next_states = self.policy(states)
        log_probs = self._compute_log_probs(states, new_actions)
        
        # Compute Q-values
        q1 = self.q1(torch.cat([states, actions], dim=-1))
        q2 = self.q2(torch.cat([states, actions], dim=-1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_actions, _ = self.policy(next_states)
            next_log_probs = self._compute_log_probs(next_states, next_actions)
            
            q1_next = self.q1_target(torch.cat([next_states, next_actions], dim=-1))
            q2_next = self.q2_target(torch.cat([next_states, next_actions], dim=-1))
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            
            q_target = rewards + (1 - dones) * gamma * q_next
        
        # Compute losses
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        policy_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()
        policy_loss = policy_loss + physics_loss + collision_loss
        
        alpha_loss = -(self.alpha * (log_probs + 2.0).detach()).mean()
        
        return policy_loss, q_loss, alpha_loss
    
    def _compute_log_probs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities of actions
        
        Args:
            states: Input states
            actions: Actions to compute log probs for
            
        Returns:
            log_probs: Log probabilities of actions
        """
        # Assume Gaussian policy with tanh squashing
        mean = self.policy.policy_net(states)
        log_std = torch.zeros_like(mean)  # Fixed std for simplicity
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        log_probs = normal.log_prob(actions)
        
        # Apply tanh squashing correction
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6)
        
        return log_probs.sum(-1, keepdim=True)
