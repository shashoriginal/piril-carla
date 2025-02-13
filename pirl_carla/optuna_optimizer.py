import optuna
import torch
import numpy as np
from typing import Dict, Any
from pirl_carla.carla_gym_env import CarlaEnv
from pirl_carla.pirl_model import PhysicsInformedSAC

class PIRLOptimizer:
    """Optimizer for Physics-Informed Reinforcement Learning"""
    
    def __init__(self,
                 state_dim: int = 6,  # [x, y, theta, v, distance_to_target, angle_to_target]
                 action_dim: int = 2,  # [throttle, steering]
                 n_trials: int = 100,
                 n_episodes: int = 1000,
                 max_steps: int = 1000):
        print(f"\nInitializing optimizer with:")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_trials = n_trials
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        
        # Create study
        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
    def optimize(self) -> Dict[str, Any]:
        """Run optimization"""
        self.study.optimize(self._objective, n_trials=self.n_trials)
        return self.study.best_params
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        # Sample hyperparameters
        params = {
            'hidden_dim': trial.suggest_int('hidden_dim', 64, 512),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_int('batch_size', 32, 256),
            'gamma': trial.suggest_float('gamma', 0.9, 0.99),
            'tau': trial.suggest_float('tau', 0.001, 0.1),
            'alpha': trial.suggest_float('alpha', 0.01, 1.0, log=True),
            'buffer_size': trial.suggest_int('buffer_size', 10000, 1000000),
            'physics_weight': trial.suggest_float('physics_weight', 0.1, 10.0, log=True),
            'collision_weight': trial.suggest_float('collision_weight', 0.1, 10.0, log=True)
        }
        
        print(f"\nTrial {trial.number} with parameters:")
        for param, value in params.items():
            print(f"{param}: {value}")
        
        # Initialize environment and agent
        env = CarlaEnv()
        
        # Verify state dimension matches environment
        state = env.reset()
        actual_state_dim = len(state)
        if actual_state_dim != self.state_dim:
            raise ValueError(f"Environment state dimension ({actual_state_dim}) does not match expected ({self.state_dim})")
        
        # Initialize agent with correct dimensions
        agent = PhysicsInformedSAC(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=params['hidden_dim']
        )
        
        optimizer = torch.optim.Adam(agent.parameters(), lr=params['learning_rate'])
        
        # Initialize replay buffer
        buffer_size = params['buffer_size']
        replay_buffer = {
            'states': np.zeros((buffer_size, self.state_dim)),
            'actions': np.zeros((buffer_size, self.action_dim)),
            'next_states': np.zeros((buffer_size, self.state_dim)),
            'rewards': np.zeros(buffer_size),
            'dones': np.zeros(buffer_size),
            'ptr': 0,
            'size': 0
        }
        
        # Training loop
        episode_rewards = []
        
        try:
            for episode in range(self.n_episodes):
                state = env.reset()
                episode_reward = 0
                
                for step in range(self.max_steps):
                    # Select action
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        action, _ = agent(state_tensor)
                        action = action.squeeze(0).numpy()
                    
                    # Take step in environment
                    next_state, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    # Store transition in buffer
                    idx = replay_buffer['ptr']
                    replay_buffer['states'][idx] = state
                    replay_buffer['actions'][idx] = action
                    replay_buffer['next_states'][idx] = next_state
                    replay_buffer['rewards'][idx] = reward
                    replay_buffer['dones'][idx] = done
                    replay_buffer['ptr'] = (idx + 1) % buffer_size
                    replay_buffer['size'] = min(replay_buffer['size'] + 1, buffer_size)
                    
                    # Update agent if enough samples
                    if replay_buffer['size'] >= params['batch_size']:
                        # Sample batch
                        idxs = np.random.randint(0, replay_buffer['size'], size=params['batch_size'])
                        batch = {
                            'states': torch.FloatTensor(replay_buffer['states'][idxs]),
                            'actions': torch.FloatTensor(replay_buffer['actions'][idxs]),
                            'next_states': torch.FloatTensor(replay_buffer['next_states'][idxs]),
                            'rewards': torch.FloatTensor(replay_buffer['rewards'][idxs]).unsqueeze(-1),
                            'dones': torch.FloatTensor(replay_buffer['dones'][idxs]).unsqueeze(-1)
                        }
                        
                        # Get nearby obstacles
                        vehicles = env.world.get_actors().filter('vehicle.*')
                        obstacle_positions = []
                        for vehicle in vehicles:
                            if vehicle.id != env.vehicle.id:
                                loc = vehicle.get_location()
                                obstacle_positions.append([loc.x, loc.y])
                        
                        # Convert obstacles to tensor (handle empty case)
                        if obstacle_positions:
                            batch['obstacles'] = torch.FloatTensor(obstacle_positions)
                        else:
                            batch['obstacles'] = torch.zeros((0, 2))  # Empty tensor with correct shape
                        
                        # Compute losses
                        policy_loss, q_loss, alpha_loss = agent.compute_loss(
                            batch['states'],
                            batch['actions'],
                            batch['next_states'],
                            batch['rewards'],
                            batch['dones'],
                            batch['obstacles'],
                            gamma=params['gamma']
                        )
                        
                        # Weight losses
                        total_loss = (policy_loss + 
                                    params['physics_weight'] * q_loss +
                                    params['collision_weight'] * alpha_loss)
                        
                        # Update networks
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                    
                    if done:
                        break
                        
                    state = next_state
                
                episode_rewards.append(episode_reward)
                
                # Report intermediate value
                if episode > 0 and episode % 10 == 0:
                    mean_reward = np.mean(episode_rewards[-10:])
                    print(f"Episode {episode}: Mean reward = {mean_reward:.2f}")
                    trial.report(mean_reward, episode)
                    
                    # Handle pruning based on the intermediate value
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        
        finally:
            env.close()
        
        return np.mean(episode_rewards)
