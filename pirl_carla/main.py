import os
import torch
import numpy as np
import json
from datetime import datetime
from pirl_carla.carla_env import CarlaEnvWrapper
from pirl_carla.pirl_model import PhysicsInformedSAC
from pirl_carla.optuna_optimizer import PIRLOptimizer

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting PIRL training with CARLA...")
    print("1. Optimizing hyperparameters using Optuna")
    print("2. Training final model with best parameters")
    print("3. Saving results and model checkpoints")
    print("\nInitializing optimizer...")
    
    # Initialize optimizer
    optimizer = PIRLOptimizer(
        state_dim=4,  # [x, y, theta, v]
        action_dim=2,  # [acceleration, steering_angle]
        n_trials=100,  # Number of optimization trials
        n_episodes=1000,  # Episodes per trial
        max_steps=1000  # Steps per episode
    )
    
    # Run optimization
    print("\nStarting hyperparameter optimization...")
    best_params = optimizer.optimize()
    
    # Save best hyperparameters
    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    
    print("\nBest hyperparameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    
    # Initialize environment and agent
    env = CarlaEnvWrapper()
    agent = PhysicsInformedSAC(
        state_dim=4,
        action_dim=2,
        hidden_dim=best_params['hidden_dim']
    )
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=best_params['learning_rate'])
    
    # Initialize replay buffer
    buffer_size = best_params['buffer_size']
    replay_buffer = {
        'states': np.zeros((buffer_size, 4)),
        'actions': np.zeros((buffer_size, 2)),
        'next_states': np.zeros((buffer_size, 4)),
        'rewards': np.zeros(buffer_size),
        'dones': np.zeros(buffer_size),
        'ptr': 0,
        'size': 0
    }
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    try:
        for episode in range(1000):  # Final training episodes
            state = env.reset()
            episode_reward = 0
            
            for step in range(1000):  # Steps per episode
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
                if replay_buffer['size'] >= best_params['batch_size']:
                    # Sample batch
                    idxs = np.random.randint(0, replay_buffer['size'], size=best_params['batch_size'])
                    batch = {
                        'states': torch.FloatTensor(replay_buffer['states'][idxs]),
                        'actions': torch.FloatTensor(replay_buffer['actions'][idxs]),
                        'next_states': torch.FloatTensor(replay_buffer['next_states'][idxs]),
                        'rewards': torch.FloatTensor(replay_buffer['rewards'][idxs]).unsqueeze(-1),
                        'dones': torch.FloatTensor(replay_buffer['dones'][idxs]).unsqueeze(-1)
                    }
                    
                    # Get nearby obstacles
                    obstacles = env.world.get_actors().filter('vehicle.*')
                    obstacle_positions = []
                    for vehicle in obstacles:
                        if vehicle.id != env.vehicle.id:
                            loc = vehicle.get_location()
                            obstacle_positions.append([loc.x, loc.y])
                    batch['obstacles'] = torch.FloatTensor(obstacle_positions)
                    
                    # Compute losses
                    policy_loss, q_loss, alpha_loss = agent.compute_loss(
                        batch['states'],
                        batch['actions'],
                        batch['next_states'],
                        batch['rewards'],
                        batch['dones'],
                        batch['obstacles'],
                        gamma=best_params['gamma']
                    )
                    
                    # Weight losses
                    total_loss = (policy_loss + 
                                best_params['physics_weight'] * q_loss +
                                best_params['collision_weight'] * alpha_loss)
                    
                    # Update networks
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                
                if done:
                    break
                    
                state = next_state
            
            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-100:])  # Moving average over 100 episodes
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'reward': best_reward
                }, os.path.join(output_dir, "best_model.pt"))
            
            # Log progress
            if episode % 10 == 0:
                print(f"Episode {episode}: Mean Reward = {mean_reward:.2f}")
    
    finally:
        env.close()
    
    print("\nTraining completed!")
    print(f"Best mean reward achieved: {best_reward:.2f}")
    print(f"Model and results saved to: {output_dir}")

if __name__ == "__main__":
    main()
