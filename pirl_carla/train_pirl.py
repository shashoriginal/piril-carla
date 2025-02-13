import os
import torch
import numpy as np
import json
from datetime import datetime
from pirl_carla.carla_gym_env import CarlaEnv
from pirl_carla.pirl_model import PhysicsInformedSAC
from pirl_carla.optuna_optimizer import PIRLOptimizer

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("Starting PIRL training with CARLA")
    print("="*50)
    
    print("\nPhase 1: Environment Setup")
    print("-"*30)
    
    # Initialize environment
    env = CarlaEnv(town='Town01')
    
    # Get state and action dimensions from environment
    state = env.reset()
    state_dim = len(state)  # Should be 6: [x, y, theta, v, distance_to_target, angle_to_target]
    action_dim = 2  # [throttle, steering]
    
    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    print("\nPhase 2: Hyperparameter Optimization")
    print("-"*30)
    
    # Initialize optimizer with correct dimensions
    optimizer = PIRLOptimizer(
        state_dim=state_dim,  # Pass actual state dimension
        action_dim=action_dim,
        n_trials=100,
        n_episodes=1000,
        max_steps=1000
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
    
    print("\nPhase 3: Training Final Model")
    print("-"*30)
    
    # Initialize agent with correct dimensions
    agent = PhysicsInformedSAC(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=best_params['hidden_dim']
    )
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=best_params['learning_rate'])
    
    # Initialize replay buffer with correct dimensions
    buffer_size = best_params['buffer_size']
    replay_buffer = {
        'states': np.zeros((buffer_size, state_dim)),
        'actions': np.zeros((buffer_size, action_dim)),
        'next_states': np.zeros((buffer_size, state_dim)),
        'rewards': np.zeros(buffer_size),
        'dones': np.zeros(buffer_size),
        'ptr': 0,
        'size': 0
    }
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    total_steps = 0
    
    try:
        for episode in range(1000):  # Final training episodes
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_violations = 0
            
            for step in range(1000):  # Steps per episode
                # Select action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action, _ = agent(state_tensor)
                    action = action.squeeze(0).numpy()
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                if not info.get('constraints_satisfied', True):
                    episode_violations += 1
                
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
            
            # Log episode statistics
            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-100:])  # Moving average over 100 episodes
            
            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'reward': best_reward,
                    'hyperparameters': best_params
                }, os.path.join(output_dir, "best_model.pt"))
            
            # Log progress
            if episode % 10 == 0:
                print(f"\nEpisode {episode}")
                print(f"Steps: {episode_steps}")
                print(f"Total Steps: {total_steps}")
                print(f"Reward: {episode_reward:.2f}")
                print(f"Mean Reward (100 ep): {mean_reward:.2f}")
                print(f"Best Mean Reward: {best_reward:.2f}")
                print(f"Constraint Violations: {episode_violations}")
                print(f"Average Speed: {info['speed']:.2f} m/s")
                if info.get('collision_intensity', 0) > 0:
                    print(f"Collision Intensity: {info['collision_intensity']:.2f}")
    
    finally:
        env.close()
    
    print("\nTraining completed!")
    print(f"Best mean reward achieved: {best_reward:.2f}")
    print(f"Model and results saved to: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f'\nError occurred: {str(e)}')
