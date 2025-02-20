import carla
import numpy as np
import math
import time
import weakref
import collections
from typing import Tuple, Dict, Any
from pirl_carla.collision_sensor import CollisionSensor

class CarlaEnv:
    """CARLA environment wrapper for Physics-Informed Reinforcement Learning"""
    
    def __init__(self, town='Town10', port=2000, render=True):
        try:
            print("Connecting to CARLA simulator...")
            self.client = carla.Client('localhost', port)
            self.client.set_timeout(20.0)
            
            # Try to connect multiple times
            max_retries = 5
            for i in range(max_retries):
                try:
                    self.world = self.client.get_world()
                    print(f"Successfully connected to CARLA simulator")
                    break
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"Attempt {i+1} failed, retrying...")
                        time.sleep(2)
                    else:
                        raise Exception("Failed to connect to CARLA simulator after multiple attempts")
            
            # Get current map name
            current_map = self.world.get_map().name.split('/')[-1]
            print(f"Current map: {current_map}")
            
            # Only load new map if different from current
            if current_map != town:
                print(f"Loading map {town}...")
                try:
                    self.world = self.client.load_world(town)
                    time.sleep(2)  # Wait for map to load
                    print(f"Successfully loaded map {town}")
                except Exception as e:
                    print(f"Failed to load map {town}, using existing map {current_map}")
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            
            # Get traffic manager and set synchronous mode
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_synchronous_mode(True)
            
            time.sleep(1)  # Wait for settings to apply
            
            self.blueprint_library = self.world.get_blueprint_library()
            self.map = self.world.get_map()
            self.vehicle = None
            self.collision_sensor = None
            self.sensors = {}
            self._setup_physics_parameters()
            self._steps = 0  # Initialize step counter
            
            # Get all spawn points and filter out those near obstacles
            self.spawn_points = self._filter_spawn_points()
            if not self.spawn_points:
                raise RuntimeError("No valid spawn points available in the map")
            
            print("Environment initialized successfully")
            
        except Exception as e:
            print(f"Error initializing CARLA environment: {str(e)}")
            raise

    def _filter_spawn_points(self):
        """Filter spawn points to avoid obstacles"""
        all_spawn_points = self.map.get_spawn_points()
        filtered_points = []
        
        for spawn_point in all_spawn_points:
            # Check for obstacles nearby
            location = spawn_point.location
            
            # Draw debug box to check spawn point
            debug = self.world.debug
            debug.draw_point(
                location + carla.Location(z=0.5),
                size=0.1,
                color=carla.Color(0, 255, 0),
                life_time=0.1
            )
            
            # Check for overlapping actors
            overlap = False
            for actor in self.world.get_actors():
                if actor.type_id.startswith('static') or actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
                    actor_loc = actor.get_location()
                    if actor_loc.distance(location) < 5.0:  # 5 meters safety margin
                        overlap = True
                        break
            
            if not overlap:
                filtered_points.append(spawn_point)
        
        return filtered_points

    def _setup_physics_parameters(self):
        """Setup enhanced physics parameters"""
        self.L = 2.5  # wheelbase length (m)
        self.dt = 0.05  # simulation timestep (s)
        self.safe_distance = 5.0  # safe distance for collision avoidance (m)
        self.max_speed = 30.0  # maximum speed (m/s)
        self.max_throttle = 1.0
        self.max_brake = 1.0
        self.max_steer = 1.0
        self.min_speed = 5.0  # minimum desired speed (m/s)
        self.stuck_speed_threshold = 0.1  # speed below which vehicle is considered stuck
        self.initial_grace_period = 50  # steps to allow vehicle to get moving
        self.mu = 0.7  # friction coefficient
        self.max_accel = 5.0  # maximum acceleration (m/s^2)
        self.max_brake_decel = 8.0  # maximum braking deceleration (m/s^2)

    def reset(self) -> np.ndarray:
        """Reset environment"""
        try:
            print("Resetting environment...")
            # Reset step counter
            self._steps = 0
            
            # Destroy existing actors
            if self.vehicle is not None:
                self.vehicle.destroy()
            if self.collision_sensor is not None:
                self.collision_sensor.destroy()
            for sensor in self.sensors.values():
                sensor.destroy()
                
            # Get random spawn point and destination
            start_point = np.random.choice(self.spawn_points)
            end_point = np.random.choice(self.spawn_points)
            while end_point.location.distance(start_point.location) < 50:  # Ensure minimum distance
                end_point = np.random.choice(self.spawn_points)
            
            # Get vehicle blueprint
            blueprint = self.blueprint_library.find('vehicle.tesla.model3')
            blueprint.set_attribute('role_name', 'hero')
            
            # Set recommended color
            if blueprint.has_attribute('color'):
                color = np.random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            
            # Try to spawn the vehicle in different locations until successful
            self.vehicle = None
            while self.vehicle is None:
                self.vehicle = self.world.try_spawn_actor(blueprint, start_point)
            
            print(f"Spawned vehicle at {start_point.location}")
            
            # Apply physics modifications
            self._modify_vehicle_physics(self.vehicle)
            
            # Setup collision sensor
            self.collision_sensor = CollisionSensor(self.vehicle)
            
            # Add other sensors
            self._setup_sensors()
            
            # Set initial waypoint and destination
            self.current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
            self.destination = end_point.location
            self.route = self._plan_route(self.current_waypoint, self.map.get_waypoint(self.destination))
            
            # Let physics settle
            for _ in range(20):  # 1 second at 20 FPS
                self.world.tick()
            
            print("Reset completed")
            return self._get_observation()
            
        except Exception as e:
            print(f"Error in reset: {str(e)}")
            raise

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take environment step"""
        try:
            # Increment step counter
            self._steps += 1
            
            # Get current state
            state = self._get_observation()
            
            # Update current waypoint
            self.current_waypoint = self.map.get_waypoint(self.vehicle.get_location())
            
            # Get next waypoint in route
            next_waypoint = None
            if self.route:
                next_waypoint = self.route[0]
                if self.current_waypoint.transform.location.distance(next_waypoint.transform.location) < 2.0:
                    self.route.pop(0)
                    if self.route:
                        next_waypoint = self.route[0]
            
            # Apply control
            if next_waypoint:
                # Calculate steering based on waypoint direction
                forward = self.vehicle.get_transform().get_forward_vector()
                right = self.vehicle.get_transform().get_right_vector()
                target = next_waypoint.transform.location - self.vehicle.get_location()
                
                # Calculate steering
                forward_dot = forward.x * target.x + forward.y * target.y
                right_dot = right.x * target.x + right.y * target.y
                
                steering = math.atan2(right_dot, forward_dot)
                steering = np.clip(steering, -self.max_steer, self.max_steer)
                
                # Apply control with throttle, steer, and brake
                throttle = np.clip(action[0], 0, 1)
                steer = np.clip(action[1], -1, 1)
                brake = np.clip(action[2], 0, 1)
                
                control = carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(steer),
                    brake=float(brake),
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False,
                    gear=1
                )
                self.vehicle.apply_control(control)
            
            # Update spectator camera
            self._update_spectator()
            
            # Tick the simulation
            self.world.tick()
                
            # Get new observation
            new_state = self._get_observation()
                
            # Calculate reward
            reward = self._compute_reward(state, action, new_state)
            
            # Check termination conditions
            done = self._check_termination()
            
            # Get info
            collision_history = self.collision_sensor.get_collision_history()
            current_speed = self.vehicle.get_velocity().length()
            info = {
                'speed': current_speed,
                'collision_intensity': sum(collision_history.values()) if collision_history else 0,
                'steps': self._steps,
                'distance_to_goal': self.current_waypoint.transform.location.distance(self.destination)
            }
            
            return new_state, reward, done, info
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            raise

    def _plan_route(self, start_waypoint, end_waypoint):
        """Plan a route from start to end waypoint"""
        # Calculate route
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        
        # Get next waypoints
        next_waypoints = []
        next_wp = start_waypoint
        for _ in range(20):  # Look ahead 20 waypoints
            next_wps = next_wp.next(2.0)  # Get waypoints 2 meters ahead
            if not next_wps:
                break
            next_wp = next_wps[0]
            next_waypoints.append(next_wp)
            
            # Stop if we're close to destination
            if next_wp.transform.location.distance(end_location) < 2.0:
                break
        
        return next_waypoints

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        loc = self.vehicle.get_location()
        rot = self.vehicle.get_transform().rotation
        vel = self.vehicle.get_velocity()
        
        # Get distance and direction to next waypoint
        next_waypoint = None
        if self.route:
            next_waypoint = self.route[0]
        
        if next_waypoint:
            target = next_waypoint.transform.location - loc
            distance_to_target = math.sqrt(target.x**2 + target.y**2)
            angle_to_target = math.atan2(target.y, target.x) - math.radians(rot.yaw)
        else:
            distance_to_target = 0.0
            angle_to_target = 0.0
        
        return np.array([
            loc.x,
            loc.y,
            math.radians(rot.yaw),
            math.sqrt(vel.x**2 + vel.y**2),
            distance_to_target,
            angle_to_target
        ])

    def _compute_reward(self, state: np.ndarray, action: np.ndarray, 
                       next_state: np.ndarray) -> float:
        """Compute reward value"""
        # Get collision history
        collision_history = self.collision_sensor.get_collision_history()
        if collision_history:
            # Large penalty for collisions
            return -200.0
        
        # Reward for velocity
        velocity_reward = next_state[3]  # Forward velocity
        
        # Reward for getting closer to target
        distance_reward = -next_state[4]  # Negative distance to target
        
        # Reward for heading towards target
        heading_reward = -abs(next_state[5])  # Negative absolute angle to target
        
        # Penalize large steering angles
        steering_penalty = -abs(action[1])
        
        # Penalize large accelerations
        acceleration_penalty = -abs(action[0])
        
        return velocity_reward + distance_reward + heading_reward + steering_penalty + acceleration_penalty

    def _check_termination(self) -> bool:
        """Check episode termination conditions"""
        # Check collision using collision sensor
        collision_history = self.collision_sensor.get_collision_history()
        if collision_history:
            total_intensity = sum(collision_history.values())
            if total_intensity > 0:
                print(f"Episode terminated due to collision with intensity {total_intensity}")
                return True
            
        # Check if vehicle is stuck - only after initial grace period
        if self._steps > self.initial_grace_period:
            vel = self.vehicle.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2)
            if speed < self.stuck_speed_threshold:
                print(f"Episode terminated due to vehicle being stuck (speed: {speed:.2f} m/s)")
                return True
            
        # Check if vehicle is off road
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
            print("Episode terminated due to vehicle going off road")
            return True
            
        # Check if reached destination
        if self.current_waypoint.transform.location.distance(self.destination) < 2.0:
            print("Episode completed - reached destination!")
            return True
            
        return False

    def close(self):
        """Cleanup environment"""
        try:
            if self.vehicle is not None:
                self.vehicle.destroy()
            if self.collision_sensor is not None:
                self.collision_sensor.destroy()
            for sensor in self.sensors.values():
                sensor.destroy()
            
            # Reset synchronous mode settings
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)
            
            print("Environment closed successfully")
        except Exception as e:
            print(f"Error in cleanup: {str(e)}")
