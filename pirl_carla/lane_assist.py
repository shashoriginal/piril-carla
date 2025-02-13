import carla
import math
import numpy as np
from typing import Tuple, Optional

class LaneAssist:
    """Lane assistance system for CARLA vehicle control"""
    
    def __init__(self):
        # PID controller parameters for lane keeping
        self.Kp = 1.0  # Proportional gain
        self.Ki = 0.0  # Integral gain
        self.Kd = 0.1  # Derivative gain
        self.prev_error = 0.0
        self.integral = 0.0
        
    def get_lane_info(self, vehicle, map) -> Tuple[float, float, float]:
        """
        Get lane information for the vehicle
        Returns: (lane_distance, lane_angle, road_width)
        """
        # Get current vehicle waypoint
        location = vehicle.get_location()
        waypoint = map.get_waypoint(location)
        
        if not waypoint:
            return 0.0, 0.0, 0.0
            
        # Calculate distance from center of lane
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        lane_center = waypoint.transform.location
        
        # Project vehicle position onto lane vector
        lane_vector = waypoint.transform.get_forward_vector()
        vehicle_vector = vehicle_location - lane_center
        
        # Calculate signed distance to lane center
        cross_product = lane_vector.x * vehicle_vector.y - lane_vector.y * vehicle_vector.x
        lane_distance = cross_product
        
        # Calculate angle between vehicle and lane
        vehicle_forward = vehicle_transform.get_forward_vector()
        lane_angle = math.atan2(
            lane_vector.x * vehicle_forward.y - lane_vector.y * vehicle_forward.x,
            lane_vector.x * vehicle_forward.x + lane_vector.y * vehicle_forward.y
        )
        
        return lane_distance, lane_angle, waypoint.lane_width
        
    def compute_steering(self, lane_distance: float, lane_angle: float, speed: float) -> float:
        """
        Compute steering command using PID control
        """
        # Combine distance and angle error
        error = lane_distance + speed * math.sin(lane_angle)
        
        # PID control
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        
        # Calculate steering command
        steering = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )
        
        # Normalize steering to [-1, 1]
        steering = np.clip(steering, -1.0, 1.0)
        
        return steering
        
    def get_lane_invasion_info(self, vehicle, map) -> bool:
        """
        Check if vehicle is invading other lanes
        Returns: True if vehicle is in wrong lane, False otherwise
        """
        location = vehicle.get_location()
        waypoint = map.get_waypoint(location)
        
        if not waypoint:
            return True  # Consider it lane invasion if no waypoint found
            
        # Get left and right lanes if they exist
        left_lane = waypoint.get_left_lane()
        right_lane = waypoint.get_right_lane()
        
        # If we're not in a valid lane, consider it lane invasion
        if waypoint.lane_type != carla.LaneType.Driving:
            return True
            
        # Check if we're in the correct lane by comparing lane IDs
        # Only check if adjacent lanes exist
        if left_lane and waypoint.lane_id == left_lane.lane_id:
            return True
        if right_lane and waypoint.lane_id == right_lane.lane_id:
            return True
            
        return False
               
    def reset(self):
        """Reset PID controller"""
        self.prev_error = 0.0
        self.integral = 0.0
