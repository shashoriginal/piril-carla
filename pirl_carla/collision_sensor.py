import carla
import math
import weakref
import collections

def get_actor_display_name(actor, truncate=250):
    """Get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

class CollisionSensor:
    """Collision sensor to detect collisions with other actors"""
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        
        # We need to pass the lambda a weak reference to self to avoid circular reference
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        
        # Get collision details
        actor_type = get_actor_display_name(event.other_actor)
        print(f'Collision with {actor_type}')
        
        # Calculate collision intensity
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        
        # Add to history
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:  # Limit history size
            self.history.pop(0)

    def destroy(self):
        """Destroy the sensor"""
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None
