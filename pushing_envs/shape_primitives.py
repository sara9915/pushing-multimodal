"""
Abstract shape primitives and concrete implementations for pushing environments.
Allows pushing of different geometric objects (boxes, cylinders, and custom meshes)
"""

from abc import ABC, abstractmethod
import pybullet as pb
import numpy as np
import os


class ShapePrimitive(ABC):
    """Abstract base class for geometric primitives that can be pushed."""
    
    def __init__(self, mass, friction, restitution, height):
        """
        Args:
            mass: Object mass
            friction: Lateral friction coefficient
            restitution: Coefficient of restitution
            height: Height in z-direction for positioning on table
        """
        self.mass = mass
        self.friction = friction
        self.restitution = restitution
        self.height = height
        
    @abstractmethod
    def get_visual_shape(self, color):
        """
        Get PyBullet visual shape parameters.
        
        Returns:
            dict with keys: shapeType, and shape-specific parameters (e.g., halfExtents, radius)
        """
        pass
    
    @abstractmethod
    def get_collision_shape(self):
        """
        Get PyBullet collision shape parameters.
        
        Returns:
            dict with keys: shapeType, and shape-specific parameters
        """
        pass
    
    @abstractmethod
    def get_contact_radius(self):
        """
        Get characteristic radius for boundary checking and pusher placement.
        Used to determine if object is within table boundaries.
        
        Returns:
            float: characteristic contact radius
        """
        pass
    
    @abstractmethod
    def get_dimensions(self):
        """
        Get shape dimensions as dict for domain randomization and logging.
        
        Returns:
            dict: dimension parameters (e.g., {'x': 0.1, 'y': 0.12, 'z': 0.07})
        """
        pass
    
    @abstractmethod
    def get_random_pusher_position(self, box_x, box_y, theta, pusher_radius, space_pusher_box, rng):
        """
        Generate random pusher starting position relative to object.
        
        Args:
            box_x, box_y: Object center position
            theta: Object orientation (in radians)
            pusher_radius: Radius of the pusher sphere
            space_pusher_box: Minimum distance between pusher and object
            rng: numpy random number generator
            
        Returns:
            tuple: (pusher_x, pusher_y) absolute position in world frame
        """
        pass
    
    def get_z_offset(self):
        """
        Get vertical offset to apply when creating the object.
        Used for mesh files that may have internal offsets.
        
        Returns:
            float: z offset in meters (default 0)
        """
        return 0.0


class BoxPrimitive(ShapePrimitive):
    """Box/rectangular prism primitive."""
    
    def __init__(self, dim_x, dim_y, dim_z, mass, friction, restitution):
        """
        Args:
            dim_x, dim_y, dim_z: Box dimensions in each axis
            mass: Object mass
            friction: Lateral friction coefficient
            restitution: Coefficient of restitution
        """
        super().__init__(mass, friction, restitution, dim_z)
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        
    def get_visual_shape(self, color=[1, 1, 0, 1]):
        return {
            'shapeType': pb.GEOM_BOX,
            'halfExtents': [self.dim_x/2, self.dim_y/2, self.dim_z/2],
            'rgbaColor': color
        }
    
    def get_collision_shape(self):
        return {
            'shapeType': pb.GEOM_BOX,
            'halfExtents': [self.dim_x/2, self.dim_y/2, self.dim_z/2]
        }
    
    def get_contact_radius(self):
        """Use half of the smallest horizontal dimension."""
        return min(self.dim_x, self.dim_y) / 2
    
    def get_dimensions(self):
        return {'x': self.dim_x, 'y': self.dim_y, 'z': self.dim_z}
    
    def get_random_pusher_position(self, box_x, box_y, theta, pusher_radius, space_pusher_box, rng):
        """Generate pusher position on one of the four sides."""
        box_side = rng.random()
        if box_side <= 0.25:
            # Back side
            pusher_start_x = -self.dim_x/2 - pusher_radius - space_pusher_box
            pusher_start_y = rng.random() * self.dim_y - self.dim_y/2
        elif box_side <= 0.5:
            # Front side
            pusher_start_x = self.dim_x/2 + pusher_radius + space_pusher_box
            pusher_start_y = rng.random() * self.dim_y - self.dim_y/2
        elif box_side <= 0.75:
            # Top side
            pusher_start_x = rng.random() * self.dim_x - self.dim_x/2
            pusher_start_y = self.dim_y/2 + pusher_radius + space_pusher_box
        else:
            # Bottom side
            pusher_start_x = rng.random() * self.dim_x - self.dim_x/2
            pusher_start_y = -self.dim_y/2 - pusher_radius - space_pusher_box
        
        pusher_start = np.array([pusher_start_x, pusher_start_y])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
        pusher_rotated = rotation_matrix @ pusher_start
        return pusher_rotated + np.array([box_x, box_y])


class CylinderPrimitive(ShapePrimitive):
    """Cylinder primitive (axis aligned with z)."""
    
    def __init__(self, radius, height, mass, friction, restitution):
        """
        Args:
            radius: Cylinder radius
            height: Cylinder height (z-direction)
            mass: Object mass
            friction: Lateral friction coefficient
            restitution: Coefficient of restitution
        """
        super().__init__(mass, friction, restitution, height)
        self.radius = radius
        self.height = height
        
    def get_visual_shape(self, color=[1, 0, 0, 1]):
        return {
            'shapeType': pb.GEOM_CYLINDER,
            'radius': self.radius,
            'length': self.height,
            'rgbaColor': color
        }
    
    def get_collision_shape(self):
        return {
            'shapeType': pb.GEOM_CYLINDER,
            'radius': self.radius,
            'height': self.height
        }
    
    def get_contact_radius(self):
        """For a cylinder, use the radius."""
        return self.radius
    
    def get_dimensions(self):
        return {'radius': self.radius, 'height': self.height}
    
    def get_random_pusher_position(self, box_x, box_y, theta, pusher_radius, space_pusher_box, rng):
        """Generate pusher position at a random angle around the cylinder."""
        angle = rng.random() * 2 * np.pi
        distance = self.radius + pusher_radius + space_pusher_box
        pusher_start_x = distance * np.cos(angle)
        pusher_start_y = distance * np.sin(angle)
        
        pusher_start = np.array([pusher_start_x, pusher_start_y])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
        pusher_rotated = rotation_matrix @ pusher_start
        return pusher_rotated + np.array([box_x, box_y])


class SpherePrimitive(ShapePrimitive):
    """Sphere primitive."""
    
    def __init__(self, radius, mass, friction, restitution):
        """
        Args:
            radius: Sphere radius
            mass: Object mass
            friction: Lateral friction coefficient
            restitution: Coefficient of restitution
        """
        super().__init__(mass, friction, restitution, radius * 2)
        self.radius = radius
        
    def get_visual_shape(self, color=[0, 1, 1, 1]):
        return {
            'shapeType': pb.GEOM_SPHERE,
            'radius': self.radius,
            'rgbaColor': color
        }
    
    def get_collision_shape(self):
        return {
            'shapeType': pb.GEOM_SPHERE,
            'radius': self.radius
        }
    
    def get_contact_radius(self):
        return self.radius
    
    def get_dimensions(self):
        return {'radius': self.radius}
    
    def get_random_pusher_position(self, box_x, box_y, theta, pusher_radius, space_pusher_box, rng):
        """Generate pusher position at a random angle around the sphere."""
        angle = rng.random() * 2 * np.pi
        distance = self.radius + pusher_radius + space_pusher_box
        pusher_start_x = distance * np.cos(angle)
        pusher_start_y = distance * np.sin(angle)
        
        pusher_start = np.array([pusher_start_x, pusher_start_y])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
        pusher_rotated = rotation_matrix @ pusher_start
        return pusher_rotated + np.array([box_x, box_y])


class CapsulePrimitive(ShapePrimitive):
    """Capsule primitive (cylinder with hemispherical ends)."""
    
    def __init__(self, radius, length, mass, friction, restitution):
        """
        Args:
            radius: Capsule radius
            length: Capsule length (including hemispherical ends)
            mass: Object mass
            friction: Lateral friction coefficient
            restitution: Coefficient of restitution
        """
        super().__init__(mass, friction, restitution, radius * 2)
        self.radius = radius
        self.length = length
        
    def get_visual_shape(self, color=[0.5, 0.5, 1, 1]):
        return {
            'shapeType': pb.GEOM_CAPSULE,
            'radius': self.radius,
            'length': self.length,
            'rgbaColor': color
        }
    
    def get_collision_shape(self):
        return {
            'shapeType': pb.GEOM_CAPSULE,
            'radius': self.radius,
            'height': self.length
        }
    
    def get_contact_radius(self):
        return self.radius
    
    def get_dimensions(self):
        return {'radius': self.radius, 'length': self.length}
    
    def get_random_pusher_position(self, box_x, box_y, theta, pusher_radius, space_pusher_box, rng):
        """Generate pusher position at a random angle around the capsule."""
        angle = rng.random() * 2 * np.pi
        distance = self.radius + pusher_radius + space_pusher_box
        pusher_start_x = distance * np.cos(angle)
        pusher_start_y = distance * np.sin(angle)
        
        pusher_start = np.array([pusher_start_x, pusher_start_y])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
        pusher_rotated = rotation_matrix @ pusher_start
        return pusher_rotated + np.array([box_x, box_y])


class WhalePrimitive(ShapePrimitive):
    """Whale mesh primitive loaded from STL file."""
    
    def __init__(self, scale=1.0, mass=0.5, friction=0.6, restitution=0.5, height=0.07):
        """
        Args:
            scale: Scale factor for the mesh
            mass: Object mass
            friction: Lateral friction coefficient
            restitution: Coefficient of restitution
            height: Height in z-direction for positioning on table
        """
        super().__init__(mass, friction, restitution, height)
        
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.stl_filepath = os.path.join(current_dir, 'meshes', 'whale.stl')
        self.scale = scale
        
        # Estimate contact radius from scale (whale is roughly 0.1-0.15 units)
        self._contact_radius = scale * 0.06
        
    def get_visual_shape(self, color=[0.2, 0.4, 0.6, 1]):  # Blue-ish color for whale
        return {
            'shapeType': pb.GEOM_MESH,
            'fileName': self.stl_filepath,
            'meshScale': [self.scale, self.scale, self.scale],
            'rgbaColor': color
        }
    
    def get_collision_shape(self):
        return {
            'shapeType': pb.GEOM_MESH,
            'fileName': self.stl_filepath,
            'meshScale': [self.scale, self.scale, self.scale]
        }
    
    def get_contact_radius(self):
        """Estimated contact radius based on whale scale."""
        return self._contact_radius
    
    def get_dimensions(self):
        return {'type': 'whale_mesh', 'scale': self.scale}
    
    def get_random_pusher_position(self, box_x, box_y, theta, pusher_radius, space_pusher_box, rng):
        """Generate pusher position at a random angle around the whale."""
        angle = rng.random() * 2 * np.pi
        distance = self._contact_radius + pusher_radius + space_pusher_box
        pusher_start_x = distance * np.cos(angle)
        pusher_start_y = distance * np.sin(angle)
        
        pusher_start = np.array([pusher_start_x, pusher_start_y])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
        pusher_rotated = rotation_matrix @ pusher_start
        return pusher_rotated + np.array([box_x, box_y])
    
    def get_z_offset(self):
        """Z offset to compensate for mesh file offset. Adjust this value based on your whale.stl"""
        return -0.02  # Adjust this value until whale rests properly on the plane

