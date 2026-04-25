"""
Mesh primitive for loading custom shapes from PLY files.
This extends the shape_primitives module with support for custom geometries.
"""

from abc import ABC, abstractmethod
import pybullet as pb
import numpy as np


class MeshPrimitive:
    """Mesh primitive for custom shapes loaded from PLY files."""
    
    def __init__(self, ply_filepath, scale=1.0, mass=0.5, friction=0.6, restitution=0.5, height=0.07):
        """
        Args:
            ply_filepath: Path to the PLY file (absolute or relative)
            scale: Scale factor for the mesh
            mass: Object mass
            friction: Lateral friction coefficient
            restitution: Coefficient of restitution
            height: Height in z-direction for positioning on table
        """
        self.ply_filepath = ply_filepath
        self.scale = scale
        self.mass = mass
        self.friction = friction
        self.restitution = restitution
        self.height = height
        
        # Estimate contact radius from scale
        self._contact_radius = scale * 0.05
        
    def get_visual_shape(self, color=[0.5, 0.5, 0.5, 1]):
        """Get visual shape for mesh."""
        return {
            'shapeType': pb.GEOM_MESH,
            'fileName': self.ply_filepath,
            'meshScale': [self.scale, self.scale, self.scale],
            'rgbaColor': color
        }
    
    def get_collision_shape(self):
        """Get collision shape for mesh."""
        return {
            'shapeType': pb.GEOM_MESH,
            'fileName': self.ply_filepath,
            'meshScale': [self.scale, self.scale, self.scale],
            'flags': pb.GEOM_FORCE_32BIT_INDICES  # For mesh stability
        }
    
    def get_contact_radius(self):
        """Estimated contact radius based on scale."""
        return self._contact_radius
    
    def get_dimensions(self):
        """Get shape dimensions as dict."""
        return {'type': 'mesh', 'filepath': self.ply_filepath, 'scale': self.scale}
    
    def get_random_pusher_position(self, box_x, box_y, theta, pusher_radius, space_pusher_box, rng):
        """Generate pusher position at a random angle around the mesh."""
        angle = rng.random() * 2 * np.pi
        distance = self._contact_radius + pusher_radius + space_pusher_box
        pusher_start_x = distance * np.cos(angle)
        pusher_start_y = distance * np.sin(angle)
        
        pusher_start = np.array([pusher_start_x, pusher_start_y])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
        pusher_rotated = rotation_matrix @ pusher_start
        return pusher_rotated + np.array([box_x, box_y])
