import gym
from gym.spaces import Box
from gym.spaces import MultiDiscrete
import numpy as np
import pybullet as pb
from .shape_primitives import BoxPrimitive, CylinderPrimitive, WhalePrimitive

class PushingEnv(gym.Env):

    def __init__(self, graphics = False, seed = None, fps = 30, randomize_shapes = True, shapes = None):
        super(PushingEnv, self).__init__()

        # Number of previous observations to use in the policy
        self.STACK_SIZE = 1

        # Action space: x velocity, y velocity
        self.action_space = MultiDiscrete([11,11])
        
        # Observation space: (target_x_box, target_y_box, target_theta), (x_box, y_box, orientation_box) * STACK_SIZE, (x_pusher, y_pusher) * STACK_SIZE
        self.observation_space = Box(low = -1, high = 1, shape = (3 + 3*self.STACK_SIZE + 2*self.STACK_SIZE,), dtype = np.float64)

        # Number of steps ellapsed in the current episode
        self.step_num = 0

        # Store the seed for the random number generator 
        self.seed = seed

        # Determines whether to generate a GUI
        self.graphics = graphics

        # Whether to apply random disturbances during training
        self.disturbances = True
        
        # Shape randomization settings
        self.randomize_shapes = randomize_shapes
        # Available shapes to randomly select from during training
        self.available_shapes = shapes if shapes is not None else ['box', 'cylinder', 'whale']


        # Distance between pusher and ground
        self.PUSHER_GROUND_CLEARENCE = 0.01
        # Width of the table (in the x direction)
        self.TABLE_DIM_X = 0.60
        # Depth of the table (in the y direction)
        self.TABLE_DIM_Y = 0.35
        # Height of the box (in the z direction)
        self.BOX_DIM_Z = 0.07

        # Defines an inner box within which the random starting positions and target positions are generated
        self.GEN_POS_MARGIN = 0.1
        # Width of x interval within which start and target positions are generated
        self.GEN_POS_X_WIDTH = self.TABLE_DIM_X - 2*self.GEN_POS_MARGIN
        # Width of y interval within which start and target positions are generated
        self.GEN_POS_Y_WIDTH = self.TABLE_DIM_Y - 2*self.GEN_POS_MARGIN
        # Starting space between the pusher and the box
        self.SPACE_PUSHER_BOX = 1e-3

        # Width of possible starting orientations
        self.GEN_THETA_WIDTH = 2*np.pi

        # Current step of the curriculum
        self.CURRICULUM_STEP = 1

        # Distance to target position that is considered successful
        self.DISTANCE_SUCCESS = 0.015
        # Distance to target theta that is considered successful
        self.THETA_DISTANCE_SUCCESS = 0.34

        # Success rate which prompts an increase in difficulty
        self.CURRICULUM_SUCCESS_THRESH = 90
        
        # Number of episodes to calculate current success rate
        self.NUM_EPISODES_SUCCESS_RATE = 100

        # Array to keep track of previous 100 episodes outcome
        self.EPISODE_SUCCESS_RATE = np.zeros(self.NUM_EPISODES_SUCCESS_RATE)

        # Number of episodes completed
        self.EPISODE_NUM = 0

        # Current success rate calculated over the past 100 episodes
        self.CURRENT_SUCCESS_RATE = 0

        # Object that is being pushed
        self.box = None
        # End effector that pushes the object
        self.pusher = None
        # Surface on which the object slides
        self.floor = None
        # Visualization of target
        self.target_vis = None
        # Shape primitive for the object being pushed
        self.shape_object = None
        # Type of shape to use ('box', 'cylinder', 'sphere', 'capsule')
        self.shape_type = 'box'

        # x coordinate of the box at the starting pose
        self.start_x_box = None
        # y coordinate of the box at the starting pose
        self.start_y_box = None
        # Orientation of the box at the starting pose
        self.start_theta = None
        # x coordinate of the box at the target pose
        self.target_x_box = None
        # y coordinate of the box at the target pose
        self.target_y_box = None
        # Orientation of the box at the target pose
        self.target_theta = None

        # Dictionary to store the information about box and pusher in the simulation
        self.sim_data = {
            "x_box" : None,
            "y_box" : None,
            "theta_box" : None,
            "velocity_box" : None,
            "distance_to_target" : None,
            "theta_distance_to_target" : None,
            "x_pusher" : None,
            "y_pusher" : None,
            "velocity_pusher" : None
        }

        # Parameters for domain randomization
        self.FRICTION_CENTER = 0.6
        self.FRICTION_WIDTH = 0.2

        self.RESTITUTION_CENTER = 0.5
        self.RESTITUTION_WIDTH = 0.2

        self.BOX_Y_CENTER = 0.12
        self.BOX_Y_WIDTH = 0.05  # Changed from 0.01 to 0.02 for wider variation (±0.01)

        self.BOX_X_CENTER = 0.1
        self.BOX_X_WIDTH = 0.05  # Changed from 0.01 to 0.02 for wider variation (±0.01)

        self.BOX_MASS_CENTER = 0.5
        self.BOX_MASS_WIDTH = 0.2

        self.PUSHER_RADIUS_CENTER = 0.0125
        self.PUSHER_RADIUS_WIDTH = 0.001

        # Parameters for cylinder and sphere radius (for planar stability)
        # Use dimensions comparable to box dimensions
        self.SHAPE_RADIUS_CENTER = 0.05  # Average radius for cylinders and spheres
        self.SHAPE_RADIUS_WIDTH = 0.05   # Variation in radius

        self.FORCE_WIDTH = 50

        # Parameters for observation noise
        self.DISTANCE_NOISE_MEAN = 0
        self.DISTANCE_NOISE_STD = 0.001

        self.THETA_NOISE_MEAN = 0
        self.THETA_NOISE_STD = 0.02

        # Episode and step noise quantities
        self.BOX_X_NOISE_EPISODE = None
        self.BOX_X_NOISE_STEP = None
        self.BOX_Y_NOISE_EPISODE = None
        self.BOX_Y_NOISE_STEP = None
        self.BOX_THETA_NOISE_EPISODE = None
        self.BOX_THETA_NOISE_STEP = None
        self.PUSHER_X_NOISE_EPISODE = None
        self.PUSHER_X_NOISE_STEP = None
        self.PUSHER_Y_NOISE_EPISODE = None
        self.PUSHER_Y_NOISE_STEP = None

        # Stack of previous box poses
        self.box_pose_stack = None

        # Stack of previous pusher positions
        self.pusher_pos_stack = None

        # Number of pybullet steps for every agent step
        self.pybullet_steps = int(240 / fps)

        # Maximum number of steps before the environment is reset
        self.max_episode_length = 300

        # Setup PyBullet
        if self.graphics:
            pb.connect(pb.GUI) # Generate a graphical interface
            self._draw_boundary() # Draw the table boundaries
        else:
            pb.connect(pb.DIRECT) # Communicate directly with the physics engine
        pb.setGravity(0,0,-9.81)

        # Reset the environment
        self.reset()

    def step(self, action):

        # Scale the action to the desired range
        action_scale = self._adjust_action(action)
        self.velocity_x = action_scale[0]
        self.velocity_y = action_scale[1]

        # Create a random disturbance to the box
        if self.disturbances and (self.rng.random() < 0.01) and self._can_apply_random_disturbance():
            self._apply_random_disturbance()

        # Update simulation
        pb.resetBaseVelocity(objectUniqueId = self.pusher, linearVelocity = [self.velocity_x, self.velocity_y, 0])
        action_steps = int(np.around(self.rng.normal(self.pybullet_steps,0.75), decimals = 0))
        if action_steps < 0:
            action_steps = self.pybullet_steps
        for _ in range(action_steps):
            pb.stepSimulation()
        self._update_sim_data()

        # Calculate reward
        reward, done = self._calculate_reward()

        # Obtain new observation
        observation = self._get_observation()

        # Log relevant information
        info = {}

        # Timeout requires careful consideration.
        self.step_num += 1
        if (not done) and (self.step_num >= self.max_episode_length):
            info["TimeLimit.truncated"] = True
            done = True
            self._record_unsuccessful_episode()

        return observation, reward, done, info

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

        self.step_num = 0
        
        # Randomly select shape if randomize_shapes is enabled
        if self.randomize_shapes:
            self.shape_type = self.rng.choice(self.available_shapes)

        # Remove the box, pusher, target and floor from the simulator
        if self.box != None:
            pb.removeBody(self.box)
        if self.pusher != None:
            pb.removeBody(self.pusher)
        if self.target_vis != None:
            pb.removeBody(self.target_vis)
        if self.floor != None:
            self._reset_floor()
        else:
            self._create_floor()

        # Generate random dimensions based on shape type
        # Initialize default dimensions that may be used by multiple shape types
        self.BOX_DIM_X = self.BOX_X_CENTER + self.rng.random()*self.BOX_X_WIDTH - self.BOX_X_WIDTH/2
        self.BOX_DIM_Y = self.BOX_Y_CENTER + self.rng.random()*self.BOX_Y_WIDTH - self.BOX_Y_WIDTH/2
        
        if self.shape_type == 'box':
            box_mass = self.BOX_MASS_CENTER + self.rng.random()*self.BOX_MASS_WIDTH - self.BOX_MASS_WIDTH/2
            friction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
            self.shape_object = BoxPrimitive(self.BOX_DIM_X, self.BOX_DIM_Y, self.BOX_DIM_Z, box_mass, friction, restitution)
        elif self.shape_type == 'cylinder':
            # Use SHAPE_RADIUS parameters for domain randomization
            obj_radius = self.SHAPE_RADIUS_CENTER + self.rng.random()*self.SHAPE_RADIUS_WIDTH - self.SHAPE_RADIUS_WIDTH/2
            box_mass = self.BOX_MASS_CENTER + self.rng.random()*self.BOX_MASS_WIDTH - self.BOX_MASS_WIDTH/2
            friction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
            self.shape_object = CylinderPrimitive(obj_radius, self.BOX_DIM_Z, box_mass, friction, restitution)
        elif self.shape_type == 'whale':
            # Whale mesh with scale variation
            whale_scale = 0.8 + self.rng.random() * 0.4  # Scale between 0.8 and 1.2
            box_mass = self.BOX_MASS_CENTER + self.rng.random()*self.BOX_MASS_WIDTH - self.BOX_MASS_WIDTH/2
            friction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
            self.shape_object = WhalePrimitive(scale=whale_scale, mass=box_mass, friction=friction, restitution=restitution)

        # Generate random dimensions of the pusher
        self.PUSHER_RADIUS = self.PUSHER_RADIUS_CENTER + self.rng.random()*self.PUSHER_RADIUS_WIDTH - self.PUSHER_RADIUS_WIDTH/2

        # Generate episode noise for the observations
        self.BOX_X_NOISE_EPISODE = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.BOX_Y_NOISE_EPISODE = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.BOX_THETA_NOISE_EPISODE = self.rng.normal(self.THETA_NOISE_MEAN, self.THETA_NOISE_STD)
        self.PUSHER_X_NOISE_EPISODE = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.PUSHER_Y_NOISE_EPISODE = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)

        # Generate a random starting configuration
        self.start_x_box, self.start_y_box, self.start_theta, pusher_x, pusher_y = self._generate_random_start()
        self._create_object(x_start=self.start_x_box, y_start=self.start_y_box, theta_start=self.start_theta)
        self._create_pusher(x_start=pusher_x, y_start=pusher_y)
        self.box_pose_stack = [[self.start_x_box, self.start_y_box, self.start_theta] for _ in range(self.STACK_SIZE)]
        self.pusher_pos_stack = [[pusher_x, pusher_y] for _ in range(self.STACK_SIZE)]

        # Generate a random target
        self.target_x_box, self.target_y_box, self.target_theta = self._generate_random_target()

        # Update information of simulation objects
        self._update_sim_data()     
        
        # Obtain observation of the environment
        observation = self._get_observation()

        return observation

    def half_success_threshold(self):
        self.CURRICULUM_STEP += 1
        self.DISTANCE_SUCCESS = 0.015/2
        self.THETA_DISTANCE_SUCCESS = 0.34/2


    def _update_sim_data(self):
        # Query PyBullet for box data
        box_position_and_orientation = pb.getBasePositionAndOrientation(self.box)
        x_box, y_box, _ = box_position_and_orientation[0]
        theta_box = pb.getEulerFromQuaternion(box_position_and_orientation[1])[2]
        velociy_box, _ = pb.getBaseVelocity(self.box)
        velociy_box = np.linalg.norm(velociy_box[:2])

        distance_to_target = self._distance_to(x_box, y_box, self.target_x_box, self.target_y_box)

        theta_distance_to_target = self._theta_distance(self.target_theta, theta_box)

        # Query PyBullet for pusher data
        pusher_position_orientation = pb.getBasePositionAndOrientation(self.pusher)
        x_pusher, y_pusher, _ = pusher_position_orientation[0]
        velociy_pusher, _ = pb.getBaseVelocity(self.pusher)
        velociy_pusher = np.linalg.norm(velociy_pusher[:2])

        # Update records of simulation data
        self.sim_data["x_box"] = x_box
        self.sim_data["y_box"] = y_box
        self.sim_data["theta_box"] = theta_box
        self.sim_data["velocity_box"] = velociy_box
        self.sim_data["distance_to_target"] = distance_to_target
        self.sim_data["theta_distance_to_target"] = theta_distance_to_target
        self.sim_data["x_pusher"] = x_pusher
        self.sim_data["y_pusher"] = y_pusher
        self.sim_data["velocity_pusher"] = velociy_pusher
        

    def _can_apply_random_disturbance(self):
        if self.sim_data["distance_to_target"] < 0.15:
            return False
        if self.TABLE_DIM_X/2 - abs(self.sim_data["x_box"]) < 0.1:
            return False
        if self.TABLE_DIM_Y/2 - abs(self.sim_data["y_box"]) < 0.1:
            return False
        return True

    def _apply_random_disturbance(self):

        # Generate random force direction
        force_x = self.rng.random()*self.FORCE_WIDTH - self.FORCE_WIDTH/2
        force_y = self.rng.random()*self.FORCE_WIDTH - self.FORCE_WIDTH/2

        # Generate random position at which to apply the force
        # For boxes, apply on faces; for cylinders/spheres, apply in x-y plane
        if self.shape_type == 'box':
            pos_x = self.rng.random()*self.BOX_DIM_X - self.BOX_DIM_X/2
            pos_y = self.rng.random()*self.BOX_DIM_Y - self.BOX_DIM_Y/2
            pos_z = self.rng.random()*self.BOX_DIM_Z - self.BOX_DIM_Z/2
        else:
            # For cylindrical shapes, apply force in x-y plane at random position
            contact_radius = self.shape_object.get_contact_radius()
            angle = self.rng.random() * 2 * np.pi
            pos_x = contact_radius * np.cos(angle)
            pos_y = contact_radius * np.sin(angle)
            pos_z = 0

        # Apply the force (direction and position relative to the object)
        pb.applyExternalForce(objectUniqueId = self.box, 
                              linkIndex = -1,
                              forceObj = [force_x, force_y, 0],
                              posObj = [pos_x, pos_y, pos_z], 
                              flags = pb.LINK_FRAME)

    def _record_unsuccessful_episode(self):
        self.EPISODE_SUCCESS_RATE[self.EPISODE_NUM%self.NUM_EPISODES_SUCCESS_RATE] = 0
        self.EPISODE_NUM += 1
        self.CURRENT_SUCCESS_RATE = 100*sum(self.EPISODE_SUCCESS_RATE) / self.NUM_EPISODES_SUCCESS_RATE

    def _record_successful_episode(self):
        self.EPISODE_SUCCESS_RATE[self.EPISODE_NUM%self.NUM_EPISODES_SUCCESS_RATE] = 1
        self.EPISODE_NUM += 1
        self.CURRENT_SUCCESS_RATE = 100*sum(self.EPISODE_SUCCESS_RATE) / self.NUM_EPISODES_SUCCESS_RATE

    def _distance_to(self, x1, y1, x2, y2):
        return ((x2-x1)**2 + (y2-y1)**2)**(1/2)

    def _theta_distance(self, theta1, theta2):
        theta_distance = abs(theta1-theta2)
        if theta_distance <= np.pi:
            return theta_distance
        else:
            return (2*np.pi - theta_distance)

    def _create_floor(self):
        floor_visual = pb.createVisualShape(shapeType = pb.GEOM_PLANE, planeNormal = [0, 0, 1])
        floor_collision = pb.createCollisionShape(shapeType = pb.GEOM_PLANE, planeNormal = [0, 0, 1])
        self.floor = pb.createMultiBody(
            baseMass = 0,
            baseCollisionShapeIndex = floor_collision,
            baseVisualShapeIndex = floor_visual
        )
        pb.changeDynamics(
            bodyUniqueId = self.floor,
            linkIndex = -1,
            lateralFriction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2,
            spinningFriction = 0,
            rollingFriction = 0,
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
        )

    def _reset_floor(self):
        pb.changeDynamics(
            bodyUniqueId = self.floor,
            linkIndex = -1,
            lateralFriction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2,
            spinningFriction = 0,
            rollingFriction = 0,
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
        )

    def _create_object(self, x_start, y_start, theta_start):
        """Create the object to be pushed using the current shape_object definition."""
        visual_shape_params = self.shape_object.get_visual_shape()
        collision_shape_params = self.shape_object.get_collision_shape()
        
        box_visual = pb.createVisualShape(**visual_shape_params)
        box_collision = pb.createCollisionShape(**collision_shape_params)
        
        # Apply z offset for mesh files
        z_offset = self.shape_object.get_z_offset()
        z_position = self.shape_object.height/2 + z_offset
        
        self.box = pb.createMultiBody(
            baseMass = self.shape_object.mass,
            baseCollisionShapeIndex = box_collision,
            baseVisualShapeIndex = box_visual,
            basePosition = [x_start, y_start, z_position],
            baseOrientation = pb.getQuaternionFromEuler([0, 0, theta_start])
        )
        pb.changeDynamics(
            bodyUniqueId = self.box,
            linkIndex = -1,
            lateralFriction = self.shape_object.friction,
            spinningFriction = 0,
            rollingFriction = 0,
            restitution = self.shape_object.restitution
        )

    def _create_pusher(self, x_start, y_start):
        pusher_visual = pb.createVisualShape(shapeType = pb.GEOM_SPHERE, radius = self.PUSHER_RADIUS, rgbaColor = [0, 0, 1, 1])
        pusher_collision = pb.createCollisionShape(shapeType = pb.GEOM_SPHERE, radius = self.PUSHER_RADIUS)
        self.pusher = pb.createMultiBody(
            baseMass = 0,
            baseCollisionShapeIndex = pusher_collision,
            baseVisualShapeIndex = pusher_visual,
            basePosition = [x_start, y_start, self.BOX_DIM_Z/2],
            baseOrientation = pb.getQuaternionFromEuler([0, 0, 0])
        )
        pb.changeDynamics(
            bodyUniqueId = self.pusher,
            linkIndex = -1,
            lateralFriction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2,
            spinningFriction = 0,
            rollingFriction = 0,
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
        )

    def _draw_boundary(self):
        pb.addUserDebugLine(lineFromXYZ = [-self.TABLE_DIM_X/2, -self.TABLE_DIM_Y/2, 0.01], lineToXYZ = [self.TABLE_DIM_X/2, -self.TABLE_DIM_Y/2, 0.01], lineColorRGB = [0, 0, 0], lineWidth = 5)
        pb.addUserDebugLine(lineFromXYZ = [-self.TABLE_DIM_X/2, self.TABLE_DIM_Y/2, 0.01], lineToXYZ = [self.TABLE_DIM_X/2, self.TABLE_DIM_Y/2, 0.01], lineColorRGB = [0, 0, 0], lineWidth = 5)
        pb.addUserDebugLine(lineFromXYZ = [-self.TABLE_DIM_X/2, -self.TABLE_DIM_Y/2, 0.01], lineToXYZ = [-self.TABLE_DIM_X/2, self.TABLE_DIM_Y/2, 0.01], lineColorRGB = [0, 0, 0], lineWidth = 5)
        pb.addUserDebugLine(lineFromXYZ = [self.TABLE_DIM_X/2, -self.TABLE_DIM_Y/2, 0.01], lineToXYZ = [self.TABLE_DIM_X/2, self.TABLE_DIM_Y/2, 0.01], lineColorRGB = [0, 0, 0], lineWidth = 5)

    def _is_box_in_boundary(self):
        # Checks if the object is within the table boundary.
        x_pos = self.sim_data["x_box"]
        y_pos = self.sim_data["y_box"]
        contact_radius = self.shape_object.get_contact_radius()

        return (x_pos >= -self.TABLE_DIM_X/2 + contact_radius) and (x_pos <= self.TABLE_DIM_X/2 - contact_radius) and (y_pos >= -self.TABLE_DIM_Y/2 + contact_radius) and (y_pos <= self.TABLE_DIM_Y/2 - contact_radius)

    def _is_pusher_in_boundary(self):
        # Checks if the pusher is within the table boundary.
        x_pos = self.sim_data["x_pusher"]
        y_pos = self.sim_data["y_pusher"]

        return (x_pos >= -self.TABLE_DIM_X/2 + self.PUSHER_RADIUS) and (x_pos <= self.TABLE_DIM_X/2 - self.PUSHER_RADIUS) and (y_pos >= -self.TABLE_DIM_Y/2 + self.PUSHER_RADIUS) and (y_pos <= self.TABLE_DIM_Y/2 - self.PUSHER_RADIUS)

    def _generate_random_start(self):

        # Generate random starting position of the object (box_x, box_y)
        box_x = self.rng.random()*self.GEN_POS_X_WIDTH - self.GEN_POS_X_WIDTH/2
        box_y = self.rng.random()*self.GEN_POS_Y_WIDTH - self.GEN_POS_Y_WIDTH/2

        # Generate random starting orientation of the object (theta)
        theta = self.rng.random()*self.GEN_THETA_WIDTH - self.GEN_THETA_WIDTH / 2

        # Generate random starting position of the pusher (pusher_x, pusher_y).
        # Use shape-specific pusher placement logic
        pusher_x, pusher_y = self.shape_object.get_random_pusher_position(
            box_x, box_y, theta, self.PUSHER_RADIUS, self.SPACE_PUSHER_BOX, self.rng
        )

        return box_x, box_y, theta, pusher_x, pusher_y

    def _generate_random_target(self):
        target_x = self.rng.random()*self.GEN_POS_X_WIDTH - self.GEN_POS_X_WIDTH/2
        target_y = self.rng.random()*self.GEN_POS_Y_WIDTH - self.GEN_POS_Y_WIDTH/2
        target_theta = self.rng.random()*self.GEN_THETA_WIDTH - self.GEN_THETA_WIDTH / 2

        # Visualize target using the same shape as the object
        if self.graphics:
            target_visual_params = self.shape_object.get_visual_shape(color=[0, 1, 0, 0.5])
            target_visual = pb.createVisualShape(**target_visual_params)
            # Apply z offset for mesh files
            z_offset = self.shape_object.get_z_offset()
            z_position = self.shape_object.height/2 + z_offset
            self.target_vis = pb.createMultiBody(
                baseMass = 0,
                baseVisualShapeIndex = target_visual,
                basePosition = [target_x, target_y, z_position],
                baseOrientation = pb.getQuaternionFromEuler([0, 0, target_theta])
            )
            
            # Draw orientation arrow for the target (blue arrow)
            arrow_length = self.shape_object.get_contact_radius() + 0.02
            end_x = target_x + arrow_length * np.cos(target_theta)
            end_y = target_y + arrow_length * np.sin(target_theta)
            pb.addUserDebugLine(
                lineFromXYZ=[target_x, target_y, z_position + 0.01],
                lineToXYZ=[end_x, end_y, z_position + 0.01],
                lineColorRGB=[0, 0, 1],  # Blue for target orientation
            )

        return target_x, target_y, target_theta

    def _calculate_reward(self):

        # value 0 when as far as possible within the table, value 0.1 when at the target
        distance_reward = 0.1*(1 - self.sim_data["distance_to_target"] / (self.TABLE_DIM_X**2 + self.TABLE_DIM_Y**2)**(1/2))

        # value 0 when pointing away from the target, value 0.02 when pointing towards the target
        theta_reward = 0.02*(1 - self.sim_data["theta_distance_to_target"] / np.pi)

        # value 0 when at maximum velocity, value 0.004 when not moving
        velocity_reward = 0.004*(1 - 5*np.sqrt(2)*self.sim_data["velocity_pusher"] + 1e-6)

        # Give a reward in terms of the current distance, orientation, and pusher velocity
        reward = distance_reward + theta_reward + velocity_reward

        # Punish if the box or the pusher leave the table
        if (not self._is_box_in_boundary()) or (not self._is_pusher_in_boundary()):
            self._record_unsuccessful_episode()
            return -20, True
        
        # Reward if the agent is successful
        if (self.sim_data["velocity_box"] <= 1e-4) and (self.sim_data["distance_to_target"] <= self.DISTANCE_SUCCESS) and (self.sim_data["theta_distance_to_target"] <= self.THETA_DISTANCE_SUCCESS):
            self._record_successful_episode()
            return 50, True
        
        return reward, False

    def _get_observation(self):
        # Generate observation noise for the current step
        self.BOX_X_NOISE_STEP = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.BOX_Y_NOISE_STEP = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.BOX_THETA_NOISE_STEP = self.rng.normal(self.THETA_NOISE_MEAN, self.THETA_NOISE_STD)
        self.PUSHER_X_NOISE_STEP = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.PUSHER_Y_NOISE_STEP = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)

        # Update box pose stack
        self.box_pose_stack.append([self.sim_data["x_box"] + self.BOX_X_NOISE_EPISODE + self.BOX_X_NOISE_STEP, 
                                    self.sim_data["y_box"] + self.BOX_Y_NOISE_EPISODE + self.BOX_Y_NOISE_STEP,
                                    self.sim_data["theta_box"] + self.BOX_THETA_NOISE_EPISODE + self.BOX_THETA_NOISE_STEP])

        # Update pusher position stack
        self.pusher_pos_stack.append([self.sim_data["x_pusher"] + self.PUSHER_X_NOISE_EPISODE + self.PUSHER_X_NOISE_STEP,
                                      self.sim_data["y_pusher"] + self.PUSHER_Y_NOISE_EPISODE + self.PUSHER_Y_NOISE_STEP])

        # Extract the current observation from the box stack
        box_pose_observation = np.array(self.box_pose_stack[-self.STACK_SIZE:])

        # Extract the current observation from the pusher stack
        pusher_pos_observation = np.array(self.pusher_pos_stack[-self.STACK_SIZE:])

        # Normalize data
        target_x_box_norm = self.target_x_box / (self.TABLE_DIM_X/2)
        target_y_box_norm = self.target_y_box / (self.TABLE_DIM_Y/2)
        target_theta_norm = self.target_theta / np.pi
        box_pose_observation[:,0] = box_pose_observation[:,0] / (self.TABLE_DIM_X/2)
        box_pose_observation[:,1] = box_pose_observation[:,1] / (self.TABLE_DIM_Y/2)
        box_pose_observation[:,2] = box_pose_observation[:,2] / np.pi
        pusher_pos_observation[:,0] = pusher_pos_observation[:,0] / (self.TABLE_DIM_X/2)
        pusher_pos_observation[:,1] = pusher_pos_observation[:,1] / (self.TABLE_DIM_Y/2)

        # Target observation
        target_observation = np.array([target_x_box_norm, target_y_box_norm, target_theta_norm])

        # Dynamic observation
        dynamic_observation = np.concatenate((box_pose_observation.flatten(), pusher_pos_observation.flatten()))

        return np.concatenate((target_observation, dynamic_observation))

    def _adjust_action(self, action):
        # Action space is discrete in range [0, 1, ..., 10].
        # We want max velocity on each axis of 0.1.
        return action*0.02 - 0.1

    def render(self, mode='human'):
        pass
    
    def close (self):
        pb.disconnect()
