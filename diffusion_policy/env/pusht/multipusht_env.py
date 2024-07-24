import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class MultiPushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=True,
        render_size=96,
        reset_to_state=None,
    ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["video.frames_per_second"]
        # legcay set_state for data compatibility
        self.legacy = legacy

        # new additions for multipusht env
        self.num_blocks = 2
        self.num_agents = 2

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0] * self.num_agents + [0, 0, 0] * self.num_blocks,
                dtype=np.float64,
            ),
            high=np.array(
                [ws, ws] * self.num_agents + [ws, ws, 2 * np.pi] * self.num_blocks,
                dtype=np.float64,
            ),
            shape=(2 * self.num_agents + 3 * self.num_blocks,),
            dtype=np.float64,
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0, 0] * self.num_agents, dtype=np.float64),
            high=np.array([ws, ws] * self.num_agents, dtype=np.float64),
            shape=(2 * self.num_agents,),
            dtype=np.float64,
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array(
                sum(
                    [
                        [rs.randint(50, 450), rs.randint(50, 450)]
                        for _ in range(self.num_agents)
                    ]
                    + [
                        [
                            rs.randint(100, 400),
                            rs.randint(100, 400),
                            rs.randn() * 2 * np.pi - np.pi,
                        ]
                        for _ in range(self.num_blocks)
                    ],
                    [],
                )
            )
        self._set_state(state)

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                for ii in range(self.num_agents):
                    acceleration = self.k_p * (
                        action[2 * ii : 2 * ii + 2] - self.agents[ii].position
                    ) + self.k_v * (Vec2d(0, 0) - self.agents[ii].velocity)
                    self.agents[ii].velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_bodies = self._get_goal_pose_bodies(self.goal_poses)
        goal_geoms = [
            pymunk_to_shapely(goal_bodies[ii], self.blocks[ii].shapes)
            for ii in range(self.num_blocks)
        ]
        block_geoms = [
            pymunk_to_shapely(self.blocks[ii], self.blocks[ii].shapes)
            for ii in range(self.num_blocks)
        ]

        reward = 0
        done = True
        for ii in range(self.num_blocks):
            block_geom = block_geoms[ii]
            goal_geom = goal_geoms[ii]
            intersection_area = goal_geom.intersection(block_geom).area
            goal_area = goal_geom.area
            coverage = intersection_area / goal_area
            reward += np.clip(coverage / self.success_threshold, 0, 1)
            done = done and coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), self.screen
            )
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return TeleopAgent(act)

    def _get_obs(self):
        blocks_pose = np.copy(self.blocks.position)
        for ii in range(self.num_blocks):
            np.insert(blocks_pose, ii * 3, self.blocks.angle[ii] % (2 * np.pi))
        obs = np.array(tuple(self.agents.position) + tuple(blocks_pose))
        return obs

    def _get_goal_pose_body(self, poses):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        bodies = []
        for ii in range(len(poses)):
            body = pymunk.Body(mass, inertia)
            # preserving the legacy assignment order for compatibility
            # the order here dosn't matter somehow, maybe because CoM is aligned with body origin
            body.position = poses[ii][:2].tolist()
            body.angle = poses[ii][2]
            bodies.append(body)
        return bodies

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        agents_position = np.array([agent.position for agent in self.agents]).flatten()
        agents_velocity = np.array([agent.velocity for agent in self.agents]).flatten()
        blocks_pose = np.array(
            [list(block.position) + [block.angle] for block in self.blocks]
        ).flatten()

        info = {
            "pos_agents": agents_position,
            "vel_agents": agents_velocity,
            "block_poses": blocks_pose,
            "goal_poses": self.goal_poses,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_bodies = self._get_goal_pose_bodies(self.goal_poses)
        goal_points = []
        for goal_body in goal_bodies:
            for shape in self.blocks[0].shapes:
                goal_points = [
                    pymunk.pygame_util.to_pygame(
                        goal_body.local_to_world(v), draw_options.surface
                    )
                    for v in shape.get_vertices()
                ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                actions = np.array(self.latest_action)
                for ii in range(self.num_agents):
                    action = actions[2 * ii : 2 * ii + 2]
                    coord = (action / 512 * 96).astype(np.int32)
                    marker_size = int(8 / 96 * self.render_size)
                    thickness = int(1 / 96 * self.render_size)
                    cv2.drawMarker(
                        img,
                        coord,
                        color=(255, 0, 0),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=marker_size,
                        thickness=thickness,
                    )
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agents = state[: 2 * self.num_agents]
        pose_blocks = state[2 * self.num_agents :]
        for ii in range(self.num_agents):
            self.agents[ii].position = pos_agents[2 * ii : 2 * ii + 2]

        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatiblity with legacy data
            for ii in range(self.num_blocks):
                self.blocks[ii].position = pose_blocks[3 * ii : 3 * ii + 2]
                self.blocks[ii].angle = pose_blocks[ii * 3]
        else:
            for ii in range(self.num_blocks):
                self.blocks[ii].angle = pose_blocks[ii * 3]
                self.blocks[ii].position = pose_blocks[3 * ii : 3 * ii + 2]

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    # TODO: Left unchanged, not used?
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], rotation=self.goal_pose[2]
        )
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2], rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(matrix=tf_img_obj.params @ tf_obj_new.params)
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0])
            + list(tf_img_new.translation)
            + [tf_img_new.rotation]
        )
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agents = [self.add_circle((256, 400), 15) for _ in range(self.num_agents)]
        self.blocks = [self.add_tee((256, 300), 0) for _ in range(self.num_blocks)]
        self.goal_color = pygame.Color("LightGreen")
        self.goal_poses = np.random.uniform(
            np.array([256, 256, np.pi / 4]) * 0.95,
            np.array([256, 256, np.pi / 4]) * 1.05,
            (self.num_blocks, 3),
        )

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95  # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color(
            "LightGray"
        )  # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color("LightSlateGray")
        self.space.add(body, shape)
        return body

    def add_tee(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body