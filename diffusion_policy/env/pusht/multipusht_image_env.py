from gym import spaces
from diffusion_policy.env.pusht.multipusht_env import MultiPushTEnv
import numpy as np
import cv2


class MultiPushTImageEnv(MultiPushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, legacy=False, block_cog=None, damping=None, render_size=96):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False,
        )
        ws = self.window_size
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=1, shape=(3, render_size, render_size), dtype=np.float32
                ),
                "agent_pos": spaces.Box(
                    low=0, high=ws, shape=(self.num_agents,), dtype=np.float32
                ),
            }
        )
        self.render_cache = None

    def _get_obs(self):
        img = super()._render_frame(mode="rgb_array")

        agents_pos = np.array([agent.position for agent in self.agents]).flatten()
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {"image": img_obs, "agents_pos": agents_pos}

        # draw action
        if self.latest_action is not None:
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
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == "rgb_array"

        if self.render_cache is None:
            self._get_obs()

        return self.render_cache
