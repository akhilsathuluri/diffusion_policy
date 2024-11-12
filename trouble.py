# %% FOR IMAGE ENVIRONMENT
from diffusion_policy.env.pusht.pusht_image_env_two_agents import PushTImageEnvTwoAgents
from diffusion_policy.env_runner.pusht_image_runner_two_agents import PushTImageRunnerTwoAgents
# from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
import numpy as np

# %%
aa=PushTImageEnvTwoAgents(legacy=False, render_size=96)
aa=PushTImageRunnerTwoAgents(output_dir='data/pusht_eval_temp')
# aa=PushTImageRunner(output_dir='data/pusht_eval_temp')
# %%
env=aa.env
# %%
obs = env.reset()
# %%
obs['image'].shape
# %%
obs['agent_pos'].shape
# %%
obs['agent_pos'] = obs['agent_pos'][:,:,:2]
# %%
act = np.ones((56, 8, 2))
zeros = np.zeros((56, 8, 2))

# %%
np.concatenate([act, zeros], axis=2).shape

# -------------------------------------------------------------------

# %% For keypoints environment

import numpy as np
import matplotlib.pyplot as plt
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
env = PushTEnv()
kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
env.reset()
obj_map = {"block": env.block, "agent": env.agent}
obs = env.render(mode="rgb_array")
img = obs.astype(np.uint8)
kp_manager.draw_keypoints_pose(img=img, pose_map=obj_map, is_obj=True)
plt.imshow(img)

kp_kwargs = kp_manager.kwargs
local_keypoint_map = kp_kwargs["local_keypoint_map"]
Dblockkps = np.prod(local_keypoint_map["block"].shape)
Dagentkps = np.prod(local_keypoint_map["agent"].shape)

color_map = kp_kwargs["color_map"]
render_size=96,
keypoint_visible_rate=1.0,
agent_keypoints=False,
draw_keypoints=False,
kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map, color_map=color_map
        )
obj_map = {"block": env.block}
kp_map = kp_manager.get_keypoints_global(pose_map=obj_map, is_obj=True)
kps = np.concatenate(list(kp_map.values()), axis=0)
n_kps = kps.shape[0]


import numpy as np
import matplotlib.pyplot as plt
from diffusion_policy.env.pusht.pymunk_keypoint_manager_two_agents import PymunkKeypointManagerTwoAgents
from diffusion_policy.env.pusht.pusht_env_two_agents import PushTEnvTwoAgents
delta = np.pi/180 * 15 # 5 deg off
reset_state = np.array([153, 256, 260, 359, 256, 256, np.pi/4+delta])
env = PushTEnvTwoAgents(reset_to_state=reset_state)
kp_manager = PymunkKeypointManagerTwoAgents.create_from_pusht_env(env)
kp_kwargs = kp_manager.kwargs
local_keypoint_map = kp_kwargs["local_keypoint_map"]
Dblockkps = np.prod(local_keypoint_map["block"].shape)
Dagentkps = np.prod(local_keypoint_map["agent1"].shape+local_keypoint_map["agent2"].shape)
agent_keypoints = False
Dagentpos = 4
Do = Dblockkps
if agent_keypoints:
    # blockkp + agnet_pos
    Do += Dagentkps
else:
    # blockkp + agnet_kp
    Do += Dagentpos
Dobs = Do * 2
env.reset()


obj_map = {"block": env.block}
kp_map = kp_manager.get_keypoints_global(pose_map=obj_map, is_obj=True)
kps = np.concatenate(list(kp_map.values()), axis=0)
n_kps = kps.shape[0]
np_random = np.random.default_rng(5649684)
visible_kps = np_random.random(size=(n_kps,)) < 1.0
kps_mask = np.repeat(visible_kps[:, None], 2, axis=1)
obs_mask = kps_mask.flatten()

obs = env.render(mode="rgb_array")
img = obs.astype(np.uint8)
kp_manager.draw_keypoints_pose(img=img, pose_map=obj_map, is_obj=True)
plt.imshow(img)

# Now lets test the keypoints_env
from diffusion_policy.env.pusht.pusht_keypoints_env_two_agents import PushTKeypointsEnvTwoAgents
aa=PushTKeypointsEnvTwoAgents()
kp_kwargs = aa.genenerate_keypoint_manager_params()
aa.observation_space.low.shape
aa.kp_manager
# bla = aa._get_obs()

kp_manager = PymunkKeypointManagerTwoAgents.create_from_pusht_env(env)
obj_map = {"block": env.block, "agent1": env.agent1, "agent2": env.agent2}
kp_map = kp_manager.get_keypoints_global(pose_map=obj_map, is_obj=True)
kps = np.concatenate(list(kp_map.values()), axis=0)
n_kps = kps.shape[0]
np_random = np.random.default_rng(5649684)
visible_kps = np_random.random(size=(n_kps,)) < 1.0
kps_mask = np.repeat(visible_kps[:, None], 2, axis=1)
vis_kps = kps.copy()
vis_kps[~visible_kps] = 0
draw_kp_map = {"block": vis_kps[: len(kp_map["block"])]}
draw_kp_map["agent"] = vis_kps[len(kp_map["block"]) :]
obs = kps.flatten()
obs_mask = kps_mask.flatten()
agent_pos = np.array(tuple(env.agent1.position)+tuple(env.agent2.position))
obs = np.concatenate([obs, agent_pos])
obs_mask = np.concatenate([obs_mask, np.ones((2,), dtype=bool)])

# %%
delta = np.pi/180 * 30 # 5 deg off
aa = PushTKeypointsEnvTwoAgents(legacy=True, render_size=96, agent_keypoints=False, draw_keypoints=False, reset_to_state=np.array([153, 256, 310, 256, 240, 240, np.pi/4+delta]))
obs = aa.reset()
bla = aa.render(mode="rgb_array")
img = bla.astype(np.uint8)
plt.imshow(img)
# %%


if not aa.agent_keypoints:
    agent_pos = np.array(tuple(aa.agent1.position)+tuple(aa.agent2.position))
    obs = np.concatenate([obs, agent_pos])
    obs_mask = np.concatenate([obs_mask, np.ones((2,), dtype=bool)])

obs = np.concatenate([obs, obs_mask.astype(obs.dtype)], axis=0)

# %%
import torch
import dill
import hydra
# checkpoint='data/outputs/2024.11.07/03.21.31_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt'
checkpoint = 'data/epoch=0550-test_mean_score=0.969.ckpt'
# %%
payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
# %%
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
# %%
cfg['task']['env_runner']['_target_']
# %%
from diffusion_policy.workspace.base_workspace import BaseWorkspace
workspace = cls(cfg, output_dir='data/temp')
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=None, include_keys=None)

# %%
length = 4
scale=30

vertices1 = [(-length*scale/2, scale),
            ( length*scale/2, scale),
            ( length*scale/2, 0),
            (-length*scale/2, 0)]
