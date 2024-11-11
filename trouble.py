# %% 
from diffusion_policy.env.pusht.pusht_image_env_two_agents import PushTImageEnvTwoAgents
from diffusion_policy.env_runner.pusht_image_runner_two_agents import PushTImageRunnerTwoAgents
# from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner
import numpy as np

# %%
# aa=PushTImageEnvTwoAgents(legacy=False, render_size=96)
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
# %%
