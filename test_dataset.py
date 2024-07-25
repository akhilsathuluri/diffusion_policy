# %%
from diffusion_policy.env.pusht.multipusht_image_env import MultiPushTImageEnv

env = MultiPushTImageEnv()

env.seed(100)
obs, info = env.reset()

action = env.action_space.sample()

obs, reward, terminated, info = env.step(action)

print(info)

# %%
