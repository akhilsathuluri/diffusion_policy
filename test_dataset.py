# %%
from diffusion_policy.env.pusht.multipusht_image_env import MultiPushTImageEnv

env = MultiPushTImageEnv()

env.seed(100)
obs, info = env.reset()

action = env.action_space.sample()

obs, reward, terminated, info = env.step(action)

print(info)

# %%
from diffusion_policy.dataset.multipusht_image_dataset import *

zarr_path = "./data/multipusht/multipusht_sat_v2.zarr"
dataset1 = MultiPushTImageDataset(zarr_path, horizon=16, num_agents=2)
bla1 = dataset1.replay_buffer["action"]
normalizer1 = dataset1.get_normalizer()
nactions1 = normalizer1["action"].normalize(dataset1.replay_buffer["action"])
diff = np.diff(nactions1, axis=0)
dists = np.linalg.norm(np.diff(nactions1, axis=0), axis=-1)

# %%
from diffusion_policy.dataset.pusht_image_dataset import *

zarr_path = "./data/pusht/pusht_cchi_v7_replay.zarr"
dataset2 = PushTImageDataset(zarr_path, horizon=16)

# bla = dataset.replay_buffer["state"][..., :2]
bla2 = dataset2.replay_buffer["action"]

normalizer2 = dataset2.get_normalizer()
nactions2 = normalizer2["action"].normalize(dataset2.replay_buffer["action"])

# %%

from diffusion_policy.env.pusht.multipusht_image_env import MultiPushTImageEnv
