# %%
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
from tqdm import tqdm

from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

# %%


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: PushTImageDataset, episode_index: int):
        ep_idxs = np.where(dataset.replay_buffer.get_episode_idxs() == episode_index)[0]
        from_idx = ep_idxs[0]
        to_idx = ep_idxs[-1]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert (
        c < h and c < w
    ), f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (
        (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    )
    return hwc_uint8_numpy


# %%
# get the length of the largest episode
dataset = PushTImageDataset(
    # Path("./data/pusht/pusht_cchi_v7_replay.zarr").resolve().__str__(),  # 206 episodes
    # Path("./data/pusht/pusht_sat_v0_replay.zarr").resolve().__str__(), # 223 episodes
    Path("./data/pusht/pusht_sat_v0_one_distractor_replay.zarr")
    .resolve()
    .__str__(),  # 221 episodes
    horizon=1,
)


# %%
episode_index = 0
episode_sampler = EpisodeSampler(dataset, episode_index)
dataloader = torch.utils.data.DataLoader(
    dataset, num_workers=8, batch_size=32, sampler=episode_sampler
)
# %% Troubleshooting

bla = next(iter(dataloader))
len(bla["action"])
batch = bla
ii = 0

# %%

rr.init(f"diffusion_policy/pusht/episode_{episode_index}", spawn="local")
frame_index = 0
for batch in tqdm(dataloader, total=len(dataloader)):
    for ii in range(len(batch["action"])):
        # rr.set_time_sequence("frame_index", ii)
        rr.log(
            "image",
            rr.Image(
                to_hwc_uint8_numpy(batch["obs"]["image"][ii][0].to(torch.float32))
            ),
        )
        for dim_idx, val in enumerate(batch["action"][ii][0]):
            rr.log(f"action/{dim_idx}", rr.Scalar(val.item()))
        for dim_idx, val in enumerate(batch["obs"]["agent_pos"][ii][0]):
            rr.log(f"agent_pos/{dim_idx}", rr.Scalar(val.item()))
        # frame_index += 1

# %%
