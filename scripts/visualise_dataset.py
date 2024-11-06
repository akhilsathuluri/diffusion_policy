# %%
from pathlib import Path
from typing import Iterator

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm
import logging

import os
import sys

sys.path.append(Path("../").resolve().__str__())
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset


# %%


# class EpisodeSampler(torch.utils.data.Sampler):
#     def __init__(self, dataset, episode_index: int):
#         # extract the info for the episode index and their corresponding frame names
#         self.frame_ids = np.where(
#             dataset.replay_buffer.get_episode_idxs() == episode_index
#         )[0]

#     def __iter__(self) -> Iterator:
#         return iter(self.frame_ids)

#     def __len__(self) -> int:
#         return len(self.frame_ids)


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


def visualize_data(
    # dataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    # web_port: int = 9090,
    # ws_port: int = 9087,
    # save: bool = False,
    # root: None = None,
    # output_dir: None = None,
) -> None:
    logging.info("Loading dataset")
    # TODO: Need to extract the dataset
    zarr_path = Path("../data/pusht/pusht_cchi_v7_replay.zarr").resolve().__str__()
    dataset = PushTImageDataset(zarr_path, horizon=16)
    logging.info("Loading dataloader")
    # episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        # sampler=episode_sampler,
    )

    # Troubleshoot by inspecting one loaded batch
    batch = next(iter(dataloader))

    logging.info("Starting rerun")
    rr.init("visualising pushT dataset", spawn="local")

    logging.info("logging to rerun")
    for i in range(len(batch["action"]) - 1):
        rr.log("frame", rr.Image(to_hwc_uint8_numpy(batch["obs"]["image"][0][i])))
        if "action" in batch:
            rr.log(f"action", batch["action"][0][i])
    # log all data corresponding to the first episode


visualize_data(episode_index=0)

# %%
