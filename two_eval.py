"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import logging

@click.command()
@click.option('-c', '--checkpoint', required=False)
@click.option('-o', '--output_dir', required=False)
@click.option('-na', '--num_agents', default=1, required=False)
def main(checkpoint, output_dir, num_agents):
    device = 'cuda:1'
    # trained on given demos
    # checkpoint = 'data/outputs/2024.11.05/21.22.21_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt'
    # trained on my demos
    # checkpoint='data/outputs/2024.11.07/03.21.31_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt'
    # lowDim checkpoint given by the paper
    checkpoint = 'data/epoch=0550-test_mean_score=0.969.ckpt'
    output_dir = 'data/pusht_eval_output_lowDim_two_active_agents_specific'
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # for image obs
    # modified_env_runner = 'pusht_image_runner_two_agents.PushTImageRunnerTwoAgents'
    # cfg['task']['env_runner']['_target_'] = cfg['task']['env_runner']['_target_'].replace('pusht_image_runner.PushTImageRunner', modified_env_runner)
    
    # for lowDim obs
    modified_env_runner = 'pusht_keypoints_runner_two_agents.PushTKeypointsRunnerTwoAgents'
    cfg['task']['env_runner']['_target_'] = cfg['task']['env_runner']['_target_'].replace('pusht_keypoints_runner.PushTKeypointsRunner', modified_env_runner)

    # for a specific reset state
    # fixed_reset_state=np.array([153, 256, 260, 359, 256, 256, np.pi/4+delta])
    logging.critical('Using fixed reset state!!!')
    cfg['task']['env_runner']['n_test'] = 10
    # cfg['task']['env_runner']['n_test'] = 10

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
