import argparse
import os
import torch
import numpy as np

from attrdict import AttrDict

from sgan.sgan.data.loader import data_loader, inference_data_loader
from sgan.sgan.models import TrajectoryGenerator
from sgan.sgan.losses import displacement_error, final_displacement_error
from sgan.sgan.utils import relative_to_abs, get_dset_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def infer(args, loader, generator, num_samples=20):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():

        result = []

        for batch in loader:

            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )

            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]
            )

            result.append(pred_traj_fake[0])

    return result


checkpoint = None
generator = None


def main(data):

    global generator,checkpoint

    model_path='./sgan/models/sgan-models'

    if generator is None:
        print('is not loaded')
        paths = None
        if os.path.isdir(model_path):
            filenames = os.listdir(model_path)
            filenames.sort()
            paths = [
                os.path.join(model_path, file_) for file_ in filenames
            ]
        else:
            paths = [model_path]
        path = paths[0]
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)

    _args = AttrDict(checkpoint['args'])
    _, loader = inference_data_loader(_args, data)
    return infer(_args, loader, generator, 20)


def run_script(data):
    return main(data)