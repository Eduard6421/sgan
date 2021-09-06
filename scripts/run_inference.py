import argparse
import os
import torch
import numpy as np

from attrdict import AttrDict

from sgan.data.loader import data_loader, inference_data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

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


def infer(args, loader, generator, num_samples):
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

    print(result)
    return result


generator = None


def main(args, data):

    global generator

    model_path='./models/sgan-models'

    if generator is None:
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
        print(path)
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)

    _args = AttrDict(checkpoint['args'])
    _, loader = inference_data_loader(_args, data)
    infer(_args, loader, generator, args.num_samples)


if __name__ == '__main__':
    args = parser.parse_args()
    data = np.array([
        [10.0,	1.0,	14.9352355744,	5.30707796623],
        [10.0,	2.0,	15.1821111479,	5.82496973161],
        [20.0,	1.0,	14.4947320999,	5.3292733276],
        [20.0,	2.0,	14.8287402289,	5.81637797882],
        [30.0,	1.0,	14.0542286255,	5.35146868897],
        [30.0,	2.0,	14.4755797749,	5.80802488583],
        [40.0,	1.0,	13.613725151	, 5.37366405035],
        [40.0,	2.0,	14.1222088559,	5.79943313304],
        [50.0,	1.0,	13.1725902812,	5.39371147352],
        [50.0,	2.0,	13.6884402649,	5.87174705235],
        [60.0,	1.0,	12.7264042488,	5.39371147352],
        [60.0,	2.0,	13.2548821391,	5.94406097166],
        [70.0,	1.0,	12.2802182164,	5.39371147352],
        [70.0,	2.0,	12.8211135482,	6.01637489098],
        [80.0,	1.0,	11.834032184	, 5.39371147352],
        [80.0,	2.0,	12.3583007721,	6.06291355192],
        [80.0,	3.0,	4.03819405839,	0.128876291845],
        [90.0,	1.0,	11.3878461516,	5.39371147352],
        [90.0,	2.0,	11.8826496243,	6.09823520228],
        [90.0,	3.0,	4.43092195295,	0.940319610871],
        [100.0,	1.0,	10.9561822118,	5.39776869012],
        [100.0,	2.0,	11.4072089417,	6.13355685264],
        [100.0,	3.0,	4.84616961424,	1.60737375107],
        [110.0,	1.0,	10.6541647795,	5.43929549527],
        [110.0,	2.0,	10.933662445,   6.12353314105],
        [110.0,	3.0,	5.27636029831,	2.17800933218],
        [120.0,	1.0,	10.3521473472,	5.48082230042],
        [120.0,	2.0,	10.4601159484,	6.11374808926],
        [120.0,	3.0,	5.73664749306,	2.73098408812],
        [130.0,	1.0,	10.0008810792,	5.4977671462],
        [130.0,	2.0,	9.99477759101,	6.11494138826],
        [130.0,	3.0,	6.26786142976,	3.24243203891],
        [140.0,	1.0,	9.63740783486,	5.50874549699],
        [140.0,	2.0,	9.53491132647,	6.12377180085],
        [140.0,	3.0,	6.79297187827,	3.71521710214],
        [150.0,	1.0,	9.19353691866,	5.4999150844],
        [150.0,	2.0,	9.07504506194,	6.13260221344],
        [150.0,	3.0,	7.29408930429,	4.03335061516],
        [160.0,	1.0,	8.69599739951,	5.47843570242],
        [160.0,	2.0,	8.5821357752	, 6.13904602803],
        [160.0,	3.0,	7.81267533441,	4.34360835478],
        [170.0,	1.0,	8.19866834546,	5.45671766065],
        [170.0,	2.0,	8.08564858159,	6.14525118282],
        [170.0,	3.0,	8.48932066186,	4.5822681545],
        [180.0,	1.0,	7.70070789609,	5.42712384548],
        [180.0,	2.0,	7.58916138799,	6.15121767781],
        [180.0,	3.0,	9.15333808272,	4.83548620199],
        [190.0,	1.0,	7.20148465606,	5.37939188554],
        [190.0,	2.0,	7.09267419438,	6.15742283261],
        [190.0,	3.0,	9.81230434096,	5.09490940428],
        [200.0,	1.0,	6.70247188113,	5.3316599256],
        [200.0,	2.0,	6.60649979115,	6.12329448125],
        [200.0,	3.0,	10.4660089715,	5.30970322402],
    ])
    main(args, data)
