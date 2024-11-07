import torch
import argparse
import pickle
import utils
import mesh_sampling
from typing import List
import os.path as osp
from torch.utils.data import DataLoader
from train_diffusion import run
from CLIP import FACE_encoder, CLIP, Transformer
from network import DiffusionPriorNetwork, DiffusionPrior
from train import DiffusionPriorTrainer
from decoder import Decoder
from psbody.mesh import Mesh
from writer import Writer

parser = argparse.ArgumentParser(description='Diffusion')

# training hyperparameters

parser.add_argument('--out_channels',
                    nargs='+',
                    default=[16, 16, 16, 32],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=128)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

parser.add_argument('--resume', type=str, default='checkpoint_100.pt')
parser.add_argument('--device_idx', type=int, default = 0)
args = parser.parse_args()
args.work_dir = osp.dirname(osp.realpath(__file__))
args.out_dir = osp.join(args.work_dir, 'out')
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
args.checkpoint_dir = osp.join(args.out_dir, 'checkpoint')
device = torch.device('cpu', args.device_idx)
writer = Writer(args)
print(args)


# print the JS visualization code to the notebook
shap.initjs()

epochs = 500

template_fp = osp.join('template.obj')

train_dataset = torch.load('train1 copy 3.pt')
test_dataset = torch.load('test1 copy 3.pt')
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

#cc generate/load transform matrices

transform_fp = osp.join('/share/home/jiaomingqi/test8/data/face', 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [32, 32, 32, 32]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh, ds_factors)
    tmp = {
        'vertices': V,
        'face': F,
        'adj': A,
        'down_transform': D,
        'up_transform': U
    }

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\'s'.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

image_encoder = FACE_encoder(args.in_channels, args.out_channels, args.latent_channels,
                            spiral_indices_list, down_transform_list,
                            up_transform_list).to(device)

text_encoder = Transformer().to(device)


clip = CLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
).to(device)
# load
# epoch = 0
# if args.resume:
""" if len(args.resume.split('/')) > 1:
        model_path = args.resume
    else:"""
model_path = osp.join(args.checkpoints_dir, args.resume)
checkpoint = torch.load(model_path, map_location='cuda:1')
if checkpoint.get('model_state_dict', None) is not None:
            checkpoint = checkpoint['model_state_dict']
clip.load_state_dict(checkpoint)
# epoch = checkpoint.get('epoch', -1) + 1
print('Load checkpoint {}'.format(model_path))

decoder = Decoder(args.in_channels, args.out_channels, args.latent_channels,
                            spiral_indices_list, down_transform_list,
                            up_transform_list).to(device)

model_path = osp.join(args.checkpoints_dir, args.resume)
checkpoint = torch.load(model_path, map_location='cuda:1')
if checkpoint.get('decoder_state_dict', None) is not None:
            checkpoint = checkpoint['decoder_state_dict']
decoder.load_state_dict(checkpoint)
# epoch = checkpoint.get('epoch', -1) + 1
print('Load checkpoint {}'.format(model_path))


# prior networks (with transformer)

prior_network = DiffusionPriorNetwork(
    dim = 128,
    depth = 2,
    dim_head = 64,
    heads = 4,
    ff_mult = 2,
    attn_dropout = 0.1,
    ff_dropout = 0.1,
    self_cond = False,
    num_timesteps = 1000,
    norm_in = False
).to(device)


diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    image_embed_dim = 128,
    image_channels = 1,
    timesteps = 1000,
    cond_drop_prob = 0,
    predict_x_start = True
).to(device)

diffusion_prior_trainer = DiffusionPriorTrainer(
    diffusion_prior, 
    lr = 0.31e-4,
    wd = 7.52e-2,
    max_grad_norm = 0.5,
    amp=False,
    group_wd_params=True,
    use_ema = True,
    ema_beta = 0.98,
    ema_update_after_step = 1000,
    ema_update_every = 100,
    warmup_steps = 100,
    device = device
).to(device)

run(decoder, clip, diffusion_prior_trainer, train_loader, test_loader, epochs, writer, device)
