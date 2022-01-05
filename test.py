import argparse
import os
import time

import numpy as np
import scipy.io as scio
import torch
from tqdm import tqdm

import dataloaders_hsi_test
from ops.utils import MSIQA
from ops.utils import str2bool
from ops.utils_blocks import block_module

parser = argparse.ArgumentParser()
from skimage.restoration import estimate_sigma
from model_loader import init_model, load_model

# model
parser.add_argument("--noise_level", type=int, dest="noise_level", help="Should be an int in the range [0,255]",
                    default=0)
parser.add_argument("--bandwise", type=str2bool, default=1, help='bandwise noise')
parser.add_argument("--num_half_layer", type=int, dest="num_half_layer", help="Number of LISTA step unfolded",
                    default=6)
parser.add_argument("--channels", type=int, dest="channels", help="Should be an int in the range [0,255]", default=16)
parser.add_argument("--nl", type=str2bool, dest="nl", help="If Nonlocal", default=1)
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Size of image blocks to process", default=56)
parser.add_argument("--rescaling_init_val", type=float, default=1.0)
parser.add_argument("--nu_init", type=float, default=1, help='convex combination of correlation map init value')
parser.add_argument("--corr_update", type=int, default=3,
                    help='choose update method in [2,3] without or with patch averaging')
parser.add_argument("--multi_theta", type=str2bool, default=1,
                    help='wether to use a sequence of lambda [1] or a single vector during lista [0]')
parser.add_argument("--diag_rescale_gamma", type=str2bool, default=0, help='diag rescaling code correlation map')
parser.add_argument("--diag_rescale_patch", type=str2bool, default=1, help='diag rescaling patch correlation map')
parser.add_argument("--freq_corr_update", type=int, default=6, help='freq update correlation_map')
parser.add_argument("--mask_windows", type=int, default=1, help='binarym, quadratic mask [1,2]')
parser.add_argument("--center_windows", type=str2bool, default=1,
                    help='compute correlation with neighboors only within a block')
parser.add_argument("--multi_std", type=str2bool, default=0)
parser.add_argument("--gpus", '--list', action='append', type=int, help='GPU')
parser.add_argument("--rs_real", type=str2bool, default=0)
parser.add_argument("--blind", type=str2bool, default=0)

# training
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=1e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="ADAM Learning rate step for decay", default=80)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--backtrack_decay", type=float, help='decay when backtracking', default=0.8)
parser.add_argument("--eps", type=float, dest="eps", help="ADAM epsilon parameter", default=1e-3)
parser.add_argument("--validation_every", type=int, default=300,
                    help='validation frequency on training set (if using backtracking)')
parser.add_argument("--backtrack", type=str2bool, default=1, help='use backtrack to prevent model divergence')
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=300)
parser.add_argument("--train_batch", type=int, default=2, help='batch size during training')
parser.add_argument("--test_batch", type=int, default=3, help='batch size during eval')
parser.add_argument("--aug_scale", type=int, default=0)

# data
parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results' dir path", default='./trained_model')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.",
                    default='trained_model_25_bandwise/MTMF_patch_56Layer_12lr_0.00100000/ckpt')
parser.add_argument("--test_path", type=str, help="Path to the dir containing the testing datasets.", default="data/")
parser.add_argument("--gt_path", type=str, help="Path to the dir containing the ground truth datasets.", default="gt/")
parser.add_argument("--resume", type=str2bool, dest="resume", help='Resume training of the model', default=True)
parser.add_argument("--dummy", type=str2bool, dest="dummy", default=False)
parser.add_argument("--tqdm", type=str2bool, default=False)
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

# inference
parser.add_argument("--kernel_size", type=int, default=12,
                    help='stride of overlapping image blocks [4,8,16,24,48] kernel_//stride')

# parser.add_argument("--stride_test", type=int, default=12, help='stride of overlapping image blocks [4,8,16,24,48] kernel_//stride')
parser.add_argument("--stride_val", type=int, default=40,
                    help='stride of overlapping image blocks for validation [4,8,16,24,48] kernel_//stride')
parser.add_argument("--test_every", type=int, default=300, help='report performance on test set every X epochs')
parser.add_argument("--block_inference", type=str2bool, default=False,
                    help='if true process blocks of large image in paralel')
parser.add_argument("--pad_image", type=str2bool, default=0, help='padding strategy for inference')
parser.add_argument("--pad_block", type=str2bool, default=1, help='padding strategy for inference')
parser.add_argument("--pad_patch", type=str2bool, default=0, help='padding strategy for inference')
parser.add_argument("--no_pad", type=str2bool, default=False, help='padding strategy for inference')
parser.add_argument("--custom_pad", type=int, default=None, help='padding strategy for inference')
parser.add_argument("--save", type=str2bool, default=0, help='padding strategy for inference')

# variance reduction
# var reg
parser.add_argument("--nu_var", type=float, default=0.01)
parser.add_argument("--freq_var", type=int, default=3)
parser.add_argument("--var_reg", type=str2bool, default=False)

parser.add_argument("--verbose", type=str2bool, default=1)

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES']= '6,7'
if args.gpus is not None and len(args.gpus):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else os.cpu_count()
    gpus = args.gpus
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if device.type == 'cuda':
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
else:
    device = torch.device("cpu")
    device_name = 'cpu'
    capability = os.cpu_count()

test_path = [args.test_path]
gt_path = args.gt_path
print(f'test data : {test_path}')
print(f'gt data : {gt_path}')
train_path = val_path = []

noise_std = args.noise_level / 255

loaders = dataloaders_hsi_test.get_dataloaders(test_path, crop_size=args.patch_size,
                                               batch_size=args.train_batch, downscale=args.aug_scale, concat=1,
                                               verbose=True, grey=False)
model = init_model(in_channels=1, channels=args.channels,
                   num_half_layer=args.num_half_layer,
                    rs=args.rs_real)
if device.type == 'cuda':
    model = torch.nn.DataParallel(model.to(device=device), device_ids=gpus)
args.model_15 = 'trained_model/ckpt_15'
args.model_55 = 'trained_model/ckpt_55'
args.model_95 = 'trained_model/ckpt_95'

tic = time.time()
phase = 'test'

num_iters = 0
psnr_tot = []
ssim_tot = []
sam_tot = []
stride_test = args.patch_size // 2
loader = loaders['test']
for batch, fname in tqdm(loader, disable=not args.tqdm):
    batch = batch.to(device=device)
    fname = fname[0]
    print(fname)
    noisy_batch = batch
    if args.blind:
        sigma_est = np.array(estimate_sigma(noisy_batch.squeeze(0).permute([1, 2, 0]).detach().cpu(), multichannel=True,
                                            average_sigmas=False)).max() * 255
    else:
        sigma_est = args.noise_level
    if sigma_est > 15 and sigma_est <= 55:
        load_model(model_name=args.model_55, model=model,device_name=device_name)
    if sigma_est <= 15:
        load_model(model_name=args.model_15, model=model,device_name=device_name)
    if sigma_est > 55:
        load_model(model_name=args.model_95, model=model,device_name=device_name)
    if args.rs_real:
        load_model(model_name=args.model_95, model=model,device_name=device_name)
        args.block_inference=1
        args.patch_size=128
        stride_test=args.patch_size//2
    model.eval()  # Set model to evaluate mode
    if args.block_inference:
        if args.patch_size > noisy_batch.shape[-1] // 2 or args.patch_size > noisy_batch.shape[-2] // 2:
            stride_test = min(args.patch_size // 8, 8)
    with torch.set_grad_enabled(False):
        if args.block_inference:
            params = {
                'crop_out_blocks': 0,
                'ponderate_out_blocks': 1,
                'sum_blocks': 0,
                'pad_even': 1,  # otherwise pad with 0 for las
                'centered_pad': 0,  # corner pixel have only one estimate
                'pad_block': args.pad_block,  # pad so each pixel has S**2 estimate
                'pad_patch': args.pad_patch,  # pad so each pixel from the image has at least S**2 estimate from 1 block
                'no_pad': args.no_pad,
                'custom_pad': args.custom_pad,
                'avg': 1}
            block = block_module(args.patch_size, stride_test, args.kernel_size, params)
            batch_noisy_blocks = block._make_blocks(noisy_batch)
            patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=args.test_batch, drop_last=False)
            batch_out_blocks = torch.zeros_like(batch_noisy_blocks)
            for i, inp in enumerate(patch_loader):  # if it doesnt fit in memory
                id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
                batch_out_blocks[id_from:id_to] = model(inp)

            output = block._agregate_blocks(batch_out_blocks)
            # print(torch.isnan(output).sum())
        else:
            output = model(noisy_batch)
        gt = dataloaders_hsi_test.get_gt(gt_path, fname);
        gt = gt.to(device=device)
        if device_name == 'cpu':
            psnr_batch, ssim_batch, sam_batch = MSIQA(gt.detach().numpy(),
                                                      output.squeeze(0).detach().numpy())
            if args.save:
               scio.savemat(fname + 'Res.mat', {'output': output.squeeze(0).detach().numpy()})
        else:
            psnr_batch, ssim_batch, sam_batch = MSIQA(gt.detach().cpu().numpy(),
                                                      output.squeeze(0).detach().cpu().numpy())

            if args.save:
                scio.savemat(fname + 'Res.mat', {'output': output.squeeze(0).detach().cpu().numpy()})
    psnr_tot.append(psnr_batch)
    ssim_tot.append(ssim_batch)
    sam_tot.append(sam_batch)
    num_iters += 1
    tqdm.write(f'psnr avg {psnr_batch} ssim avg {ssim_batch} sam avg {sam_batch} ')
    if args.dummy:
        break
tac = time.time()
psnr_mean = np.mean(psnr_tot)
ssim_mean = np.mean(ssim_tot)
sam_mean = np.mean(sam_tot)
# scio.savemat(args.out_dir + 'GT.mat', {'psnr': psnr_tot, 'ssim': ssim_tot, 'sam': sam_tot})
# psnr_tot = psnr_tot.item()

tqdm.write(
    f'psnr: {psnr_mean:0.4f}  ssim: {ssim_mean:0.4f} sam: {sam_mean:0.4f}({(tac - tic) / num_iters:0.3f} s/iter)')
