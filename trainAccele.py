"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
# os.environ['NCCL_P2P_DISABLE'] = "1"
# os.environ['NCCL_IB_DISABLE'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
# from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT
from diffusion import create_diffusion

from dataset import EITdataset
import scipy.io as sio
import random
import matplotlib.pyplot as plt
from accelerate import Accelerator
from ema_pytorch import EMA


# from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def init_seed(seed=2019, reproducibility=True) -> None:
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    accelerator = Accelerator(mixed_precision='fp16')
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    seed = args.global_seed
    init_seed(seed)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(device)
    device = accelerator.device
    # print(device)
    # exit()
    gpus = torch.cuda.device_count()
    # Setup an experiment folder:
    if accelerator.is_local_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = 'deit'  # args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        # logger = create_logger(checkpoint_dir)
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{checkpoint_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    model = DiT()

    #####################
    '''
    model_string_name = 'deit'  # args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    state_dict = torch.load(checkpoint_dir + '/best.pt', map_location='cpu')
    # #
    model.load_state_dict(state_dict["model"])'''

    # while 1 :pass
    model = model.to(device)
    #####################

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    # scheduler  = torch.optim.lr_scheduler.MultiStepLR(
    #     opt , milestones=[10000, 20000, 30000,40000], gamma=0.5
    # )
    gpuname = torch.cuda.get_device_name(0)
    modelname = 'DEIT'
    if '4090' in gpuname:
        datapath = '../../data'
    elif '100' in gpuname:
        datapath = '/dpc/kuin0055/eitssk/data/dataDiscrete'
    elif '3070' in gpuname:
        datapath = '../../data/eitdata/dataDiscrete'

    path = datapath + '/train/'
    dataset = EITdataset(path, modelname)

    path = datapath + '/valid/'
    dataVal = EITdataset(path, modelname)

    args.epochs = int(np.ceil(50000 / (len(dataset) / args.global_batch_size / gpus)))
    batch_size = args.global_batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,

        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    loaderVal = DataLoader(
        dataVal,
        batch_size=batch_size * 4,
        shuffle=False,

        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if accelerator.is_main_process:
        ema = EMA(model, beta=0.995, update_every=10)
        ema.to(device)
        ema.ema_model.eval()

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    load_weight = False  # True#
    if load_weight == True:
        model_string_name = 'deit'  # args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_dir + '/best.pt', map_location='cpu')
        # #
        # model.load_state_dict(state_dict["model"])

        # checkpoint = torch.load(weight, map_location='cpu')
        current_epoch = checkpoint["epoch"] + 1
        # model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        # epochs = 0
        accelerator.print('load weight')

        model.load_state_dict(checkpoint['model'])
        if accelerator.is_main_process:
            ema = EMA(model, beta=0.995, update_every=10)
            ema.to(device)
            ema.ema_model.load_state_dict(checkpoint['model'])
            ema.ema_model.eval()
        model, opt, loader, loaderVal = accelerator.prepare(model, opt, loader, loaderVal)
    else:
        current_epoch = 0
        model, opt, loader, loaderVal = accelerator.prepare(model, opt, loader, loaderVal)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    best_loss = 1000000
    Loss_tr = []
    Loss_val = []
    model.train()

    for epoch in range(current_epoch, current_epoch + args.epochs):

        # logger.info(f"Beginning epoch {epoch}...")
        for y, y_st, x in loader:

            x = x.to(device)  # image
            y = y.to(device)  # voltage
            y_st = y_st.to(device)
            # batch_mask = x != 0
            # x = torch.cat([x, batch_mask], dim=1)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, y_st=y_st)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"]  # *args.global_batch_size

            opt.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            # if accelerator.sync_gradients:
            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), 1)
            opt.step()

            if accelerator.is_local_main_process:
                ema.update()
            # scheduler.step()
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            #if train_steps>11:break
            if train_steps % args.log_every == 0:
                logger.info('*' * 40)
                # Measure training speed:

                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)

                # avg_loss = avg_loss.item()
                # if accelerator.is_local_main_process:
                avg_loss = accelerator.gather(avg_loss)
                avg_loss = avg_loss.mean().item()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:

                running_loss = 0
                log_steps = 0
                start_time = time()
                Loss_tr.append(avg_loss)

                # Save DiT checkpoint:
            # exit()
            if train_steps % args.ckpt_every == 0:

                model.eval()
                val_loss_v = 0
                log_steps_v = 0
                with torch.no_grad():
                    for y, y_st, x in loaderVal:
                        x = x.to(device)
                        y = y.to(device)
                        y_st = y_st.to(device)
                        # batch_mask = x != 0
                        # x = torch.cat([x, batch_mask], dim=1)
                        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                        model_kwargs = dict(y=y, y_st=y_st)
                        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                        loss = loss_dict["loss"].mean()

                        val_loss_v += loss.item()
                        log_steps_v += 1

                    val_loss_v = torch.tensor(val_loss_v / log_steps_v, device=device)

                    # val_loss_v = val_loss_v.item()

                    val_loss_v = accelerator.gather(val_loss_v)
                    val_loss_v = val_loss_v.mean().item()
                    logger.info(
                        f"(step={train_steps:07d}) Valid Loss: {val_loss_v:.4f}")
                    Loss_val.append(val_loss_v)
                    if val_loss_v < best_loss:
                        best_loss = val_loss_v
                        if accelerator.is_local_main_process:
                            checkpoint = {
                                "model": ema.ema_model.state_dict(),
                                # "model":  model.state_dict(),
                                # "optimizer": opt.state_dict(),
                                "epoch": epoch
                            }
                            checkpoint_path = f"{checkpoint_dir}/best.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                model.train()

    if accelerator.is_local_main_process:
        sio.savemat(checkpoint_dir + '/loss1.mat',
                    {'loss_stage1Tr': np.stack(Loss_tr),
                     'loss_stage1Val': np.stack(Loss_val)})


def test(args):
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    gpus = torch.cuda.device_count()
    seed = args.global_seed
    init_seed(seed)

    model_string_name = 'deit'  # args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    # os.makedirs(checkpoint_dir, exist_ok=True)
    # logger = create_logger(experiment_dir)
    # logger.info(f"Experiment directory created at {experiment_dir}")

    model = DiT().to(device)
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    # model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")

    # model_name='DEIT'
    # state_dict = torch.load(checkpoint_dir + '/best.pt')

    state_dict = torch.load(checkpoint_dir + '/best.pt', map_location='cpu')
    # print(model.y_embedder)
    # model.load_state_dict(state_dict["model"])
    # print(state_dict["epoch"])
    model.load_state_dict(state_dict['model'])

    gpuname = torch.cuda.get_device_name(0)
    # print(gpuname)
    modelname = 'DEIT'
    if '4090' in gpuname:
        datapath = '../../data'
    elif '100' in gpuname:
        datapath = '/dpc/kuin0055/eitssk/data/dataDiscrete'
    elif '3070' in gpuname:
        datapath = '../../data/eitdata/dataDiscrete'
    path = datapath + '/test/'

    dataTe = EITdataset(path, modelname, dataset='data')  #

    loaderTe = DataLoader(
        dataTe,
        batch_size=args.global_batch_size * 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    model, loaderTe = accelerator.prepare(model, loaderTe)

    accelerator.print('sampling steps', args.samplingsteps)
    model.eval()
    with torch.no_grad():
        pred = []
        gt1 = []
        RMSE = []
        for i, (y, y_st, x) in enumerate(loaderTe):
            x = x.to(device)
            y = y.to(device)
            y_st = y_st.to(device)
            # print(x.shape)
            # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, y_st=y_st)

            # batch_mask = x != 0
            # x = torch.cat([x, batch_mask], dim=1)
            shape = x.shape
            # shape = [x.shape[0], 4, 16, 16]
            if gpus == 1:
                out = diffusion.ddim_sampleEIT(model, shape, args.samplingsteps, model_kwargs)
            else:
                out = diffusion.ddim_sampleEIT(model.module, shape, args.samplingsteps, model_kwargs)
            # print(out.shape)
            out = accelerator.gather(out)
            # out = out[:, 0:1, :, :]
            #

            # plt.subplot(121)
            # plt.imshow(x[0, 0].cpu())
            # plt.subplot(122)
            # plt.imshow(out[0, 0].cpu())
            # plt.show()
            x = accelerator.gather(x)
            rmse = (x - out).square().mean().sqrt()
            accelerator.print('out', i, out.shape, 'rmse: ', rmse)
            # while 1: pass
            RMSE.append(rmse)
            out = out.squeeze()
            x = x.squeeze()
            # accelerator.wait_for_everyone()
            # accelerator.print(type(gt1),gt1)
            # accelerator.wait_for_everyone()
            gt1.append(x)
            pred.append(out)

        pred = torch.cat(pred, dim=0)
        gt1 = torch.cat(gt1, dim=0)
        accelerator.print('out', pred.shape)
        RMSE = torch.stack(RMSE, dim=0)
        # accelerator.print('average RMSE', RMSE.mean())
        rmse = (gt1 - pred).square().mean().sqrt()
        pred1=pred.clone()
        pred = pred / 2 + 0.5
        gt1 = gt1 / 2 + 0.5
        max1, _ = torch.max(gt1, 1)
        max1, _ = torch.max(max1, 1)
        psnr = 10 * torch.log10(max1.square() / ((gt1 - pred).square().mean([1, 2]) + 1e-12))
        accelerator.print('PSNR ', psnr.mean())
        accelerator.print('RMSE whole ', rmse)
        torch.save(psnr.mean(),checkpoint_dir + '/'+'PSNR.pt' )
        # pred1 = pred.clone()
        # pred1[ind] = pred
        # pred = pred1
        if accelerator.is_main_process:
            sio.savemat(checkpoint_dir + '/' + modelname + '.mat',
                        {'pred': pred1.cpu() * dataTe.current / dataTe.voltage})
            # sio.savemat(checkpoint_dir + '/' +     'GT2023.mat',
            # {'GT': gt1.cpu()})


if __name__ == "__main__":

    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--results-dir", type=str, default="results")
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    # parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    # parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--samplingsteps", type=int, default=5)
    args = parser.parse_args()

    if os.name == 'nt':
        # main1gpu(args)
        # test1gpu(args)
        # main(args)
        test(args)
        #
        pass
    else:
        tic = time()
        main(args)
        toc = time()
        print('train time', toc - tic)
        test(args)
        tic = time()
        print('test time', tic - toc)

    # CUDA_VISIBLE_DEVICES=1 accelerate launch trainAccele.py

