import argparse
import copy
import os
from tqdm import tqdm
from datetime import datetime
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

from model import Generator, Discriminator
from dataset import Dataset, infinite_loader
from augmentation import DiffAugment
from loss import get_adversarial_losses, get_regularizer


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="path to the dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu/cuda (does not support multi-GPU training for now)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="log root directory",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=100,
        help="sample log period",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=10000,
        help="checkpoint save period",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="base learning rate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="train steps",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="num of log samples",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="num of workers for dataloader",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint file to continue training",
    )
    parser.add_argument(
        "--n_basis",
        type=int,
        default=6,
        help="subspace dimension for a generator layer",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=512,
        help="noise dimension for the input layer of the generator",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=16,
        help="num of base channels for generator/discriminator",
    )
    parser.add_argument(
        "--max_channels",
        type=int,
        default=512,
        help="max num of channels for generator/discriminator",
    )
    parser.add_argument(
        "--adv_loss",
        choices=["hinge", "non_saturating", "lsgan"],
        default="hinge",
        help="adversarial loss type",
    )
    parser.add_argument(
        "--orth_reg",
        type=float,
        default=100.0,
        help="basis orthogonality regularization weight",
    )
    parser.add_argument(
        "--d_reg",
        type=float,
        default=10.0,
        help="discriminator r1 regularization weight",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=4,
        help="discriminator lazy regularization period",
    )
    args = parser.parse_args()

    device = args.device

    # models
    generator = Generator(
        size=args.size,
        n_basis=args.n_basis,
        noise_dim=args.noise_dim,
        base_channels=args.base_channels,
        max_channels=args.max_channels,
    ).to(device).train()
    g_ema = copy.deepcopy(generator).eval()

    discriminator = Discriminator(
        size=args.size,
        base_channels=args.base_channels,
        max_channels=args.max_channels,
    ).to(device).train()

    # optimizers
    g_optim = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.99),
    )

    d_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.99),
    )

    start_step = 0
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        start_step = ckpt["step"]

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    # losses
    d_adv_loss_fn, g_adv_loss_fn = get_adversarial_losses(args.adv_loss)
    d_reg_loss_fn = get_regularizer("r1")

    # data
    transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    loader = infinite_loader(
        DataLoader(
            Dataset(args.path, transform),
            batch_size=args.batch,
            shuffle=True,
            drop_last=True,
            num_workers=args.n_workers,
        )
    )

    # train utils
    augment = DiffAugment(policy='color,translation,cutout', p=0.6)
    ema = partial(accumulate, decay=0.5 ** (args.batch / (10 * 1000)))

    logdir = os.path.join(
        args.logdir,
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    os.makedirs(os.path.join(logdir, "samples"))
    os.makedirs(os.path.join(logdir, "checkpoints"))
    tb_writer = SummaryWriter(logdir)
    log_sample = g_ema.sample_latents(args.n_sample)
    print(f"training log directory: {logdir}")

    # train loop
    for step in (iterator := tqdm(range(args.steps), initial=start_step)):

        step = step + start_step

        real = next(loader).to(device)

        # D
        with torch.no_grad():
            fake = generator.sample(args.batch)
        real_pred = discriminator(augment(real))
        fake_pred = discriminator(augment(fake))

        d_loss = d_adv_loss_fn(real_pred, fake_pred)

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if step % args.d_reg_every == 0:
            real.requires_grad = True
            real_pred = discriminator(augment(real))
            r1 = d_reg_loss_fn(real_pred, real) * args.d_reg

            discriminator.zero_grad()
            r1.backward()
            d_optim.step()

        # G
        fake = generator.sample(args.batch)
        fake_pred = discriminator(augment(fake))

        g_loss_adv = g_adv_loss_fn(fake_pred)
        g_loss_reg = generator.orthogonal_regularizer() * args.orth_reg
        g_loss = g_loss_adv + g_loss_reg

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        ema(g_ema, generator)

        # log
        iterator.set_description(
            f"d: {d_loss.item():.4f}; g: {g_loss.item():.4f} "
        )

        tb_writer.add_scalar("loss/D", d_loss.item(), step)
        tb_writer.add_scalar("loss/D_r1", r1.item(), step)
        tb_writer.add_scalar("loss/G", g_loss.item(), step)
        tb_writer.add_scalar("loss/G_orth", g_loss_reg.item(), step)
        tb_writer.add_scalar("loss/G_adv", g_loss_adv.item(), step)

        if step % args.sample_every == 0:
            with torch.no_grad():
                utils.save_image(
                    g_ema(log_sample),
                    os.path.join(logdir, "samples", f"{str(step).zfill(7)}.png"),
                    nrow=int(args.n_sample ** 0.5),
                    normalize=True,
                    value_range=(-1, 1),
                )

        if step % args.ckpt_every == 0:
            torch.save(
                {
                    "step": step,
                    "args": args,
                    "g": generator.state_dict(),
                    "d": discriminator.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                },
                os.path.join(logdir, "checkpoints", f"{str(step).zfill(7)}.pt"),
            )
