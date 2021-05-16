import os
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import utils

from model import Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt",
        type=str,
        help="checkpoint file to continue training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu/cuda",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./results",
        help="result save directory",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="num of samples",
    )
    parser.add_argument(
        "--traverse",
        action="store_true",
        help="traverse all eigen dimensions",
    )
    args = parser.parse_args()

    device = args.device

    ckpt = torch.load(args.ckpt, map_location="cpu")

    train_args = ckpt["args"]

    g_ema = Generator(
        size=train_args.size,
        n_basis=train_args.n_basis,
        noise_dim=train_args.noise_dim,
        base_channels=train_args.base_channels,
        max_channels=train_args.max_channels,
    ).to(device).eval()
    g_ema.load_state_dict(ckpt["g_ema"])

    logdir = os.path.join(args.logdir, train_args.name, str(ckpt["step"]).zfill(7))
    os.makedirs(logdir)
    print(f"result path: {logdir}")

    with torch.no_grad():
        utils.save_image(
            g_ema.sample(args.n_sample),
            os.path.join(logdir, "sample.png"),
            nrow=int(args.n_sample ** 0.5),
            normalize=True,
            value_range=(-1, 1),
        )

    if args.traverse:
        print("traversing:")
        traverse_samples = 8
        traverse_range = 4.0
        intermediate_points = 9
        truncation = 0.7

        es, zs = g_ema.sample_latents(traverse_samples, truncation=truncation)

        _, n_layers, n_dim = zs.shape

        offsets = np.linspace(-traverse_range, traverse_range, intermediate_points)
        for i_layer in range(n_layers):
            for i_dim in range(n_dim):
                print(f"  layer {i_layer} - dim {i_dim}")
                imgs = []
                for offset in offsets:
                    _zs = zs.clone()
                    _zs[:, i_layer, i_dim] = offset
                    with torch.no_grad():
                        img = g_ema((es, _zs)).cpu()
                        img = torch.cat([_img for _img in img], dim=1)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=2)

                imgs = (imgs.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
                Image.fromarray(imgs).save(
                    os.path.join(logdir, f"traverse_L{i_layer}_D{i_dim}.png")
                )
