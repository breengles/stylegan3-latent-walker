#!/usr/bin/env python


import torch
import pickle
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from tqdm.auto import trange
import matplotlib.pyplot as plt


@torch.no_grad()
def main(model_path="models/StyleGAN3-R_makeup_512x512.pkl", num_imgs=10, batch_size=1, device="cuda"):

    with open(model_path, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
        G.eval()

    for idx in trange(num_imgs // batch_size):
        zs = torch.randn([batch_size, G.mapping.z_dim], device=device)

        # q = G.mapping(zs, None, truncation_psi=0.7)
        # q = (q - G.mapping.w_avg) / w_stds
        images = (G(zs, None) * 127.5 + 128).clamp(0, 255).cpu().numpy()

        for img_ in images:
            img = img_.transpose(1, 2, 0)
            Image.fromarray(img.astype(np.uint8)).save(f"makeup_dataset/{idx}.png")


if __name__ == "__main__":
    main(num_imgs=10000)
