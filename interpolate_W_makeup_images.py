#!/usr/bin/env python


import torch
import pickle
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from tqdm.auto import trange
import matplotlib.pyplot as plt


@torch.no_grad()
def main(model_path="models/StyleGAN3-R_makeup_512x512.pkl", num_imgs=10, device="cuda"):

    with open(model_path, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
        G.eval()

    for i in range(10):
        z1 = torch.randn([1, G.mapping.z_dim], device=device)
        z2 = torch.randn([1, G.mapping.z_dim], device=device)
        z3 = torch.randn([1, G.mapping.z_dim], device=device)
        z4 = torch.randn([1, G.mapping.z_dim], device=device)

        w1 = G.mapping(z1, None, truncation_psi=1, truncation_cutoff=8)
        w2 = G.mapping(z2, None, truncation_psi=1, truncation_cutoff=8)
        w3 = G.mapping(z3, None, truncation_psi=1, truncation_cutoff=8)
        w4 = G.mapping(z4, None, truncation_psi=1, truncation_cutoff=8)

        images = []
        for x in np.linspace(0, 1, num_imgs, endpoint=True):
            for y in np.linspace(0, 1, num_imgs, endpoint=True):
                w = w1 * (1 - x) * (1 - y) + w2 * (1 - x) * y + w3 * x * (1 - y) + w4 * x * y
                image = (G.synthesis(w) * 127.5 + 128).clip(0, 255).cpu().squeeze(0)
                images.append(image)

        grid = make_grid(images, nrow=10).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(grid).save(f"W{i}.png")


if __name__ == "__main__":
    main()
