#!/usr/bin/env python


import torch
import pickle
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from tqdm.auto import trange
import matplotlib.pyplot as plt


@torch.no_grad()
def main(model_path="pretrained_models/sg3-make-faces.pkl", num_imgs=10, device="cuda"):

    with open(model_path, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
        G.eval()

    for i in range(10):
        z1 = torch.randn([1, G.mapping.z_dim], device=device)
        z2 = torch.randn([1, G.mapping.z_dim], device=device)
        z3 = torch.randn([1, G.mapping.z_dim], device=device)
        z4 = torch.randn([1, G.mapping.z_dim], device=device)

        images = []
        for x in np.linspace(0, 1, num_imgs, endpoint=True):
            for y in np.linspace(0, 1, num_imgs, endpoint=True):
                z = z1 * (1 - x) * (1 - y) + z2 * (1 - x) * y + z3 * x * (1 - y) + z4 * x * y
                image = (G(z, None) * 127.5 + 128).clip(0, 255).cpu().squeeze(0)
                images.append(image)

        grid = make_grid(images, nrow=10).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(grid).save(f"Z{i}.png")


if __name__ == "__main__":
    main()
