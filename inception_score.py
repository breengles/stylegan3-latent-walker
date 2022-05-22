#!/usr/bin/env python


from torchmetrics.image.inception import InceptionScore
import pickle
import torch
from tqdm.auto import trange


@torch.no_grad()
def update_score(inception_score, G, n=1000):
    gen_images = []
    for _ in range(n):
        zs = torch.randn([1, G.mapping.z_dim], device="cuda")
        images = (G(zs, None) * 127.5 + 128).clamp(0, 255).cpu().to(torch.uint8).squeeze(0)
        gen_images.append(images)

    gen_images = torch.stack(gen_images)
    inception_score.update(gen_images)

    del gen_images


@torch.no_grad()
def main(model_path="models/StyleGAN3-R_makeup_512x512.pkl", device="cuda", n=50, k=1000):
    with open(model_path, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device)
        G.eval()

    inception_score = InceptionScore()

    for _ in trange(n):
        update_score(inception_score, G, k)

    del G

    print(inception_score.compute())


if __name__ == "__main__":
    main()
