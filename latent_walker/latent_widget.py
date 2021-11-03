# This code is a derivative work by https://github.com/taptoi of two works: 
# 1. Alias-Free Generative Adversarial Networks (StyleGAN3)
# (https://github.com/NVlabs/stylegan3) NVIDIA Source Code License for StyleGAN3
# 2. GANSpace: Discovering Interpretable GAN Controls
# (https://github.com/harskish/ganspace) Apache License v2.0
# All changes are applied to the StyleGAN3 repo and no code was copied or
# otherwise used from the GANSpace repo. Copied and modified source files
# retain the original Copyright notices.

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils
import torch
import legacy

#----------------------------------------------------------------------------

def run_pca(self, viz, samples):
    if viz.args.pkl is None:
        return

    viz.defer_rendering()
    seeds = range(samples)
    device = torch.device('cuda')
    with dnnlib.util.open_url(viz.args.pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    # Generate zs from seeds
    # np.random.shuffle(seeds)
    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in seeds])).to(device)
    # Map zs to the intermediate latent space w
    ws = G.mapping(z=zs, c=None, truncation_psi=1.0)
    # Perform principle component analysis on ws:
    # take layer on all samples separately
    s = ws.transpose(0,1)
    print('Perform principle component analysis on ws...')
    # create principal component matrix vs
    # ws has n samples (amount of seeds), each sample has 18 layers with 512 channels each
    # Run PCA on each layer independently producing up to 20 components.
    # The produced principal directions matrix vs is 18x20x512 (layers x components x channels)
    # When moving along a component, we need to add the respective component (multiplied by some factor) 
    # from vs to the layers channel values
    layers = 16
    components = 20
    channels = 512
    vs = torch.zeros((layers,components,channels), device = torch.device('cuda'))
    #skiplayers = 1
    for layer in range(layers):
        _, _, V = torch.pca_lowrank(s[layer], q=components, center=True) # v is 512x20
        vs[layer] = torch.transpose(V, 0, 1)

    self.vs = dnnlib.EasyDict(V=vs.detach().cpu().numpy())


class LatentWidget:
    def __init__(self, viz):
        self.viz        = viz
        self.latent     = dnnlib.EasyDict(x=0, y=0, anim=False)
        self.latent_def = dnnlib.EasyDict(self.latent)
        self.step_y     = 100
        self.samples    = 2000
        self.vs         = dnnlib.EasyDict(V=np.zeros((16, 20, 512)))
        self.save_path  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca', 'pca_vectors.npy'))

    def drag(self, dx, dy):
        viz = self.viz
        self.latent.x += dx / viz.font_size * 4e-2
        self.latent.y += dy / viz.font_size * 4e-2

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Start Latent')
            imgui.same_line(viz.label_w)
            seed = round(self.latent.x) + round(self.latent.y) * self.step_y
            with imgui_utils.item_width(viz.font_size * 8):
                changed, seed = imgui.input_int('##seed', seed)
                if changed:
                    self.latent.x = seed
                    self.latent.y = 0
            imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
            frac_x = self.latent.x - round(self.latent.x)
            frac_y = self.latent.y - round(self.latent.y)
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (new_frac_x, new_frac_y) = imgui.input_float2('##frac', frac_x, frac_y, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.latent.x += new_frac_x - frac_x
                    self.latent.y += new_frac_y - frac_y
            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.spacing * 2)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.button_w)
            if dragging:
                self.drag(dx, dy)
            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.button_w + viz.spacing * 3)
            imgui.same_line()
            if imgui_utils.button('Reset', width=-1, enabled=(self.latent != self.latent_def)):
                self.latent = dnnlib.EasyDict(self.latent_def)

        viz.args.w0_seeds = [] # [[seed, weight], ...]
        for ofs_x, ofs_y in [[0, 0], [1, 0], [0, 1], [1, 1]]:
            seed_x = np.floor(self.latent.x) + ofs_x
            seed_y = np.floor(self.latent.y) + ofs_y
            seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
            weight = (1 - abs(self.latent.x - seed_x)) * (1 - abs(self.latent.y - seed_y))
            if weight > 0:
                viz.args.w0_seeds.append([seed, weight])

        imgui.text('Samples')
        imgui.same_line(viz.label_w)
        samples = self.samples
        with imgui_utils.item_width(viz.font_size * 8):
            changed, samples = imgui.input_int('##samples', samples)
            if changed:
                self.samples = samples

        imgui.same_line(viz.label_w + viz.font_size * 8 + viz.spacing)
        imgui.text('Run PCA')
        imgui.same_line(viz.label_w * 2 + viz.font_size * 8 + viz.spacing)
        if imgui_utils.button('Run_PCA##opts', width=-1, enabled=True):
            run_pca(self, viz, self.samples)

        imgui.text('PCA Vectors')
        imgui.same_line(viz.label_w)
        _changed, self.save_path = imgui_utils.input_text('##path', self.save_path, 1024,
            flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
            width=(-1 - viz.button_w * 2 - viz.spacing * 2),
            help_text='PATH')
        if imgui.is_item_hovered() and not imgui.is_item_active() and self.save_path != '':
            imgui.set_tooltip(self.save_path)
        imgui.same_line()
        if imgui_utils.button('Load', width=viz.button_w, enabled=True):
            self.vs.V = np.load(self.save_path)
        imgui.same_line()
        if imgui_utils.button('Save', width=-1, enabled=True):
           np.save(self.save_path, self.vs.V)

        viz.args.vs = self.vs

#----------------------------------------------------------------------------
