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

import imgui
from gui_utils import imgui_utils
import dnnlib
import numpy as np

#----------------------------------------------------------------------------

class EditWidget:
    def __init__(self, viz):
        self.viz                    = viz
        self.enables                = []
        self.pca_weights            = dnnlib.EasyDict(weights=np.zeros(20))
        self.batch_edit             = dnnlib.EasyDict(layer_weights=np.zeros((16, 20)))
        self.current_edit           = dnnlib.EasyDict(layer_weights=np.zeros((16, 20)))
        self.display_mode           = 0 # 0:Feature editor, 1:Original, 2:Keyframing
        self.display_mode_radio_idx = 0

    def set_batch_edit(self, weights):
        self.batch_edit.layer_weights = weights

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        num_ws = viz.result.get('num_ws', 0)
        num_enables = viz.result.get('num_ws', 16)
        self.enables += [True] * max(num_enables - len(self.enables), 0)

        if show:
            imgui.text('Layer toggles')
            pos2 = imgui.get_content_region_max()[0] - 1 - viz.button_w
            pos1 = pos2 - imgui.get_text_line_height() - viz.spacing
            pos0 = viz.label_w + viz.font_size * 5
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            for idx in range(num_enables):
                imgui.same_line(round(pos0 + (pos1 - pos0) * (idx / (num_enables - 1))))
                if idx == 0:
                    imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 3)
                with imgui_utils.grayed_out(num_ws == 0):
                    _clicked, self.enables[idx] = imgui.checkbox(f'##{idx}', self.enables[idx])
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f'{idx}')
            imgui.pop_style_var(1)

            imgui.same_line(pos2)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() - 3)
            with imgui_utils.grayed_out(num_ws == 0):
                if imgui_utils.button('Toggle All', width=-1, enabled=True):
                    if all(self.enables):
                        self.enables = [False for _ in self.enables]
                    else: 
                        self.enables = [True for _ in self.enables]

            bg_color = [0.16, 0.29, 0.48, 0.2]
            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5

            # Begin list.
            width = viz.font_size * 40
            height = imgui.get_text_line_height_with_spacing() * 21 + viz.spacing
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *bg_color)
            imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.16, 0.29, 0.48, 0.5)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.16, 0.29, 0.48, 0.9)
            imgui.begin_child('##list', width=width, height=height, border=True)

            # List items.
            for ch in range(len(self.pca_weights.weights)):
                with imgui_utils.item_width(-1 - viz.spacing), imgui_utils.grayed_out(False):
                    changed, offset = imgui.slider_float(f'##ch_offset{ch}', self.pca_weights.weights[ch], -100.0, 100.0, format='ch ' + str(ch+1) + ' %.1f', power=3)
                    if changed:
                        self.pca_weights.weights[ch] = offset

            imgui.end_child()
            imgui.pop_style_color(4)
            imgui.pop_style_var(1)

            def reset():
                self.pca_weights = dnnlib.EasyDict(weights = np.zeros(20))
                self.enables += [True] * max(num_enables - len(self.enables), 0)

            if imgui_utils.button('Add edit', width=viz.label_w, enabled=True):          
                layer_idx = [idx for idx, enable in enumerate(self.enables) if enable]  
                for l in layer_idx:
                    self.batch_edit.layer_weights[l] += self.current_edit.layer_weights[l]
                reset()

            imgui.same_line()
            if imgui_utils.button('Delete edits', width=viz.label_w, enabled=True):
                reset()
                self.batch_edit = dnnlib.EasyDict(layer_weights=np.zeros((16, 20)))

            imgui.same_line()
            if imgui_utils.button('Reset', width=viz.label_w, enabled=True):
                reset()

            imgui.begin_child('##options', width=-1, height=imgui.get_text_line_height_with_spacing() * 1, border=False)

            imgui.text('Display mode: ')
            imgui.same_line()
            clicked = imgui.radio_button('Edit Target', self.display_mode == 0)
            if clicked:
                self.display_mode = 0
            imgui.same_line()
            clicked = imgui.radio_button('Original', self.display_mode == 1)
            if clicked:
                self.display_mode = 1
            imgui.same_line()
            clicked = imgui.radio_button('Keyframing', self.display_mode == 2)
            if clicked and viz.keyframing_widget.can_display_keyframe():
                self.display_mode = 2

            # End options.
            imgui.end_child()


        self.current_edit = dnnlib.EasyDict(layer_weights=np.zeros((16, 20)))
        layer_idx = [idx for idx, enable in enumerate(self.enables) if enable]
        for l in layer_idx:
            self.current_edit.layer_weights[l] = self.pca_weights.weights

        viz.args.display_mode = self.display_mode

        target_batch_edit = dnnlib.EasyDict(layer_weights=np.zeros((16, 20)))
        target_batch_edit.layer_weights = np.add(self.batch_edit.layer_weights , self.current_edit.layer_weights)
        viz.args.batch_edit = target_batch_edit


#---------------------------------------------------------------------------- 
