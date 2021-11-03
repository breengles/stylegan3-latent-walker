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
import re
import numpy as np
import imgui
import PIL.Image
from gui_utils import imgui_utils
from . import renderer
import dnnlib

#----------------------------------------------------------------------------

class CaptureWidget:
    def __init__(self, viz):
        self.viz            = viz
        self.save_path           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca', 'edit_tensor.npy'))
        self.search_dirs         = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca'))]
        self.load_path           = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca', 'edit_tensor.npy'))
        self.dump_tensor     = False
        self.defer_frames   = 0
        self.disabled_time  = 0


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            with imgui_utils.grayed_out(self.disabled_time != 0):
                imgui.text('Load')
                imgui.same_line(viz.label_w)
                _changed, self.load_path = imgui_utils.input_text('##path_load', self.load_path, 1024,
                    flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                    width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                    help_text='LOADPATH')
                if imgui.is_item_hovered() and not imgui.is_item_active() and self.load_path != '':
                    imgui.set_tooltip(self.load_path)
                imgui.same_line()
                if imgui_utils.button('Load tensor', width=viz.button_w, enabled=(self.disabled_time == 0)):
                    imgui.open_popup(f'load_edit_popup')

                if imgui.begin_popup(f'load_edit_popup'):
                    def recurse(parents):
                        items = self.list_npys(parents)
                        if(items):
                            for item in items:
                                clicked, _state = imgui.menu_item(item.name)
                                if clicked:
                                    self.load_path = item.path
                                    weights = np.load(self.load_path)
                                    viz.edit_widget.set_batch_edit(weights)
                        else:
                            with imgui_utils.grayed_out():
                                imgui.menu_item('No results found')
                    
                    recurse(self.search_dirs)
                    imgui.end_popup()

                    
                imgui.same_line()
            with imgui_utils.grayed_out(self.disabled_time != 0):
                imgui.new_line()
                imgui.text('Save')
                imgui.same_line(viz.label_w)
                _changed, self.save_path = imgui_utils.input_text('##path_save', self.save_path, 1024,
                    flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                    width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                    help_text='SAVEPATH')
                if imgui.is_item_hovered() and not imgui.is_item_active() and self.save_path != '':
                    imgui.set_tooltip(self.save_path)
                imgui.same_line()
                if imgui_utils.button('Save tensor', width=viz.button_w, enabled=(self.disabled_time == 0)):
                    np.save(self.save_path, viz.args.batch_edit.layer_weights)
                    # self.dump_tensor = True
                    # self.defer_frames = 2
                    # self.disabled_time = 0.5
                imgui.same_line()


    def list_npys(self, parents):
        items = []
        npy_regex = re.compile(r'^.*\.(npy|NPY)$')
        for parent in set(parents):
            if os.path.isdir(parent):
                for entry in os.scandir(parent):
                    if entry.is_file() and npy_regex.fullmatch(entry.name):
                        items.append(dnnlib.EasyDict(name=entry.name, path=os.path.join(parent, entry.name)))
        return items

#----------------------------------------------------------------------------
