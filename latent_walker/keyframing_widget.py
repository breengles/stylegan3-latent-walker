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

import numpy as np
import imgui
import dnnlib
from gui_utils import imgui_utils
import os
import re
import imageio

#----------------------------------------------------------------------------

class KeyframingWidget:
    def __init__(self, viz):
        self.viz                            = viz
        self.search_dirs                    = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca'))]
        self.output_dir                     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pca'))
        self.keyframes                      = dnnlib.EasyDict([])
        self.edits                          = dnnlib.EasyDict(arr=[])
        self.keyframes_edits_weights        = dnnlib.EasyDict(arr=[])
        self.current_keyframe_idx           = 0
        self.name                           = "Edit Name"
        self.remove_edits_queue             = (-1, -1) # keyframe_idx, edit_idx
        self.remove_keyframe_queue          = -1
        self.current_frame                  = 0
        self.video_capturing                = False
        self.image_buffer                   = []
        self.keyframes_increment_default    = 100
        self.render_status_labels           = ["Idle", "Rendering "]


    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            bg_color = [0.16, 0.29, 0.48, 0.2]
            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5

            # Begin list.
            width = viz.font_size * 40
            height = imgui.get_text_line_height_with_spacing() * 5 + viz.spacing
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, *bg_color)
            imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.16, 0.29, 0.48, 0.5)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.16, 0.29, 0.48, 0.9)
            imgui.begin_child('##keyframes', width=width, height=height, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
           
            # Keyframes:
            self.remove_edits_if_queued()
            self.remove_keyframe_if_queued()
            for keyframe_idx in range(len(self.keyframes_edits_weights.arr)):
                # Edits:             
                for edit_idx in range(len(self.keyframes_edits_weights.arr[keyframe_idx].edits)):
                    # Load edit button:
                    load_hash = 100 * keyframe_idx + edit_idx
                    if imgui_utils.button(f'Load edit {load_hash}...', width=viz.button_w, enabled=True):
                        imgui.open_popup(f'load_edit_popup_{load_hash}')


                    if imgui.begin_popup(f'load_edit_popup_{load_hash}'):
                        def recurse(parents):
                            items = self.list_npys(parents)
                            if(items):
                                for item in items:
                                    clicked, _state = imgui.menu_item(item.name)
                                    if clicked:
                                        self.load_edit(keyframe_idx, edit_idx, item.name, item.path)
                            else:
                                with imgui_utils.grayed_out():
                                    imgui.menu_item('No results found')
                        
                        recurse(self.search_dirs)
                        imgui.end_popup()

                    # Edit Name:
                    imgui.same_line()
                    changed, name = imgui_utils.input_text(f'##Name_{load_hash}', self.keyframes_edits_weights.arr[keyframe_idx].edits[edit_idx].name, 1024,
                        flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                        width=(viz.button_w * 2),
                        help_text='Edit name')
                    # Edit Strength:
                    imgui.same_line()
                    with imgui_utils.item_width(-1 - 3 * viz.spacing), imgui_utils.grayed_out(False):
                        strength = self.keyframes_edits_weights.arr[keyframe_idx].edits[edit_idx].strength
                        changed, strength = imgui.slider_float(f'##edit_strength_{load_hash}', self.keyframes_edits_weights.arr[keyframe_idx].edits[edit_idx].strength, -200.0, 200.0, format='%.4f', power=3)
                        if changed:
                            self.keyframes_edits_weights.arr[keyframe_idx].edits[edit_idx].strength = strength
                    # Remove edit:
                    imgui.same_line()
                    if imgui_utils.button(f'X{load_hash}', width=-1, enabled=True):
                        self.queue_edit_for_delete(keyframe_idx, edit_idx)
                # Add edit button:
                if imgui_utils.button(f'Add edit ({keyframe_idx})', width=viz.button_w, enabled=True):
                    self.add_edit(keyframe_idx)
                # Target Latent:
                imgui.text('Latent:')
                imgui.same_line()
                target_latent = self.keyframes_edits_weights.arr[keyframe_idx].lat
                with imgui_utils.item_width(viz.font_size * 5):
                    changed, target_latent = imgui.input_int(f'##target_latent_{keyframe_idx}', target_latent)
                    if changed:
                        self.keyframes_edits_weights.arr[keyframe_idx].lat = target_latent
                # Target Frame:
                imgui.same_line()
                imgui.text('Frame:')
                imgui.same_line()
                target_frame = self.keyframes_edits_weights.arr[keyframe_idx].frame
                with imgui_utils.item_width(viz.font_size * 5):
                    changed, target_frame = imgui.input_int(f'##target_frame_{keyframe_idx}', target_frame)
                    if changed:
                        self.keyframes_edits_weights.arr[keyframe_idx].frame = target_frame

                # Delete button:
                imgui.same_line()
                if imgui_utils.button(f'Delete ({keyframe_idx})', width=viz.button_w, enabled=True):
                    self.queue_keyframe_for_delete(keyframe_idx)
                imgui.separator()

            imgui.end_child()
            imgui.pop_style_color(4)
            imgui.pop_style_var(1)

            # Add keyframe button:
            if imgui_utils.button('Add keyframe', width=viz.button_w * 3, enabled=True):
                self.add_keyframe()

            # Select keyframe:
            imgui.same_line()
            imgui.text('Select keyframe:')
            imgui.same_line()
            keyframe_idx = self.current_keyframe_idx
            with imgui_utils.item_width(viz.font_size * 5):
                changed, keyframe_idx = imgui.input_int('##selected_keyframe', keyframe_idx)
                if changed:
                    keyframes = len(self.keyframes_edits_weights.arr)
                    if keyframe_idx < 0:
                        self.current_keyframe_idx = 0
                    elif keyframe_idx > keyframes - 1:
                        self.current_keyframe_idx = max(keyframes - 1, 0)
                    else:
                        self.current_keyframe_idx = keyframe_idx
                    self.current_frame = self.keyframes_edits_weights.arr[self.current_keyframe_idx].frame
                    
            
            # Keyframing timeline:
            imgui.text('Timeline:')
            with imgui_utils.item_width(-1), imgui_utils.grayed_out(False):
                last_frame = self.get_last_frame()
                frame = self.current_frame
                if frame > last_frame: 
                    frame = last_frame
                if frame < 0: frame = 0
                changed, frame = imgui.slider_int(f'##timeline', frame, 0, last_frame)
                if changed:
                    self.current_frame = frame
                    self.video_capturing = False

            
            if imgui_utils.button('Render video', width=viz.button_w * 3, enabled=True):
                self.image_buffer = []
                self.current_frame = 0
                self.video_capturing = not self.video_capturing
                

            imgui.same_line()
            render_status_message = (self.render_status_labels[1] + self.get_video_out_path() if self.video_capturing
                                        else self.render_status_labels[0])
            imgui.text(f"Render status: {render_status_message}")

        if self.video_capturing:
            self.image_buffer.append(viz.result.image)
            self.current_frame += 1
            last_frame = self.get_last_frame()
            if self.current_frame == last_frame:
                self.video_capturing = False
                save_file_name = self.get_video_out_path()
                video_out = imageio.get_writer(save_file_name, mode='I', fps=60, codec='libx264', quality=9)
                viz.result.message = f'Saving video {save_file_name}...'
                viz.defer_rendering()
                for img in self.image_buffer:
                    video_out.append_data(img)
                video_out.close()
                
        viz.args.current_frame = self.get_current_frame_data()

    def get_video_out_path(self):
        return os.path.join(self.output_dir, "video_out.mp4")

    def add_keyframe(self):
        last = self.get_last_frame()
        f = 0 if last == -1 else last + 100
        self.keyframes_edits_weights.arr.extend([dnnlib.EasyDict(lat=100, frame=f, edits=[])])

    def can_display_keyframe(self):
        return len(self.keyframes_edits_weights.arr) > 0 and len(self.keyframes_edits_weights.arr[self.current_keyframe_idx].edits) > 0

    def ease_in_out(self, x):
        if x < 0.5:
            return 2.0 * x * x
        return -2.0 * x * x + 4 * x - 1

    def get_interpolated_frame_data(self, keyframe_from_idx, keyframe_to_idx, amount):
        weights_from = np.zeros((16, 20))
        weights_to = np.zeros((16, 20))
        weights_result = np.zeros((16, 20))
        seed_from = self.keyframes_edits_weights.arr[keyframe_from_idx].lat
        seed_to = self.keyframes_edits_weights.arr[keyframe_to_idx].lat
        amount = self.ease_in_out(amount)
        for e in self.keyframes_edits_weights.arr[keyframe_from_idx].edits:
            if(len(e.weights) > 0):
                weights_from += e.weights * e.strength * 0.01
        for e in self.keyframes_edits_weights.arr[keyframe_to_idx].edits:
            if(len(e.weights) > 0):
                weights_to += e.weights * e.strength * 0.01
        
        weights_result = (1.0 - amount) * weights_from + amount * weights_to

        return dnnlib.EasyDict(layer_weights=weights_result, w0_seeds=[[seed_from, 1-amount], [seed_to, amount]])

    def get_frame_data(self, keyframe_idx):
        weights_result = np.zeros((16, 20))
        seeds = [[self.keyframes_edits_weights.arr[keyframe_idx].lat, 1]]
        for e in self.keyframes_edits_weights.arr[keyframe_idx].edits:
            if(len(e.weights) > 0):
                weights_result += e.weights * e.strength * 0.01
        return dnnlib.EasyDict(layer_weights=weights_result, w0_seeds=seeds)
    
    def get_current_frame_data(self):
        weights = np.zeros((16, 20))
        keyframe_from_idx = -1
        keyframe_to_idx = -1
        current = self.current_frame

        if(self.keyframes_edits_weights.arr):
            frame_from = -1
            frame_to = 999999
            idx = 0
            for keyframe in self.keyframes_edits_weights.arr:
                if keyframe.frame <= current:
                    if keyframe.frame > frame_from:
                        frame_from = keyframe.frame
                        keyframe_from_idx = idx
                if keyframe.frame > current:
                    if keyframe.frame < frame_to:
                        frame_to = keyframe.frame
                        keyframe_to_idx = idx
                idx += 1
            
        if(keyframe_from_idx == -1):
            return dnnlib.EasyDict(layer_weights = weights, w0_seeds=[[0, 1]])
        if(keyframe_to_idx == -1 and keyframe_from_idx >= 0):
            return self.get_frame_data(keyframe_from_idx)
        
        frame_from = self.keyframes_edits_weights.arr[keyframe_from_idx].frame
        frame_to = self.keyframes_edits_weights.arr[keyframe_to_idx].frame
        amt = float(current - frame_from) / float(frame_to - frame_from)

        return self.get_interpolated_frame_data(keyframe_from_idx, keyframe_to_idx, amt)

    def get_current_frame_latent(self):
        if(self.keyframes_edits_weights.arr):
            return self.keyframes_edits_weights.arr[self.current_keyframe_idx].lat
        return 0

    def get_last_frame(self):
        last = -1
        if(self.keyframes_edits_weights.arr):
            for keyframe in self.keyframes_edits_weights.arr:
                last = max(keyframe.frame, last)
        return last

    def queue_keyframe_for_delete(self, keyframe_idx):
        self.remove_keyframe_queue = (keyframe_idx)

    def remove_keyframe_if_queued(self):
        if(self.remove_keyframe_queue >= 0):
            self.keyframes_edits_weights.arr.pop(self.remove_keyframe_queue)
            self.remove_keyframe_queue = -1

    def queue_edit_for_delete(self, keyframe_idx, edit_idx):
        self.remove_edits_queue = (keyframe_idx, edit_idx)

    def remove_edits_if_queued(self):
        if(self.remove_edits_queue[0] >= 0 and self.remove_edits_queue[1] >= 0):
            keyframe_idx = self.remove_edits_queue[0]
            edit_idx = self.remove_edits_queue[1]
            self.keyframes_edits_weights.arr[keyframe_idx].edits.pop(edit_idx)
            self.remove_edits_queue = (-1, -1)

    def add_edit(self, keyframe_idx):
        self.keyframes_edits_weights.arr[keyframe_idx].edits.extend([dnnlib.EasyDict(name="N/A", strength=100.0, weights=[])])

    def load_edit(self, keyframe_idx, edit_idx, name, path):
        weights = np.load(path)
        self.keyframes_edits_weights.arr[keyframe_idx].edits[edit_idx] = dnnlib.EasyDict(name=name, strength=100.0, weights=weights)

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
