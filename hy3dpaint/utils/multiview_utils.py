# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import torch
import random
import numpy as np
from PIL import Image
from typing import List
import huggingface_hub
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler
import torch.nn as nn


class multiviewDiffusionNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        # 1. Make the custom pipeline path absolute (this is still good practice)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        custom_pipeline_path = os.path.join(current_dir, "..", "hunyuanpaintpbr")

        # 2. Let diffusers handle everything with device_map="auto"
        # This will automatically:
        # - Download the model if needed (with synchronization)
        # - Instantiate the pipeline with empty weights
        # - Infer the device map
        # - Load the weights directly onto the target devices
        print("Loading and sharding multiview pipeline with device_map='auto'...")
        self.pipeline = DiffusionPipeline.from_pretrained(
            config.multiview_pretrained_path,
            subfolder="hunyuan3d-paintpbr-v2-1", # Use subfolder for a cleaner repo ID
            custom_pipeline=custom_pipeline_path,
            torch_dtype=torch.float16,
            device_map="auto" # <-- The magic argument!
        )

        # The pipeline is now a sharded model.
        # We need to place the dino_v2 model on one of the GPUs,
        # typically the one where the first part of the UNet is.
        if hasattr(self.pipeline, "device"):
             dino_device = self.pipeline.device
        else:
             # Fallback if the main pipeline is on 'meta'
             dino_device = self.pipeline.unet.device

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            # Load Dino_v2 and move it to the correct device
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(dtype=torch.float16, device=dino_device)

        self.no_split_modules = ["CLIPTextModel", "CLIPVisionModel", "HunyuanDiTTexEncoder", "LlamaForCausalLM"]

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)
    
    @torch.no_grad()
    def forward(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input
        )
        return pils
    

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False):
        self.seed_everything(0)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        if not isinstance(input_images, List):
            input_images = [input_images]
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 30,
            "UniPCMultistepScheduler": 15,
            "DDIMScheduler": 50,
            "ShiftSNRScheduler": 15,
        }

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__],
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=3.0,
            **kwargs,
        ).images

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image
