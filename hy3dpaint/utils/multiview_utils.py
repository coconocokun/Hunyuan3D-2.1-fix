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
    def __init__(self, config, accelerator) -> None:
        super().__init__()

        self.device = config.device
        cfg_path = config.multiview_cfg_path
        
        # 1. MAKE THE CUSTOM PIPELINE PATH ABSOLUTE
        # This resolves any ambiguity about the current working directory.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        custom_pipeline_path = os.path.join(current_dir, "..", "hunyuanpaintpbr")

        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        # 2. SYNCHRONIZE MODEL DOWNLOAD
        # Only the main process should download the model from the Hub.
        # Other processes will wait and then load from the local cache.
        if accelerator.is_main_process:
            print("Main process downloading model snapshot...")
            model_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path,
                allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
            )
        
        # All processes wait here until the main process is done downloading.
        accelerator.wait_for_everyone()

        # Non-main processes can now safely load from the cache.
        if not accelerator.is_main_process:
            model_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path,
                allow_patterns=["hunyuan3d-paintpbr-v2-1/*"],
            )

        # model_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")
        self.checkpoint_path = os.path.join(model_path, "hunyuan3d-paintpbr-v2-1")
        
        # Now, from_pretrained will work correctly on all processes.
        pipeline = DiffusionPipeline.from_pretrained(
            self.checkpoint_path,
            custom_pipeline=custom_pipeline_path, # Use the absolute path
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        pipeline.set_progress_bar_config(disable=True)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        
        # Note: self.pipeline is not moved to a device here.
        # Accelerate's dispatch_model will handle placing its components.
        self.pipeline = pipeline

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            # The dino_v2 model will also be placed by dispatch_model

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
