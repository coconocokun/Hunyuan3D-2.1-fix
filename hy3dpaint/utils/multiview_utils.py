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
from accelerate.utils import get_balanced_memory, infer_auto_device_map


class multiviewDiffusionNet(nn.Module):
    def __init__(self, config, accelerator, dino_v2_model) -> None:
        super().__init__()
        
        if accelerator.is_main_process:
            print(f"Main process downloading repo '{config.multiview_pretrained_path}'...")
            local_repo_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path
            )
        
        accelerator.wait_for_everyone() # All processes wait here

        if not accelerator.is_main_process:
            # Other processes now load from the cache, which is very fast.
            local_repo_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path
            )
        print(f"Repo downloaded to: {local_repo_path}")

        # 1. Download the repo first
        pipeline_local_path = os.path.join(local_repo_path, "hunyuan3d-paintpbr-v2-1")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        custom_pipeline_path = os.path.join(current_dir, "..", "hunyuanpaintpbr")
        
        # --- START OF FIX ---

        # 1. Create an empty shell of the pipeline on the 'meta' device
        #    This is done on ALL processes and is very cheap.
        with torch.device('meta'):
             temp_pipeline = DiffusionPipeline.from_pretrained(
                 pipeline_local_path,
                 custom_pipeline=custom_pipeline_path,
                 torch_dtype=torch.float16,
                 low_cpu_mem_usage=True,
             )

        # 2. Get the available memory for each GPU.
        #    We pass the `temp_pipeline` object to be measured.
        max_memory = get_balanced_memory(
            temp_pipeline, # <-- Pass the meta-device model shell here
            max_memory=None,
            no_split_module_classes=["CLIPTextModel", "CLIPVisionModel"],
            dtype=torch.float16,
        )

        # 3. Exclude the DINO GPU from the memory map
        dino_device = dino_v2_model.device if dino_v2_model is not None else None
        if dino_device is not None and dino_device.type == 'cuda':
             dino_gpu_id = dino_device.index
             if dino_gpu_id in max_memory:
                 print(f"Process {accelerator.process_index}: Excluding GPU {dino_gpu_id} from device map.")
                 max_memory[dino_gpu_id] = 0

        # 4. Infer the device map using the model shell and our modified memory constraints.
        device_map = infer_auto_device_map(
             temp_pipeline, # <-- Use the same shell here
             max_memory=max_memory,
             no_split_module_classes=["CLIPTextModel", "CLIPVisionModel"],
             dtype=torch.float16
        )
        del temp_pipeline # Free the meta object
        print(f"Process {accelerator.process_index}: Inferred device map: {device_map}")

        # --- END OF FIX ---
        # 5. Load the pipeline from the local path using our custom device map.
        self.pipeline = DiffusionPipeline.from_pretrained(
            pipeline_local_path,
            custom_pipeline=custom_pipeline_path,
            torch_dtype=torch.float16,
            device_map=device_map, # <-- Use our custom map
        )


        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            self.dino_v2 = dino_v2_model

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
