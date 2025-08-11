# In file: hy3dpaint/utils/multiview_utils.py

import os
import torch
import torch.nn as nn
import huggingface_hub
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler
from omegaconf import OmegaConf

# Import the necessary accelerate utilities for manual loading
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from accelerate import load_checkpoint_and_dispatch


class multiviewDiffusionNet(nn.Module):
    def __init__(self, config, accelerator, dino_v2_model) -> None:
        super().__init__()

        # --- 1. Synchronize repository download and get local paths ---
        if accelerator.is_main_process:
            print(f"Main process downloading repo '{config.multiview_pretrained_path}'...")
            local_repo_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path
            )
        
        accelerator.wait_for_everyone() # All processes wait here

        if not accelerator.is_main_process:
            local_repo_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path
            )
        print(f"Repo downloaded to: {local_repo_path}")

        pipeline_local_path = os.path.join(local_repo_path, "hunyuan3d-paintpbr-v2-1")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        custom_pipeline_path = os.path.join(current_dir, "..", "hunyuanpaintpbr")

        # --- 2. Load the pipeline as an empty shell on the 'meta' device ---
        print("Loading pipeline shell on 'meta' device...")
        pipeline = DiffusionPipeline.from_pretrained(
            pipeline_local_path,
            custom_pipeline=custom_pipeline_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, # This is key for loading as a shell
        )

        # --- 3. Calculate the device map for the large UNet component ---
        print("Calculating device map for UNet...")
        
        # Get total available memory on all GPUs
        max_memory = get_balanced_memory(
            pipeline.unet,
            max_memory=None,
            no_split_module_classes=["CLIPTextModel", "CLIPVisionModel"],
            dtype=torch.float16,
        )

        # Exclude the GPU reserved for DINO from the memory map
        dino_device = dino_v2_model.device if dino_v2_model is not None else None
        if dino_device is not None and dino_device.type == 'cuda':
             dino_gpu_id = dino_device.index
             if dino_gpu_id in max_memory:
                 print(f"Process {accelerator.process_index}: Excluding GPU {dino_gpu_id} from device map for UNet.")
                 max_memory[dino_gpu_id] = 0

        # Infer the device map for the UNet using our modified memory constraints
        unet_device_map = infer_auto_device_map(
             pipeline.unet,
             max_memory=max_memory,
             no_split_module_classes=["CLIPTextModel", "CLIPVisionModel"],
             dtype=torch.float16
        )
        print(f"Process {accelerator.process_index}: Inferred UNet device map: {unet_device_map}")

        # --- 4. Manually dispatch the UNet and load its weights ---
        # This function moves the empty UNet shell to the GPUs and
        # then loads the checkpoint weights directly into it.
        print("Dispatching and loading UNet...")
        load_checkpoint_and_dispatch(
            pipeline.unet,
            os.path.join(pipeline_local_path, "unet"), # Path to the UNet's weights
            device_map=unet_device_map,
            no_split_module_classes=["CLIPTextModel", "CLIPVisionModel"],
            dtype=torch.float16
        )

        # --- 5. Manually place the smaller components (VAE, text_encoder) ---
        # Place them on the first available GPU that is not reserved for DINO.
        aux_device = 'cuda:0'
        if dino_device is not None and dino_device.type == 'cuda' and dino_device.index == 0:
            # If cuda:0 is taken by DINO, use the next one if available
            if torch.cuda.device_count() > 1:
                aux_device = 'cuda:1'
            
        print(f"Placing VAE and Text Encoder on {aux_device}")
        pipeline.vae.to(aux_device)
        pipeline.text_encoder.to(aux_device)

        # --- 6. Finalize the model object ---
        self.pipeline = pipeline
        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            self.dino_v2 = dino_v2_model
            
        self.no_split_modules = ["CLIPTextModel", "CLIPVisionModel", "HunyuanDiTTextEncoder", "LlamaForCausalLM"]

        print("multiviewDiffusionNet initialization complete.")

    def seed_everything(self, seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def forward(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False):
        """
        The forward pass is now named 'forward' to be a proper nn.Module.
        The logic inside is the same as the original __call__ method.
        """
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False):
        from typing import List
        from PIL import Image

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
        
        # When using a sharded model, device placement is handled by accelerate hooks.
        # We don't need to manually specify the device for the generator.
        kwargs = dict(generator=torch.Generator().manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            # The dino_v2 model is already on its correct device.
            # The input_images will be moved to the correct device by the model.
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states

        sync_condition = None
        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 30,
            "UniPCMultistepScheduler": 15,
            "DDIMScheduler": 50,
        }
        
        scheduler_name = self.pipeline.scheduler.__class__.__name__
        if scheduler_name not in infer_steps_dict:
            # Fallback for other schedulers
            infer_steps_dict[scheduler_name] = 15

        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=infer_steps_dict[scheduler_name],
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=3.0,
            **kwargs,
        ).images

        if "pbr" in self.pipeline.config.get("_name_or_path", "").lower():
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image