# worker.py

import torch
import time
import os
import json
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import infer_auto_device_map, get_balanced_memory
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from PIL import Image
import traceback

# --- Configuration ---
# Folders for communication between Gradio and the worker
JOB_DIR = "./job_queue"
os.makedirs(JOB_DIR, exist_ok=True)
JOB_FILE = os.path.join(JOB_DIR, "job.json")
RESULT_FILE = os.path.join(JOB_DIR, "result.json")
# ---

def main():
    print("Initializing Accelerate...")
    accelerator = Accelerator()
    print(f"Worker running on device: {accelerator.device}")

    # Load the configuration for the pipeline
    conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    
    # Load the pipeline using the accelerator for model sharding
    if accelerator.is_main_process:
        print("Main process is loading the pipeline...")
    
    # We must use init_empty_weights to handle models larger than system RAM
    with init_empty_weights():
        tex_pipeline = Hunyuan3DPaintPipeline(conf, accelerator=accelerator)

    # The main process will print the device map
    if accelerator.is_main_process:
        print("--- Multiview Model Device Map ---")
        print(tex_pipeline.models["multiview_model"].hf_device_map)
        print("--- Super-Res Model Device Map ---")
        print(tex_pipeline.models["super_model"].hf_device_map)
    
    # Main worker loop
    if accelerator.is_main_process:
        print("\nWorker is ready and waiting for jobs...")
        while True:
            if os.path.exists(JOB_FILE):
                try:
                    print("Found a job. Processing...")
                    with open(JOB_FILE, 'r') as f:
                        job_data = json.load(f)
                    
                    # Clean up the job file immediately
                    os.remove(JOB_FILE)

                    # Execute the pipeline
                    # Note: image_path is not used, we pass the image directly
                    # For simplicity, we assume the input mesh path is absolute or reachable
                    result_path = tex_pipeline(
                        mesh_path=job_data['mesh_path'],
                        image_path=Image.open(job_data['image_path']), # Re-open the image
                        output_mesh_path=job_data['output_mesh_path'],
                        save_glb=False
                    )

                    # Write success result
                    with open(RESULT_FILE, 'w') as f:
                        json.dump({'status': 'success', 'path': result_path}, f)
                    print("Job finished successfully.")

                except Exception as e:
                    print(f"ERROR processing job: {e}")
                    traceback.print_exc()
                    # Write error result
                    with open(RESULT_FILE, 'w') as f:
                        json.dump({'status': 'error', 'message': str(e)}, f)

            time.sleep(1) # Poll every second

if __name__ == "__main__":
    main()