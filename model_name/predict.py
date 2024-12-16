from typing import Any
from cog import BasePredictor, Input, Path

import torch
import tempfile
from diffusers import StableDiffusionXLPipeline


class Predictor(BasePredictor):
    def setup(self):
        """Load the model"""
        model_id = "segmind/SSD-1B"
        model_dir = "model"
        device = self.get_device()

        if device == "cuda" or device == "mps":
            self.pipe = StableDiffusionXLPipeline.from_pretrained(model_dir, torch_dtype=torch.float16, use_safetensors=True)

        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_dir, use_safetensors=True
            )

        if device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(device)


    def get_device(self):
        # 우선순위: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS device (Apple Silicon).")
        else:
            device = "cpu"
            print("Using CPU.")
        return device

    # Define the arguments and types the model takes as input
    def predict(
        self, 
        prompt: str = Input(
            description="Input prompt",
            default="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        ),
        negative_prompt: str = Input(
            description="Negative Input prompt",
            default="ugly, deformed, noisy, blurry, distorted"
        ),
        width: int = Input(
            description="Width of output image",
            default=512, ge=256, le=1536
        ),
        height: int = Input(
            description="Height of output image",
            default=512, ge=256, le=1536
        ),
        step: int = Input(
            description="Number of denoising steps", ge=1, le=60, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0.1, le=20, default=3
        )
    ) -> Path:
        image = self.pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=step, 
            guidance_scale=guidance_scale,
            width=width,
            height=height    
        ).images[0]

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        image.save(str(out_path), 'png')
        print(str(out_path))

        return out_path

