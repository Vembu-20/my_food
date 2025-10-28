'''
!pip insatll diffusers
!pip install torch torchvision transformers accelerate safetensors
!pip install pillow
!pip install diffusers transformers accelerate safetensors torch --upgrade

from diffusers import StableDiffusionPipeline
import torch

# ✅ Use v1-5 instead of CompVis/v1-4
model = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompt = "A beautiful landscape painting in the style of Van Gogh"
image = model(prompt).images[0]
image.save("output.png")
prompt = "A 22-year-old Indian girl wearing business casual clothes, standing outside a modern multinational company building in Bangalore city, bustling urban background, realistic photograph"

image = model(prompt).images[0]
image.save("bangalore_mnc_outside.png")
'''