from PIL import Image

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from diffusers import StableDiffusionXLPipeline as SDXLPipeline
from ip_adapter import IPAdapterXL


base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_adapter_path = "sdxl_models/ip-adapter_sdxl.bin"


# load SDXL pipeline
pipe = SDXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks = ["block"] # for original IP-Adapter
# target_blocks = ["up_blocks.0.attentions.1"] # for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_adapter_path, device, 
                        target_blocks=["up_blocks.0.attentions.1"])

image = "./assets/InstantStyle/0.jpg"
image = Image.open(image)
image.resize((512, 512))

# generate image
images = ip_model.generate(pil_image=image,
                           prompt="a cat, masterpiece, best quality, high quality",
                  negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=30, 
                           seed=42,
                           # neg_content_prompt="a rabbit",
                           # neg_content_scale=0.5,
                          )

images[0].save("result.png")