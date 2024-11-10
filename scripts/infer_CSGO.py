import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from PIL import Image

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import BlipProcessor, BlipForConditionalGeneration as BlipGenerator
from transformers import AutoImageProcessor, AutoModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline as SDXLControlnetPipeline,
)

from ip_adapter.utils import BLOCKS, CTRLNET_BLOCKS
from ip_adapter.utils import resize_content
from ip_adapter import CSGO


base_model_path =  "../../../base_models/stable-diffusion-xl-base-1.0"
image_encoder_path = "../../../base_models/IP-Adapter/sdxl_models/image_encoder"
auto_encoder_path ='../../../base_models/sdxl-vae-fp16-fix'
controlnet_path = "../../../base_models/TTPLanet_SDXL_Controlnet_Tile_Realistic"

ip_adapter_path = ".../ip_adapter.bin"

blip_model = "Salesforce/blip-image-captioning-large"
blip_processor = BlipProcessor.from_pretrained(blip_model)
blip_generator = BlipGenerator.from_pretrained(blip_model).to(device)

auto_encoder = AutoencoderKL.from_pretrained(auto_encoder_path, torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
pipe = SDXLControlnetPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16,
                                                                controlnet=controlnet, 
                                                                       vae=auto_encoder, add_watermarker=False)
pipe.enable_vae_tiling()

target_content_blocks = BLOCKS['content']
target_style_blocks = BLOCKS['style']

controlnet_target_content_blocks = CTRLNET_BLOCKS['content']
controlnet_target_style_blocks = CTRLNET_BLOCKS['style']

csgo = CSGO(pipe, image_encoder_path, ip_adapter_path, device, 
            num_content_tokens=4,
            num_style_tokens=32,
            target_content_blocks=target_content_blocks, 
            target_style_blocks=target_style_blocks,
            controlnet_adapter=True,
            controlnet_target_content_blocks=controlnet_target_content_blocks,
            controlnet_target_style_blocks=controlnet_target_style_blocks,
            content_model_resampler=True,
            style_model_resampler=True,)

style_name = 'img_1.png'
content_name = 'img_0.png'
style_image = Image.open(f"../assets/CSGO/{style_name}").convert('RGB')
content_image = Image.open(f"../assets/CSGO/{content_name}").convert('RGB')

with torch.no_grad():
    inputs = blip_processor(content_image, return_tensors="pt").to(device)
    output = blip_generator.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)

num_samples = 1
width, height, content_image = resize_content(content_image)

images = csgo.generate( pil_content_image=content_image, 
                        pil_style_image=style_image,
                        prompt=caption,
               negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                        height=height,
                        width=width,
                        style_scale=1.0,
                        content_scale=0.5,
                        guidance_scale=10,
                        num_images_per_prompt=num_samples,
                        num_samples=1,
                        num_inference_steps=50,
                        seed=42,
                        image=content_image.convert('RGB'),
                        controlnet_conditioning_scale=0.6,)

images[0].save("../assets/CSGO/content_img_0_style_imag_1.png")
