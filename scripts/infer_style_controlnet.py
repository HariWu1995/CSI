import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline as SDXLControlnetPipeline

import cv2
from PIL import Image

from ip_adapter import IPAdapterXL


base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_adapter_path = "sdxl_models/ip-adapter_sdxl.bin"

controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=False).to(device)

# load SDXL pipeline
pipe = SDXLControlnetPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16,
                                                                controlnet=controlnet, add_watermarker=False)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks = ["block"] # for original IP-Adapter
# target_blocks = ["up_blocks.0.attentions.1"] # for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_adapter_path, device, 
                        target_blocks=["up_blocks.0.attentions.1"])

# style image
image = "./assets/InstantStyle/4.jpg"
image = Image.open(image)
image.resize((512, 512))

# control image
input_image = cv2.imread("./assets/InstantStyle/yann-lecun.jpg")
detected_map = cv2.Canny(input_image, 50, 200)
canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

# generate image
images = ip_model.generate(pil_image=image,
                           prompt="a man, masterpiece, best quality, high quality",
                  negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=30, 
                           seed=42,
                           image=canny_map,
                           controlnet_conditioning_scale=0.6,
                          )

images[0].save("result.png")

