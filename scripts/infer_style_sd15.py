from PIL import Image

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from diffusers import StableDiffusionPipeline as SDPipeline, UniPCMultistepScheduler as Scheduler
from ip_adapter import IPAdapter


base_model_path = "sd-legacy/stable-diffusion-v1-5"
image_encoder_path = "models/image_encoder"
ip_adapter_path = "models/ip-adapter_sd15.bin"
clip_model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


# load SDXL pipeline
pipe = SDPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
)
pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks = ["block"] # for original IP-Adapter
# target_blocks = ["up_blocks.0.attentions.1"] # for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapter(pipe, image_encoder_path, ip_adapter_path, device, target_blocks=["block"])

image = "./assets/InstantStyle/3.jpg"
image = Image.open(image)
image.resize((512, 512))

# set negative content
neg_content = "a girl"
neg_content_scale = 0.8

if neg_content is None:
    neg_content_emb = None

else:
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection as CLIPTextModel

    text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(pipe.device, dtype=pipe.dtype)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

    tokens = tokenizer([neg_content], return_tensors='pt').to(pipe.device)
    neg_content_emb = text_encoder(**tokens).text_embeds
    neg_content_emb *= neg_content_scale

# generate image with content subtraction
images = ip_model.generate(pil_image=image,
                           prompt="a cat, masterpiece, best quality, high quality",
                  negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=30, 
                           seed=42,
                           neg_content_emb=neg_content_emb,
                          )

images[0].save("result.png")
