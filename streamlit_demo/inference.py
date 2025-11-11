import torch
from PIL import Image

def generate_image(
    pipe, 
    hed, 
    input_image: Image.Image, 
    prompt: str, 
    neg_prompt: str, 
    guidance_scale: float, 
    control_scale: float, 
    device: str,
    seed: int = 42
) -> (Image.Image, Image.Image):

    condition_image = hed(
        input_image, 
        detect_resolution=512, 
        image_resolution=512
    )
    
    # generator = torch.Generator(device=device).manual_seed(seed)
    
    output_image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=condition_image,
        num_inference_steps=30,
        # generator=generator,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=control_scale
    ).images[0]
    
    return output_image, condition_image