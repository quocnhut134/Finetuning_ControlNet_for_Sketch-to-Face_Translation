import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from src import config

def setup_model_and_optimizer():
    controlnet = ControlNetModel.from_pretrained(
        config.train_controlnet_name
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.train_stable_diff_name, 
        controlnet=controlnet, 
        dtype=config.dtype
    )
    
    # Freeze layers
    for param in pipe.unet.parameters(): param.requires_grad = False
    for param in pipe.text_encoder.parameters(): param.requires_grad = False
    for param in pipe.vae.parameters(): param.requires_grad = False

    controlnet.to(torch.float32) 
    for param in controlnet.parameters():
        param.requires_grad = True

    pipe.to(config.device) 
    controlnet.to(config.device) 
    
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=config.learning_rate, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    return pipe, controlnet, optimizer, scheduler