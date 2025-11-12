import torch
from tqdm import tqdm
from torch.amp import autocast 
from src import config

def train_epoch(controlnet, pipe, train_dataloader, optimizer):
    controlnet.train()
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training Epoch")
    
    for step, batch in enumerate(progress_bar):
        hed_images = batch["hed"].to(config.device)
        photos = batch["photo"].to(config.device)
        
        with autocast(config.device, enabled=(config.device == 'cuda')): 
            with torch.no_grad():
                latents = pipe.vae.encode(photos).latent_dist.sample() * pipe.vae.config.scaling_factor
            
            bsz = latents.shape[0]
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=config.device).long()
            noise = torch.randn_like(latents)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            use_null_prompt = torch.rand(1).item() < 0.1
            final_prompt = "" if use_null_prompt else config.prompt
                
            text_inputs = pipe.tokenizer(
                final_prompt, 
                padding=config.padding, 
                max_length=pipe.tokenizer.model_max_length, 
                truncation=True, 
                return_tensors=config.return_tensors
            )
            text_input_ids = text_inputs.input_ids.to(config.device)
            
            with torch.no_grad():
                encoder_hidden_states = pipe.text_encoder(text_input_ids)[0]
                if encoder_hidden_states.shape[0] != bsz:
                    encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)

            controlnet_output = controlnet(
                sample=noisy_latents, timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=hed_images, return_dict=True 
            )
            
            noise_pred = pipe.unet(
                noisy_latents, timestep=timesteps, 
                encoder_hidden_states=encoder_hidden_states, 
                down_block_additional_residuals=controlnet_output.down_block_res_samples, 
                mid_block_additional_residual=controlnet_output.mid_block_res_sample
            ).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            epoch_loss += loss.item()
            loss = loss / config.accumulation_steps
        
        config.scaler.scale(loss).backward()
        
        if (step + 1) % config.accumulation_steps == 0:
            config.scaler.step(optimizer)
            config.scaler.update() 
            optimizer.zero_grad()
        
        progress_bar.set_postfix(Loss=f"{loss.item() * config.accumulation_steps:.4f}")
    
    return epoch_loss / len(train_dataloader)

def validate_epoch(controlnet, pipe, val_dataloader):
    controlnet.eval()
    val_loss = 0
    val_progress_bar = tqdm(val_dataloader, desc="Validation Epoch")
    
    with torch.no_grad():
        for batch in val_progress_bar:
            hed_images = batch["hed"].to(config.device)
            photos = batch["photo"].to(config.device)
            
            with autocast(config.device, enabled=(config.device == 'cuda')): 
                latents = pipe.vae.encode(photos).latent_dist.sample() * pipe.vae.config.scaling_factor
                bsz = latents.shape[0]
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=config.device).long()
                noise = torch.randn_like(latents)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                text_inputs = pipe.tokenizer(
                    config.prompt, padding=config.padding, 
                    max_length=pipe.tokenizer.model_max_length, 
                    truncation=True, return_tensors=config.return_tensors
                )
                text_input_ids = text_inputs.input_ids.to(config.device)
                encoder_hidden_states = pipe.text_encoder(text_input_ids)[0]
                
                if encoder_hidden_states.shape[0] != bsz:
                    encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)
                
                controlnet_output = controlnet(
                    sample=noisy_latents, timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=hed_images, return_dict=True
                )
                
                noise_pred = pipe.unet(
                    noisy_latents, timestep=timesteps, 
                    encoder_hidden_states=encoder_hidden_states, 
                    down_block_additional_residuals=controlnet_output.down_block_res_samples, 
                    mid_block_additional_residual=controlnet_output.mid_block_res_sample
                ).sample
                
                val_loss += torch.nn.functional.mse_loss(noise_pred, noise).item()
            
            val_progress_bar.set_postfix(Val_Loss=f"{val_loss / len(val_dataloader):.4f}")
            
    return val_loss / len(val_dataloader)