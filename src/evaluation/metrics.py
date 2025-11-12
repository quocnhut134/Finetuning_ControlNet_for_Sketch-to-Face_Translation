import torch
import os
import lpips
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from torch_fidelity import calculate_metrics
from src import config

def load_evaluation_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        config.train_best_model_path, 
        torch_dtype=config.dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.train_stable_diff_name, 
        controlnet=controlnet, 
        torch_dtype=config.dtype
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(config.device)
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.device)
    return pipe, loss_fn_vgg

def run_lpips_generation(pipe, loss_fn_vgg):
    real_dir = os.path.join(config.eval_test_dir, "test", "photos")
    sketch_dir = os.path.join(config.eval_test_dir, "test", "sketches")
    os.makedirs(config.eval_generated_dir, exist_ok=True)
    
    generator = torch.Generator(device=config.device).manual_seed(1234)

    lpips_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    total_lpips_distance = 0
    image_count = 0
    sketch_files = sorted(os.listdir(sketch_dir))

    for i, filename in enumerate(tqdm(sketch_files, desc="LPIPS Generation")):
        if config.eval_max_samples and i >= config.eval_max_samples:
            break

        sketch_path = os.path.join(sketch_dir, filename)
        real_path = os.path.join(real_dir, filename)
        generated_path = os.path.join(config.eval_generated_dir, filename)

        condition_image = load_image(sketch_path).resize((512, 512))

        generated_image_pil = pipe(
            prompt=config.eval_prompt,
            negative_prompt=config.eval_negative_prompt,
            image=condition_image, num_inference_steps=30,
            generator=generator, guidance_scale=7.5,
            controlnet_conditioning_scale=0.9
        ).images[0]
        
        generated_image_pil.save(generated_path)
        real_image_pil = load_image(real_path)
        real_tensor = lpips_transform(real_image_pil).to(config.device)
        gen_tensor = lpips_transform(generated_image_pil).to(config.device)

        with torch.no_grad():
            dist = loss_fn_vgg(real_tensor.unsqueeze(0), gen_tensor.unsqueeze(0))
        
        total_lpips_distance += dist.item()
        image_count += 1
    
    avg_lpips = total_lpips_distance / image_count
    print(f"Average LPIPS: {avg_lpips:.4f}")
    return image_count

def run_fid_kid(image_count):
    real_dir = os.path.join(config.eval_test_dir, "test", "photos")

    kid_subset_size = min(image_count, 1000)
    if image_count < kid_subset_size:
        kid_subset_size = image_count

    metrics = calculate_metrics(
        input1=real_dir,
        input2=config.eval_generated_dir,
        cuda=True, fid=True, kid=True,
        input1_max_samples=image_count,
        input2_max_samples=image_count,
        kid_subset_size=kid_subset_size 
    )
    
    print(f"FID: {metrics['frechet_inception_distance']:.4f}")
    print(f"KID Mean: {metrics['kernel_inception_distance_mean']:.4f}")
    print(f"KID Std: {metrics['kernel_inception_distance_std']:.4f}")