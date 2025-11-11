import torch
controlnet_path = "../saved_models/controlnet_best_model" 
base_model = "botp/stable-diffusion-v1-5"
hed_model = 'lllyasviel/Annotators'
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32