import torch
from torch.amp import GradScaler
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data
data_creation_ffhq_dir = os.path.join(project_root, "data_dir", "ffhq")
data_creation_output_dir = os.path.join(project_root, "data_dir", "large_hed-augmented_ffhq_dataset")
hed_caffe_prototxt = os.path.join(project_root, "saved_models", "hed_model", "deploy.prototxt")
hed_caffe_model = os.path.join(project_root, "saved_models", "hed_model", "hed_pretrained_bsds.caffemodel")
data_train_ratio = 0.8  
data_eval_ratio = 0.1  

# Data Augmentation
augment_prob = 0.95
erosion_prob = 1
erosion_size = 6
dropout_prob = 1     
dropout_holes = 65536
dropout_min_size = 1
dropout_max_size = 3     
dropout_fill_value = 255 

# Models
train_controlnet_name = "lllyasviel/sd-controlnet-hed"
train_stable_diff_name = "botp/stable-diffusion-v1-5" 

# Training
train_data_dir = os.path.join(project_root, "data_dir", "large_hed-augmented_ffhq_dataset")
train_best_model_path = os.path.join(project_root, "saved_models", "controlnet_best_model")
train_latest_model_path = os.path.join(project_root, "saved_models", "controlnet_latest_model")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
num_epochs = 20
batch_size = 4 
accumulation_steps = 8
learning_rate = 5e-6
prompt = "a realistic photo of a human face"
padding = "max_length"
return_tensors = "pt"
scaler = GradScaler(device, enabled=(device == 'cuda'))
patience = 5  
best_eval_loss = float('inf')

# Evaluation
eval_test_dir = os.path.join(project_root, "data_dir", "large_hed-augmented_ffhq_dataset")
eval_generated_dir = os.path.join(project_root, "outputs", "generated_for_metrics")

eval_max_samples = 500
eval_prompt = """(hyper-realistic photo:1.2), (ultra-detailed skin texture:1.1), 
            detailed pores, realistic eyes, sharp focus, 
            8k UHD, professional studio lighting, DSLR"""
eval_negative_prompt = """(drawing:1.4), (sketch:1.4), (painting:1.3), cartoon, 3D, 
                    render, CGI, anime, illustration, (deformed:1.2), (disfigured:1.2), 
                    ugly, bad anatomy, (blurry:1.1), low quality, low-res"""

# Streamlit demo
app_controlnet_path = train_best_model_path
app_base_model = train_stable_diff_name
app_hed_model = 'lllyasviel/Annotators'
app_device = device
app_dtype = dtype