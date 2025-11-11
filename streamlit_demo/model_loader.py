import streamlit as st
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import HEDdetector
import config 

@st.cache_resource
def load_hed_detector():
    hed = HEDdetector.from_pretrained(config.hed_model)
    hed = hed.to(config.device)
    return hed

@st.cache_resource
def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        config.controlnet_path, 
        torch_dtype=config.dtype
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.base_model, 
        controlnet=controlnet, 
        torch_dtype=config.dtype,
        safety_checker=None 
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # pipe.enable_xformers_memory_efficient_attention()
        
    pipe = pipe.to(config.device)
    return pipe