import streamlit as st
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import HEDdetector
from src import config 

@st.cache_resource
def load_hed_detector():
    hed = HEDdetector.from_pretrained(config.app_hed_model)
    hed = hed.to(config.app_device)
    return hed

@st.cache_resource
def load_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        config.app_controlnet_path, 
        torch_dtype=config.app_dtype
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.app_base_model, 
        controlnet=controlnet, 
        torch_dtype=config.app_dtype,
        safety_checker=None
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(config.app_device)
    return pipe