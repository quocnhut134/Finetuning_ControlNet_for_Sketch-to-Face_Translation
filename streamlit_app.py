import streamlit as st
from PIL import Image
from src import config 
from src.app import loader, inference

st.set_page_config(layout="wide")
st.title("Demo Generating Image from Face Sketch ")

st.markdown("---")

with st.spinner("Loading models..."):
    pipe = loader.load_pipeline()
    hed = loader.load_hed_detector()

with st.sidebar:
    st.header("Configuration")
    prompt = st.text_area(
        "Positive Prompt",
        "(hyper-realistic photo:1.2), (ultra-detailed skin texture:1.1), detailed pores, realistic eyes, (Caucasian man:1.1), sharp focus, 8k UHD, professional studio lighting",
        height=100
    )
    negative_prompt = st.text_area(
        "Negative Prompt",
        "(drawing:1.4), (sketch:1.4), (painting:1.3), cartoon, 3D, render, CGI, anime, illustration, (deformed:1.2), (disfigured:1.2), ugly, bad anatomy, (blurry:1.1)",
        height=100
    )
    guidance_scale = st.slider("Guidance Scale", 1.0, 15.0, 8.0, 0.5)
    control_scale = st.slider("ControlNet Scale", 0.0, 1.0, 0.9, 0.1)
    uploaded_file = st.file_uploader("Upload your face sketch here...", type=["png", "jpg", "jpeg"])
    run_button = st.button("Generate Image", width='stretch', type="primary")
    
col1, col2 = st.columns(2)

with col1:
    st.header("1. Uploaded Face Sketch")
    st.markdown("Input Face Sketch")
    input_image_placeholder = st.empty()
    st.markdown("HED Image")
    hed_image_placeholder = st.empty()

with col2:
    st.header("2. Generated Image")
    
    st.markdown("Output Generated Image")
    output_image_placeholder = st.empty()

    if run_button and uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        input_image_placeholder.image(input_image, caption="Uploaded sketch", width='stretch')

        with st.spinner("Generating image..."):
            output_image, condition_image = inference.generate_image(
                pipe=pipe,
                hed=hed,
                input_image=input_image,
                prompt=prompt,
                neg_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                control_scale=control_scale,
                device=config.app_device
            )

        hed_image_placeholder.image(condition_image, caption="HED Image", width='stretch')
        output_image_placeholder.image(output_image, caption="Generated Image", width='stretch')

    elif uploaded_file:
        input_image_placeholder.info("Click the generate image button on sidebar")