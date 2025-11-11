
import io, os, time, random
import numpy as np
from PIL import Image
import streamlit as st
import torch, cv2

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

APP_TITLE = "Sketch → Face Demo (Stable Diffusion v1.5 + ControlNet Canny)"

@st.cache_resource(show_spinner=False)
def load_pipeline(model_id="runwayml/stable-diffusion-v1-5",
                  controlnet_id="lllyasviel/sd-controlnet-canny"):
    """
    Load SD1.5 + ControlNet(Canny). If GPU is available, use float16 for speed/memory.
    """
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load ControlNet backbone first
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)

    # Compose ControlNet pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,        # demo/research only
    )

    if torch.cuda.is_available():
        pipe.to("cuda")
        pipe.enable_attention_slicing()
        # pipe.enable_model_cpu_offload()  # alternative if VRAM is limited
    return pipe

def to_edges(img_pil, low=80, high=160, size=512):
    """
    Normalize uploaded sketch into an edge map for ControlNet.
    If the input is already a sketch, Canny will standardize line thickness.
    """
    img = img_pil.convert("RGB").resize((size, size))
    np_img = np.array(img)
    # auto-threshold safety
    low = int(max(0, min(255, low)))
    high = int(max(0, min(255, high)))
    if low >= high:
        high = min(255, low + 1)

    edges = cv2.Canny(np_img, low, high)
    edges = np.stack([edges]*3, axis=-1)
    return Image.fromarray(edges)

def infer(pipe, cond_img, prompt, negative_prompt, steps, guidance, cond_scale, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

    t0 = time.time()
    out = pipe(
        prompt=prompt,
        image=cond_img,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=cond_scale,
        generator=generator
    ).images[0]
    dt = time.time() - t0
    return out, dt

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.markdown("**How this works**")
        st.markdown(
            "- Uses a pretrained [ControlNet (Canny)](https://huggingface.co/lllyasviel/sd-controlnet-canny)\n"
            "- Converts your sketch into an edge map with Canny\n"
            "- Feeds the edge map as a condition into SD 1.5 to generate a realistic portrait"
        )
        st.info("Tip: If first run is slow, it's downloading ~4–5GB of model weights.")

    pipe = load_pipeline()

    left, right = st.columns([1, 1])
    with left:
        st.subheader("1) Upload sketch")
        file = st.file_uploader("PNG/JPG", type=["png", "jpg", "jpeg"])
        example = st.checkbox("Use example sketch", value=not file)
        low = st.slider("Canny low", 0, 255, 80, 1)
        high = st.slider("Canny high", 1, 255, 160, 1)
        size = st.select_slider("Working size", options=[384, 448, 512, 640], value=512)

    with right:
        st.subheader("2) Generation options")
        prompt = st.text_area("Prompt", "a realistic portrait photo, detailed face, soft natural lighting, high quality", height=60)
        neg = st.text_area("Negative prompt", "lowres, blurry, artifact, cartoon, anime", height=60)
        steps = st.slider("Inference steps", 10, 50, 25, 1)
        guidance = st.slider("Guidance scale", 1.0, 15.0, 9.0, 0.5)
        cond_scale = st.slider("Control strength", 0.0, 2.0, 1.0, 0.05)
        seed = st.text_input("Seed (optional, leave blank for random)", "")
        btn = st.button("Generate")

    # Input image
    if example and not file:
        sketch = Image.open(os.path.join("samples", "dummy_sketch.png"))
    elif file:
        sketch = Image.open(file)
    else:
        sketch = None

    if sketch is not None:
        cond_img = to_edges(sketch, low=low, high=high, size=size)
        colA, colB = st.columns(2)
        with colA:
            st.image(sketch, caption="Input sketch", use_container_width=True)
        with colB:
            st.image(cond_img, caption="Edge map (Canny)", use_container_width=True)
    else:
        st.warning("Please upload a sketch or tick 'Use example sketch'.")
        return

    if btn:
        s_val = None if seed.strip()=="" else int(seed.strip())
        with st.spinner("Generating…"):
            result, secs = infer(pipe, cond_img, prompt, neg, steps, guidance, cond_scale, seed=s_val)
        st.success(f"Done in {secs:.1f}s")
        st.image(result, caption="Generated image", use_container_width=True)
        buf = io.BytesIO(); result.save(buf, format="PNG")
        st.download_button("Download PNG", data=buf.getvalue(), file_name="result.png", mime="image/png")

if __name__ == "__main__":
    main()
