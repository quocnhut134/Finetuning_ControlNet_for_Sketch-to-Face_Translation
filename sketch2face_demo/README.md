
# Sketch → Face Demo (Streamlit + ControlNet Canny)

This is a minimal demo to turn a face **sketch** into a **realistic portrait** using **Stable Diffusion v1.5** with **ControlNet (Canny)**.

> Research/demo only. The safety checker is disabled in this demo.

## 1) Environment

- Python 3.10+ recommended (note python 3.11 )
- GPU (NVIDIA, CUDA 12.x) recommended. CPU works but will be slow.

```bash
# 1) Create env (optional)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install PyTorch (pick the right wheel for your system)
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3) Install dependencies
pip install -r requirements.txt
```

If you get a 403 when downloading `runwayml/stable-diffusion-v1-5`, create a Hugging Face account and accept its license, then login:

```bash
pip install huggingface_hub
huggingface-cli login
```

## 2) Run

```bash
streamlit run app.py
```
Open the local URL printed by Streamlit. On first run, the model weights (~4–5GB) will download.

## 3) Usage

1. Upload a face sketch (or tick **Use example sketch**).
2. Tune **Canny** thresholds if needed.
3. Set your prompt/negative prompt.
4. Click **Generate**. Download the PNG result when done.

## 4) Troubleshooting

- **Out of memory (VRAM):**
  - Lower **Working size** to 384.
  - Lower **Inference steps**.
  - Uncomment `pipe.enable_model_cpu_offload()` in `app.py` (slower, but reduces VRAM).
- **Slow on CPU:** This is expected. A GPU is strongly recommended.
- **Results look “off”:** Try increasing or decreasing **Control strength**; adjust **Canny** thresholds.

## 5) Swap in your fine-tuned ControlNet (LoRA)

Replace the `load_pipeline` function to load your ControlNet base and then `.load_attn_procs(<path-to-lora>)`:

```python
from diffusers import ControlNetModel

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)
controlnet.load_attn_procs("/path/to/checkpoints/controlnet-lora/epoch-X")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=dtype, safety_checker=None
)
```

## License
For research/demo purposes only. You are responsible for complying with all model licenses.
