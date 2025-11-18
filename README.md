# Finetuning ControlNet for Sketch-to-Face Translation

In this project, I implement a complete, end-to-end pipeline for finetuning a ControlNet model for the sketch-to-face translation task. The core solution utilizes a frozen **Stable Diffusion v1.5** model, guided by a trainable **ControlNet adapter**.

The model is specifically trained to be **robust against imperfect, noisy, or incomplete input sketches**, simulating real-world user drawings.

## Demo Gallery

| Original Sketch (Input) | HED Sketch (ControlNet Input) | Generated Face (Output) |
| :---: | :---: | :---: |
| ![Image](https://github.com/user-attachments/assets/c2c4e8ee-81f5-4895-8ca1-6bac0d30ae16) | ![Image](https://github.com/user-attachments/assets/ee7692d0-fdd8-4fd6-a342-3ee4bbf4b8ce) | ![Image](https://github.com/user-attachments/assets/6fd80d35-a0e9-49fa-a2e8-25a42ea9a987)|

## Deployment

You can enjoy the deployment here: [Image to Sketch Translation with Finetuned ControlNet for Diffusion Model](https://huggingface.co/spaces/SaitoHoujou/Sketch-to-Face_Translation)

## Installation

**1. Install Dependencies:**

You can install needed requirements with:

```
pip install -r requirements.txt
```

**2. Prepare Data & Prerequisite Models:**
This project requires external assets. Please download them and place them in the correct directories as defined in `src/config.py`.

  * **FFHQ Dataset:** Download the FFHQ dataset (e.g., 70,000 images). The default config assumes it's in `data_dir/ffhq/`.
  * **HED Caffe Model:** Download the HED model (`deploy.prototxt` and `hed_pretrained_bsds.caffemodel`) used for *data generation*. The config assumes it's in `models/hed_model/`.

## Usage Pipeline

The project is structured around 3 main scripts in the root directory. To continue, Open `src/config.py` and verify all paths and parameters in it. 

### 1. Creating sketch data

If you have the FFHQ dataset, you can generate the augmented HED dataset.

  * **Run Script:**
    ```bash
    python run_data_creation.py
    ```
    This will read images from FFHQ, apply HED processing and augmentation, and save the paired images into `train`, `eval`, and `test` splits in your output directory.

### 2. Training

Start finetuning the ControlNet model:

  * **Run Script:**
    ```bash
    python run_training.py
    ```
    The script will log training and validation loss for each epoch. It uses **Early Stopping** based on the `patience` setting.
  * **Outputs:**
      * The best model (lowest validation loss) is saved to `outputs/controlnet_best_model/`.
      * The final epoch's model is saved to `outputs/controlnet_latest_model/`.

### 3. Evaluating

Once training is complete, you can evaluate your best model on the test set.

  * **Run Script:**
    ```bash
    python run_evaluation.py
    ```
  * **Outputs:**
    * **Average LPIPS** score.
    * **FID** and **KID Mean** scores.

## Acknowledgements

  * [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for the core library.
  * The original [ControlNet](https://github.com/lllyasviel/ControlNet) paper and implementation.

  * The [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset) by NVlabs.




