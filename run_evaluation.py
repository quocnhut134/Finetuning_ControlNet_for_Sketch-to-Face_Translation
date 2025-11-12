import os
import sys
from src.evaluation.metrics import load_evaluation_pipeline, run_lpips_generation, run_fid_kid

def main():
    pipe, loss_fn_vgg = load_evaluation_pipeline()
    image_count = run_lpips_generation(pipe, loss_fn_vgg)
    run_fid_kid(image_count)

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    main()