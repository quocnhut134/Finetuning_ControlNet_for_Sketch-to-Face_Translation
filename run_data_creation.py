import os
import cv2
import glob
import random
from tqdm import tqdm
from src import config
from src.data.hed_caffe import load_hed_net, apply_hed_caffe

def main():
    net = load_hed_net()
    
    all_image_paths = sorted(glob.glob(f"{config.data_creation_ffhq_dir}/**/*.png", recursive=True))
    print(f"Total images:{len(all_image_paths)}")
    
    random.seed(42) 
    random.shuffle(all_image_paths)
    
    num_total = len(all_image_paths)
    num_train = int(num_total * config.data_train_ratio)
    num_eval = int(num_total * config.data_eval_ratio)

    splits = {
        "train": all_image_paths[:num_train],
        "val": all_image_paths[num_train : num_train + num_eval],
        "test": all_image_paths[num_train + num_eval:],
    }

    for split_name, file_list in splits.items():    
        target_dir = os.path.join(config.data_creation_output_dir, split_name, "photos")
        source_dir = os.path.join(config.data_creation_output_dir, split_name, "sketches")
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(source_dir, exist_ok=True)
        
        should_augment = (split_name == "train") 
        
        for img_path in tqdm(file_list, desc=f"Processing {split_name}"):
            photo_original = cv2.imread(img_path)
            if photo_original is None:
                continue
            
            photo_resized = cv2.resize(photo_original, (512, 512), interpolation=cv2.INTER_AREA)
            
            sketch_hed = apply_hed_caffe(
                photo_resized, 
                net, 
                target_size=512, 
                apply_augmentation=should_augment
            )
            
            filename = os.path.basename(img_path)
            target_save_path = os.path.join(target_dir, filename)
            source_save_path = os.path.join(source_dir, filename)
            
            cv2.imwrite(target_save_path, photo_resized)
            cv2.imwrite(source_save_path, sketch_hed)
        
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__))
    main()