import cv2
import numpy as np
import random
from src import config

def augment_sketch(sketch_image):
    augmented_sketch = sketch_image.copy()

    if random.random() > config.augment_prob:
        return augmented_sketch

    if random.random() < config.erosion_prob:
        kernel = np.ones((config.erosion_size, config.erosion_size), np.uint8)
        augmented_sketch = cv2.erode(augmented_sketch, kernel, iterations=1)

    if random.random() < config.dropout_prob:
        rows, cols, _ = augmented_sketch.shape
        num_holes = random.randint(int(config.dropout_holes * 0.8), config.dropout_holes)
        
        for _ in range(num_holes):
            hole_w = random.randint(config.dropout_min_size, config.dropout_max_size) 
            hole_h = random.randint(config.dropout_min_size, config.dropout_max_size)
            x1 = random.randint(0, cols - hole_w)
            y1 = random.randint(0, rows - hole_h)
            x2 = x1 + hole_w
            y2 = y1 + hole_h
            fill_value = config.dropout_fill_value
            augmented_sketch[y1:y2, x1:x2] = (fill_value, fill_value, fill_value)
            
    return augmented_sketch