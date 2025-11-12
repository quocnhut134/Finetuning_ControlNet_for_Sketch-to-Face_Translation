import cv2
import random
from src import config
from src.data.augmentation import augment_sketch

def load_hed_net():
    net = cv2.dnn.readNetFromCaffe(config.hed_caffe_prototxt, config.hed_caffe_model)
    return net

def apply_hed_caffe(image, net, target_size=512, apply_augmentation=False):
    (h, w) = image.shape[:2]
    mean_pixel_values = (104.00698793, 116.66876762, 122.67891434)
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(w, h), 
                                 mean=mean_pixel_values, swapRB=False, crop=False)
    
    net.setInput(blob)
    hed_output = net.forward()
    hed_output = hed_output[0, 0] 
    hed_output = cv2.resize(hed_output, (w, h))
    hed_output = cv2.normalize(hed_output, None, 0, 255, cv2.NORM_MINMAX)
    hed_output = hed_output.astype("uint8")
    hed_output = 255 - hed_output
    hed_output_rgb = cv2.cvtColor(hed_output, cv2.COLOR_GRAY2BGR)
    
    if apply_augmentation and random.random() < config.augment_prob:
        hed_output_rgb = augment_sketch(hed_output_rgb)
    
    return hed_output_rgb