import tensorflow as tf
import os
import cv2
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import glob
from mrcnn import utils, visualize
import mrcnn.model as modellib
from samples.stone import stone
from importlib import reload
from easydict import EasyDict as edict

print("TensorFlow version:", tf.__version__)

# Define the Root directory and weights path
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Get the most recent custom weights
CUSTOM_WEIGHTS_PATH = sorted(glob.glob("logs/mask_rcnn_stone_*.h5"))[-1]

print("Root Directory:", ROOT_DIR)
print("Custom Weights Path:", CUSTOM_WEIGHTS_PATH)
print("Model Directory:", MODEL_DIR)

# Define dataset directories
custom_DIR = os.path.join(ROOT_DIR, "datasets")
custom_DIRE = os.path.join(custom_DIR, "stone_dataset")
custom_DIREE = os.path.join(custom_DIRE, "stone")

print("Custom Dataset Directory:", custom_DIREE)

# Inference configuration
class InferenceConfig(stone.StoneConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
DEVICE = "/gpu:0"  # Set to '/cpu:0' if you don't have a GPU

TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes."""
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# Load validation dataset
dataset = stone.StoneDataset()
dataset.load_stone(custom_DIREE, "val")
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
print("Loading weights ", CUSTOM_WEIGHTS_PATH)
model.load_weights(CUSTOM_WEIGHTS_PATH, by_name=True)

# Reload the visualize module to apply changes
reload(visualize)

# Run detection on all images in the dataset
for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, dataset.image_reference(image_id)))

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax, title="Predictions")
    print("gt_class_id", gt_class_id)
    print("gt_bbox", gt_bbox)
    print("gt_mask", gt_mask)
