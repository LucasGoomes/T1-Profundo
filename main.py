import os
import torch
from generate_images import generate_images

os.environ["HF_TOKEN"] = "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"


prompts = ["A beautiful sunset over the city"]
num_images = 1
save_dir = "/images"

generate_images(prompts, num_images, save_dir)