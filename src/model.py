import time

print("Importing Libraries", flush=True)
start = time.perf_counter()

import torch
from PIL import Image
from ram.models import ram_plus
from ram import get_transform
from ram.utils import build_openset_llm_label_embedding
import os

import numpy as np
from PIL import Image

from torch import nn
import json


image_size = int(os.getenv("IMAGE_SIZE", "384"))

my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, "data")
tag_info_file = os.path.join(data_path, "openimages_rare_200_llm_tag_descriptions.json")
sample_image_file = os.path.join(data_path, "101-Copenhagen-NORR11-4.webp")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = get_transform(image_size=image_size)

stop = time.perf_counter()
print(f"Imported Libraries in {stop-start:.4f} seconds", flush=True)


def prepare_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    return image


sample_image = prepare_image(Image.open(sample_image_file))


def load_model():
    print("Loading model", flush=True)
    start = time.perf_counter()
    model = ram_plus(
        pretrained=os.path.join(data_path, "ram_plus_swin_large_14m.pth"),
        image_size=image_size,
        vit="swin_l",
    )

    with open(tag_info_file) as f:
        tag_info = json.load(f)
    openset_label_embedding, openset_categories = build_openset_llm_label_embedding(
        tag_info
    )

    model.tag_list = np.array(openset_categories)

    model.label_embed = nn.Parameter(openset_label_embedding.float())

    model.num_class = len(openset_categories)
    # the threshold for unseen categories is often lower
    model.class_threshold = torch.ones(model.num_class) * 0.5

    model.eval()

    model = model.to(device)
    stop = time.perf_counter()
    print(f"Loaded model in {stop - start:0.4f} seconds", flush=True)
    print("Warming up Model", flush=True)
    start = time.perf_counter()

    res = tag_image(model, sample_image)
    print("Image Tags:", res, flush=True)
    stop = time.perf_counter()
    print(f"Warmed up Model in {stop - start:0.4f} seconds", flush=True)
    return model


def tag_image(model, image) -> list[str]:
    tags = [s.strip() for s in model.generate_tag_openset(image)[0].split("|")]
    return tags
