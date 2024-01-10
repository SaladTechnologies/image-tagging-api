import time

start = time.perf_counter()
from model import load_model, prepare_image, tag_image
from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
import io
import os
import json
import requests
import torch
from __version__ import __version__

host = os.getenv("HOST", "*")
port = int(os.getenv("PORT", "8000"))
salad_machine_id = os.getenv("SALAD_MACHINE_ID", "")
salad_container_group_id = os.getenv("SALAD_CONTAINER_GROUP_ID", "")

model = load_model()


def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    else:
        return "CUDA is not available"


gpu_name = get_gpu_name()

default_response_headers = {
    "X-Salad-Machine-ID": salad_machine_id,
    "X-Salad-Container-Group-ID": salad_container_group_id,
    "X-GPU-Name": gpu_name,
}


def download_image(url):
    r = requests.get(url)
    image = Image.open(io.BytesIO(r.content))
    return image


app = FastAPI()


@app.get("/hc")
async def healthcheck():
    return {"status": "ok", "version": __version__}


@app.post("/tag")
async def get_image_tags(file: UploadFile = File(...)):
    # Read image file as PIL Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    start = time.perf_counter()
    tags = tag_image(model, prepare_image(image))
    stop = time.perf_counter()
    headers = default_response_headers.copy()
    headers["X-Inference-Time"] = f"{stop-start:.4f}"
    return Response(content=json.dumps({"tags": tags}), headers=headers)


@app.get("/tag")
async def get_image_tags_from_url(url: str):
    start = time.perf_counter()
    image = download_image(url)
    image_download_time = time.perf_counter() - start
    start = time.perf_counter()
    tags = tag_image(model, prepare_image(image))
    inference_time = time.perf_counter() - start
    headers = default_response_headers.copy()
    headers["X-Inference-Time"] = f"{inference_time:.4f}"
    headers["X-Image-Download-Time"] = f"{image_download_time:.4f}"
    return Response(content=json.dumps({"tags": tags}), headers=headers)


stop = time.perf_counter()
print(f"Server Ready in {stop-start:.4f} seconds", flush=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port)
