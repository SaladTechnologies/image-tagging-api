import time

start = time.perf_counter()
from model import load_model, prepare_image, tag_image
from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
import io
import os
import json
from __version__ import __version__

host = os.getenv("HOST", "*")
port = int(os.getenv("PORT", "8000"))
salad_machine_id = os.getenv("SALAD_MACHINE_ID", "")
salad_container_group_id = os.getenv("SALAD_CONTAINER_GROUP_ID", "")

model = load_model()

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
    headers = {
        "X-Salad-Machine-ID": salad_machine_id,
        "X-Salad-Container-Group-ID": salad_container_group_id,
        "X-Inference-Time": f"{stop-start:.4f}",
    }
    return Response(content=json.dumps({"tags": tags}), headers=headers)


stop = time.perf_counter()
print(f"Server Ready in {stop-start:.4f} seconds", flush=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port)
