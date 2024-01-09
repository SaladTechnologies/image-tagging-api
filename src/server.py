from model import load_model, prepare_image, tag_image
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os
from __version__ import __version__

host = os.getenv("HOST", "*")
port = int(os.getenv("PORT", "8000"))

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
    tags = tag_image(model, prepare_image(image))

    return {"tags": tags}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=host, port=port)
