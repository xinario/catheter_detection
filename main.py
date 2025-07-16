# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from inference import segment_xray
import io

app = FastAPI()

@app.post("/segment/")
async def segment(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result_img = segment_xray(image_bytes)

    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")