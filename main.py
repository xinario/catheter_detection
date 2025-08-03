# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from inference import segment_xray, classify_xray
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


@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    probs = classify_xray(image_bytes).detach().cpu().numpy()[0]
    class_labels = ["NGT", "ETT", "UVC", "UAC"]
    result_dict = {
        label: f"{float(prob):.2f}: {float(prob)>0.5}"
        for label, prob in zip(class_labels, probs)
    }
    return JSONResponse(content=result_dict)
