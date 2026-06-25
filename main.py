from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLOWorld
from PIL import Image
import io, numpy as np, cv2, base64, json, os, httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# YOLO-World: zero-shot detection with text prompts — no training needed
yolo = YOLOWorld("yolov8s-world.pt")
yolo.set_classes([
    "tyre tread wear",
    "bald tyre",
    "tyre crack",
    "sidewall bulge",
    "tyre puncture",
    "tyre cut",
    "uneven tread wear",
    "tyre"
])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

PROMPT = """You are a tyre safety expert. Analyse this tyre image.
Respond ONLY in this exact JSON, no markdown:
{"tread_depth_mm": <number or null>, "status": "Safe|Warning|Danger", "confidence": "High|Medium|Low", "summary": "<one sentence>", "recommendation": "<action>"}
Depth guide: Safe >3mm, Warning 1.6-3mm, Danger <1.6mm."""


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))

        img_np = np.array(image)

        # --- YOLO-World detection ---
        results = yolo(img_np, conf=0.25, verbose=False)[0]
        detections = []
        annotated = img_np.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = yolo.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({"label": label, "confidence": round(conf, 2), "bbox": [x1, y1, x2, y2]})
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (230, 57, 70), 2)
            cv2.putText(annotated, f"{label} {conf:.0%}", (x1, max(y1-8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 57, 70), 2)

        _, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        img_b64 = base64.b64encode(buf).decode()

        # --- Groq Vision analysis ---
        groq_result = {"tread_depth_mm": None, "status": "Warning",
                       "confidence": "Low", "summary": "Analysis unavailable.",
                       "recommendation": "Manual inspection recommended."}

        if GROQ_API_KEY:
            raw_b64 = base64.b64encode(contents).decode()
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    GROQ_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.2-90b-vision-preview",
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{raw_b64}"}}
                        ]}],
                        "temperature": 0.1, "max_tokens": 300
                    }
                )
            data = resp.json()
            if "choices" in data:
                text = data["choices"][0]["message"]["content"].strip()
                start, end = text.find("{"), text.rfind("}") + 1
                if start != -1:
                    groq_result = json.loads(text[start:end])

        return {
            "detections": detections,
            "count": len(detections),
            "annotated_image": img_b64,
            **groq_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
