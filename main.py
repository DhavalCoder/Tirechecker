from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import io, numpy as np, cv2, base64, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLOv8 model — uses pretrained weights on first run, downloads automatically
# yolov8n.pt = nano (fast), works for tyre detection with general objects
# Replace with custom tyre model path when available
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
model = YOLO(MODEL_PATH)

# Tyre-related class names to highlight (COCO pretrained)
TYRE_CLASSES = {"car", "truck", "bus", "motorcycle"}  # vehicles imply tyres

# If using custom tyre defect model, these would be:
# DEFECT_CLASSES = {"tread_wear", "sidewall_crack", "bulge", "puncture", "bald"}


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)

        # Run YOLO inference
        results = model(img_np, conf=0.3, verbose=False)[0]

        detections = []
        annotated = img_np.copy()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

            # Draw bounding box
            color = (230, 57, 70)  # red
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {conf:.0%}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Encode annotated image to base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        img_b64 = base64.b64encode(buffer).decode()

        # Simple tyre condition assessment based on detections
        status = "Safe"
        summary = "No issues detected."
        if not detections:
            summary = "No objects detected. Ensure tyre fills the frame."
            status = "Warning"

        return {
            "detections": detections,
            "count": len(detections),
            "status": status,
            "summary": summary,
            "annotated_image": img_b64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
