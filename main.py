from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import google.generativeai as genai
import os, base64, json
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

model = genai.GenerativeModel("gemini-2.0-flash")

PROMPT = """You are a tyre safety expert. Analyse this tyre image for ALL visible problems.
Respond ONLY in this exact JSON format, no markdown:
{
  "problems": ["list", "of", "detected", "problems"],
  "tread_depth_mm": <number or null>,
  "status": "Safe|Warning|Danger",
  "confidence": "High|Medium|Low",
  "summary": "<one sentence overall assessment>",
  "recommendation": "<immediate action to take>"
}

Check for: tread wear, sidewall cracks, bulges, cuts, punctures, uneven wear, bald spots.
Tread depth: Safe >3mm, Warning 1.6-3mm, Danger <1.6mm.
If no tyre visible, set status to Warning and explain in summary."""


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize if too large
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))

        response = model.generate_content(
            [PROMPT, image],
            generation_config={"temperature": 0.1}
        )

        text = response.text.strip()
        # Strip markdown if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        result = json.loads(text)
        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI response parse error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
