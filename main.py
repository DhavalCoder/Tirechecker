from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os, base64, json, httpx
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

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

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.2-90b-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                            ]
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            )

        data = response.json()

        # Debug: return raw if unexpected format
        if "choices" not in data:
            raise HTTPException(status_code=500, detail=f"Groq error: {json.dumps(data)}")

        text = data["choices"][0]["message"]["content"].strip()

        # Strip markdown code block if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        # Extract JSON object if wrapped in extra text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {str(e)} | Raw: {text[:200]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
