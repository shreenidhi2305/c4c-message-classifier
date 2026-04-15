from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import zipfile
import gdown
import requests
import time
import threading

app = FastAPI()

# =========================
# 🔐 ENV VARIABLES
# =========================
UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

SUPABASE_URL = os.getenv("SB_URL")
SUPABASE_KEY = os.getenv("SB_SERVICE_ROLE_KEY")

# =========================
# 📦 MODEL SETUP
# =========================
MODEL_PATH = "model"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")

    file_id = "1Arcs9XeOKkhG8l4ybQPQSAEX6uDWR3p7"
    url = f"https://drive.google.com/uc?id={file_id}"

    output = "model.zip"
    gdown.download(url, output, quiet=False)

    print("Extracting model...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(MODEL_PATH)

    print("Model ready!")

# 🔥 Load once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()
torch.set_num_threads(1)

labels = ["Collaboration", "Issue", "Question"]

# =========================
# 🧠 CLASSIFICATION FUNCTION
# =========================
def classify_text(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        padding=False
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()
    return labels[pred]

# =========================
# 🔁 WORKER LOOP
# =========================
def worker():
    print("🚀 Worker started...")

    while True:
        try:
            # 🔹 Pop job from Redis queue
            res = requests.post(
                f"{UPSTASH_URL}/rpop/jobs",
                headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"}
            )

            data = res.json()

            if not data.get("result"):
                time.sleep(2)  # avoid spamming Redis
                continue

            job = data["result"]
            job_id = job["id"]
            message = job["message"]

            print(f"Processing job: {job_id}")

            # 🧠 Run model
            category = classify_text(message)

            # 🔹 Update Supabase
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/contacts?id=eq.{job_id}",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                },
                json={"category": category},
            )

            print(f"✅ Done: {job_id} → {category}")

        except Exception as e:
            print("❌ Worker error:", str(e))
            time.sleep(5)

# =========================
# 🚀 START WORKER THREAD
# =========================
@app.on_event("startup")
def start_worker():
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

# =========================
# 🌐 API (OPTIONAL)
# =========================
class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/classify")
def classify(input: InputText):
    category = classify_text(input.text)
    return {"category": category}

# =========================
# 🌍 CORS
# =========================
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)