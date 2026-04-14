from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

import os
import zipfile
import gdown

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

# Load model (runs once)
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model")

labels = ["Collaboration", "Issue", "Question"]

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/classify")
def classify(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    return {
        "category": labels[pred]
    }

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict
    allow_methods=["*"],
    allow_headers=["*"],
)
