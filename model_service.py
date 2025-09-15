from flask import Flask
from pydantic import BaseModel
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch

app = Flask(__name__)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
model.eval()

class Query(BaseModel):
    text: str

@app.post("/encode")
def encode(query: Query):
    inputs = tokenizer(query.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).pooler_output.cpu().numpy()[0]
    return {"embedding": embeddings.tolist()}

