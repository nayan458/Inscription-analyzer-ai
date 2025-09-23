from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import clip
from PIL import Image
import io

# ---------------- Load Model ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
categories = ["a stone with inscription", "a plain stone", "a palm leaf manuscript"]
text = clip.tokenize(categories).to(device)

# ---------------- FastAPI App ----------------
app = FastAPI(title="Stone Inscription Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Inference Function ----------------
def check_inscription(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Similarity scores
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity[0].cpu().numpy()

    
    fine_label = categories[probs.argmax()]
    confidence = float(probs.max())

    # Map to binary label
    if fine_label == "a stone with inscription":
        binary_label = "Stone Inscription"
    else:
        binary_label = "Not a Stone Inscription"

    return fine_label, binary_label, confidence

# ---------------- API Endpoints ----------------
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stone Inscription Classifier API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        fine_label, binary_label, confidence = check_inscription(image)

        return {
            "result": binary_label,
            "confidence": round(confidence, 4),
            "internal_label": fine_label  
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
