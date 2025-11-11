from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import clip
from PIL import Image
import io
from pathlib import Path
import mimetypes
import hashlib
import re
import html
from typing import Optional
from pydantic import BaseModel, validator
from utils import get_wiki_data
from prometheus_fastapi_instrumentator import Instrumentator

# ---------------- Load Model ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
categories = ["a stone with inscription", "a plain stone", "a palm leaf manuscript"]
text = clip.tokenize(categories).to(device)

# ---------------- FastAPI App ----------------
app = FastAPI(title="Stone Inscription Classifier API")

# Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Configuration ----------------
ALLOWED_EXTENSIONS = {".png", ".jpeg", ".jpg", ".webp"}
ALLOWED_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp"
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_FILE_SIZE = 100  # 100 bytes minimum

MAGIC_BYTES = {
    'png': b'\x89PNG\r\n\x1a\n',
    'jpeg': b'\xff\xd8\xff',
    'webp': b'RIFF',
}

# ---------------- Sanitization Functions ----------------
def sanitize_filename(filename: str) -> str:
    if not filename:
        return "unknown_file"
    
    filename = filename.replace('/', '').replace('\\', '').replace('\x00', '')
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    filename = filename.lstrip('.')
    filename = filename[:200]
    
    file_hash = hashlib.sha256(filename.encode()).hexdigest()[:8]
    return f"{file_hash}_{filename}"

def sanitize_text_output(text: str) -> str:
    sanitized = html.escape(str(text))
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
    return sanitized

def sanitize_wiki_title(title: str) -> str:
    """Sanitize Wikipedia title input."""
    if not title:
        raise ValueError("Title cannot be empty")
    
    # Remove null bytes and control characters
    title = title.replace('\x00', '').strip()
    
    # Limit length to reasonable Wikipedia title length (e.g., 200 chars)
    if len(title) > 200:
        title = title[:200]
    
    # Replace disallowed characters with underscores (Wikipedia titles allow letters, numbers, spaces, some punctuation)
    # But for safety, restrict to alphanumeric, spaces, and basic punctuation
    title = re.sub(r'[<>:"/\\|?*]', '_', title)
    
    # Remove leading/trailing spaces
    title = title.strip()
    
    # Check for malicious patterns
    forbidden_patterns = [
        r'<script', r'javascript:', r'eval\(', r'exec\(', r'system\(',
        r'base64_decode', r'passthru\(', r'shell_exec\('
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            raise ValueError("Invalid characters or patterns detected in title")
    
    return title

# ---------------- Validation Functions ----------------
def validate_file_extension(filename: str) -> bool:
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS

def validate_file_size(contents: bytes) -> tuple[bool, Optional[str]]:
    size = len(contents)
    if size < MIN_FILE_SIZE:
        return False, "File too small. Minimum size is 100 bytes."
    if size > MAX_FILE_SIZE:
        return False, "File too large. Maximum size is 10MB."
    return True, None

def validate_magic_bytes(contents: bytes, filename: str) -> bool:
    ext = Path(filename).suffix.lower().lstrip('.')
    if ext == 'png' and contents.startswith(MAGIC_BYTES['png']):
        return True
    if ext in ['jpg', 'jpeg'] and contents.startswith(MAGIC_BYTES['jpeg']):
        return True
    if ext == 'webp' and contents.startswith(MAGIC_BYTES['webp']):
        return True
    return False

def validate_image_integrity(contents: bytes) -> tuple[bool, Optional[str]]:
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
        img = Image.open(io.BytesIO(contents))
        width, height = img.size
        if width > 10000 or height > 10000:
            return False, "Image dimensions too large. Maximum 10000x10000 pixels."
        img.convert("RGB")
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def scan_for_malicious_content(contents: bytes) -> bool:
    forbidden_patterns = [
        b'<?php', b'<%', b'<script', b'javascript:', b'eval(',
        b'exec(', b'system(', b'base64_decode', b'passthru(',
        b'shell_exec('
    ]
    check_sections = [contents[:2048], contents[-2048:]]
    for section in check_sections:
        for pattern in forbidden_patterns:
            if pattern in section:
                return False
    return True

# ---------------- Comprehensive Validation ----------------
def validate_uploaded_file(contents: bytes, filename: str) -> tuple[bool, Optional[str]]:
    is_valid, error = validate_file_size(contents)
    if not is_valid:
        return False, error
    
    if not validate_file_extension(filename):
        return False, "Invalid file extension. Allowed: png, jpeg, jpg, webp"
    
    if not validate_magic_bytes(contents, filename):
        return False, "File type mismatch. The file content doesn't match its extension."
    
    if not scan_for_malicious_content(contents):
        return False, "Suspicious content detected in file."
    
    is_valid, error = validate_image_integrity(contents)
    if not is_valid:
        return False, error
    
    return True, None

# ---------------- Inference Function ----------------
def check_inscription(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity[0].cpu().numpy()

    fine_label = categories[probs.argmax()]
    confidence = float(probs.max())

    if fine_label == "a stone with inscription":
        binary_label = "Stone Inscription"
    else:
        binary_label = "Not a Stone Inscription"

    return fine_label, binary_label, confidence

# Request model for Wikipedia fetch
class WikiRequest(BaseModel):
    title: str

    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        if len(v.strip()) > 200:
            raise ValueError('Title too long. Maximum 200 characters')
        return v.strip()

# ---------------- API Endpoints ----------------
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Stone Inscription Classifier API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        original_filename = file.filename or "unknown"
        safe_filename = sanitize_filename(original_filename)
        contents = await file.read()
        
        is_valid, error_message = validate_uploaded_file(contents, original_filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        fine_label, binary_label, confidence = check_inscription(image)
        
        return {
            "result": sanitize_text_output(binary_label),
            "confidence": round(confidence, 4),
            "internal_label": sanitize_text_output(fine_label),
            "filename": sanitize_text_output(safe_filename)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Failed to process image file. Please ensure you're uploading a valid image."
        )

@app.post("/fetch_wiki")
def fetch_wiki(request: WikiRequest):
    """Fetch Wikipedia data for a given title."""
    try:
        # Additional sanitization
        sanitized_title = sanitize_wiki_title(request.title)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    result = get_wiki_data(sanitized_title)
    if result["status"] == "success":
        # Sanitize output if needed
        data = result["data"]
        data['title'] = sanitize_text_output(data['title'])
        data['summary'] = sanitize_text_output(data['summary'])
        return data
    else:
        raise HTTPException(status_code=404, detail=result["message"])