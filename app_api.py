from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import clip
from PIL import Image
import io
import logging

# Import ClamAV scanner utility
from clamav_scanner import (
    ClamAVScanner, 
    create_scanner, 
    scan_uploaded_file,
    ScanStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Load Model ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
categories = ["a stone with inscription", "a plain stone", "a palm leaf manuscript"]
text = clip.tokenize(categories).to(device)

# ---------------- Initialize ClamAV Scanner ----------------
# Create scanner instance (adjust connection_type as needed: "unix" or "tcp")
clamav_scanner = create_scanner(connection_type="unix")

# Check if ClamAV is available at startup
CLAMAV_ENABLED = clamav_scanner.is_available()
if CLAMAV_ENABLED:
    logger.info("ClamAV scanner initialized successfully")
    logger.info(f"ClamAV version: {clamav_scanner.get_version()}")
else:
    logger.warning("ClamAV scanner not available - virus scanning disabled")

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
    return {
        "status": "ok", 
        "message": "Stone Inscription Classifier API is running",
        "clamav_enabled": CLAMAV_ENABLED
    }

@app.get("/health")
async def detailed_health_check():
    """Detailed health check including ClamAV status"""
    health_info = {
        "api_status": "healthy",
        "model_device": device,
        "clamav_enabled": CLAMAV_ENABLED
    }
    
    if CLAMAV_ENABLED:
        health_info["clamav_version"] = clamav_scanner.get_version()
        health_info["clamav_status"] = "connected"
    else:
        health_info["clamav_status"] = "disconnected"
    
    return health_info

@app.post("/predict/")
async def predict(file: UploadFile = File(...), skip_virus_scan: bool = False):
    """
    Predict if uploaded image contains stone inscription
    
    Args:
        file: Image file to classify
        skip_virus_scan: Set to True to skip virus scanning (not recommended)
    
    Returns:
        Classification result with confidence score
    """
    try:
        # Read file content
        contents = await file.read()
        
        # Virus scan (if enabled and not skipped)
        if CLAMAV_ENABLED and not skip_virus_scan:
            logger.info(f"Scanning file for viruses: {file.filename}")
            scan_result = await scan_uploaded_file(
                scanner=clamav_scanner,
                file_content=contents,
                filename=file.filename,
                cleanup=True
            )
            
            # Check if file is infected
            if scan_result.status == ScanStatus.INFECTED:
                logger.warning(f"Infected file detected: {file.filename} - {scan_result.threat_name}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "File infected with malware",
                        "threat": scan_result.threat_name,
                        "message": scan_result.message
                    }
                )
            
            # Log scan errors but continue (optional: you can make this stricter)
            if scan_result.status == ScanStatus.ERROR:
                logger.error(f"Virus scan error for {file.filename}: {scan_result.message}")
                # Optionally raise exception here to block processing on scan errors
            
            logger.info(f"File clean: {file.filename}")
        
        # Process image for classification
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        fine_label, binary_label, confidence = check_inscription(image)

        response = {
            "result": binary_label,
            "confidence": round(confidence, 4),
            "internal_label": fine_label
        }
        
        # Include virus scan info if performed
        if CLAMAV_ENABLED and not skip_virus_scan:
            response["security"] = {
                "virus_scan_performed": True,
                "scan_status": scan_result.status.value
            }
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/scan-only/")
async def scan_file_only(file: UploadFile = File(...)):
    """
    Endpoint to only scan file for viruses without classification
    Useful for testing virus scanning functionality
    
    Args:
        file: File to scan
        
    Returns:
        Virus scan results
    """
    if not CLAMAV_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="ClamAV service not available"
        )
    
    try:
        contents = await file.read()
        
        scan_result = await scan_uploaded_file(
            scanner=clamav_scanner,
            file_content=contents,
            filename=file.filename,
            cleanup=True
        )
        
        return {
            "filename": file.filename,
            "file_size": len(contents),
            "scan_result": scan_result.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Scan failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))