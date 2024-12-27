from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from pathlib import Path
import os
from model import TamilInscriptionModel
from datetime import datetime
import base64
import cv2
import numpy as np

app = FastAPI(title="Tamil Stone Inscription Reader API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory structure
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "saved_model/tamil_inscription_model.joblib"
TRANSLATION_DIR = BASE_DIR / "translations"
DEBUG_DIR = BASE_DIR / "debug"
TEMP_DIR = BASE_DIR / "temp"

# Create all necessary directories
for dir_path in [MODEL_PATH.parent, TRANSLATION_DIR, DEBUG_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize model
model = TamilInscriptionModel()

def create_session_dirs(session_id: str) -> dict:
    """Create session-specific directories"""
    session_dirs = {
        'base': DEBUG_DIR / session_id,
        'preprocessed': DEBUG_DIR / session_id / 'preprocessed',
        'segmented': DEBUG_DIR / session_id / 'segmented',
        'translation': TRANSLATION_DIR / session_id
    }
    
    # Clean up existing session directories if they exist
    for dir_path in session_dirs.values():
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return session_dirs

def cleanup_old_sessions(max_age_hours: int = 24):
    """Clean up old session directories"""
    current_time = datetime.now()
    for session_dir in DEBUG_DIR.glob("*"):
        if session_dir.is_dir():
            dir_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
            age_hours = (current_time - dir_time).total_seconds() / 3600
            if age_hours > max_age_hours:
                shutil.rmtree(session_dir)

def cleanup_session(session_id: str):
    """Clean up all files for a specific session"""
    session_base = DEBUG_DIR / session_id
    session_translation = TRANSLATION_DIR / session_id
    
    if session_base.exists():
        shutil.rmtree(session_base)
    if session_translation.exists():
        shutil.rmtree(session_translation)

def image_to_base64(image_array):
    """Convert image array to base64 string"""
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/preprocess/")
async def preprocess_image(
    file: UploadFile = File(...),
    scale: float = Form(30.0),
    noise_divisor: float = Form(0.9),
    debug_mode: bool = Form(False),
    background_tasks: BackgroundTasks = None,
):
    """Preprocess the image with given parameters and return the result"""
    try:
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        session_dirs = create_session_dirs(session_id)
        
        # Save uploaded file
        file_path = session_dirs['base'] / f"input_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Enable debug mode if requested
        if debug_mode:
            model.set_debug_mode(True, session_dirs['base'])
            
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError("Failed to read image file")
        
        model.scale_percent = scale
        model.noise_divisor = noise_divisor
        preprocessed = model.preprocess_image(image)
        preprocessed_b64 = image_to_base64(preprocessed)
        
        preprocessed_path = session_dirs['preprocessed'] / "preprocessed.png"
        cv2.imwrite(str(preprocessed_path), preprocessed)
        
        background_tasks.add_task(cleanup_old_sessions)
        
        return {
            "session_id": session_id,
            "preprocessed_image": preprocessed_b64,
            "debug_enabled": debug_mode
        }
        
    except Exception as e:
        if 'session_id' in locals():
            cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/{session_id}")
async def translate_preprocessed(
    session_id: str,
    debug_mode: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """Process the preprocessed image and return translation"""
    try:
        session_dirs = {
            'base': DEBUG_DIR / session_id,
            'preprocessed': DEBUG_DIR / session_id / 'preprocessed',
            'translation': TRANSLATION_DIR / session_id
        }
        
        if debug_mode:
            model.set_debug_mode(True, session_dirs['base'])
            
        preprocessed_path = session_dirs['preprocessed'] / "preprocessed.png"
        if not preprocessed_path.exists():
            raise HTTPException(status_code=404, detail="Preprocessed image not found")
            
        preprocessed = cv2.imread(str(preprocessed_path), cv2.IMREAD_GRAYSCALE)
        characters = model.segment_image(preprocessed)
        predicted_text = model.predict_text(characters)
        
        translation_path = session_dirs['translation'] / "translation.txt"
        with open(translation_path, "w", encoding="utf-8") as f:
            f.write(predicted_text)
        
        if not debug_mode:
            background_tasks.add_task(cleanup_session, session_id)
            
        return FileResponse(
            path=str(translation_path),
            media_type='text/plain',
            filename='translation.txt'
        )
        
    except Exception as e:
        if not debug_mode:
            cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize model and clean up old sessions on startup"""
    try:
        if not MODEL_PATH.exists():
            model.train_and_save(BASE_DIR, str(MODEL_PATH))
        else:
            model.load_saved_model(str(MODEL_PATH))
        cleanup_old_sessions()
    except Exception as e:
        print(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    cleanup_old_sessions(max_age_hours=0)  # Clean up all sessions