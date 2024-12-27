from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from pathlib import Path
import os
from model import TamilInscriptionModel
from datetime import datetime

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

# Create all necessary directories
for dir_path in [MODEL_PATH.parent, TRANSLATION_DIR, DEBUG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize model
model = TamilInscriptionModel()

def create_session_dirs(session_id: str) -> dict:
    """Create session-specific directories for debugging and results"""
    session_dirs = {
        'base': DEBUG_DIR / session_id,
        'preprocessed': DEBUG_DIR / session_id / 'preprocessed',
        'segmented': DEBUG_DIR / session_id / 'segmented',
        'translation': TRANSLATION_DIR / session_id
    }
    
    for dir_path in session_dirs.values():
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

def cleanup_translation_file(file_path: Path):
    """Remove translation file after it's been sent"""
    try:
        if file_path.exists():
            file_path.unlink()
            if file_path.parent.exists() and not any(file_path.parent.iterdir()):
                file_path.parent.rmdir()
    except Exception as e:
        print(f"Error cleaning up translation file: {e}")

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

@app.post("/predict/")
async def predict_inscription(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    debug_mode: bool = False,
    cleanup_translation: bool = True,
    cleanup_debug: bool = True
):
    """
    Process an inscription image and return the translation file
    
    Args:
        file: Image file to process
        background_tasks: FastAPI background tasks
        debug_mode: Enable debug mode to save intermediate processing images
        cleanup_translation: Whether to delete translation file after sending
        cleanup_debug: Whether to delete debug files after processing
    """
    try:
        # Generate session ID and create directories
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        session_dirs = create_session_dirs(session_id)
        
        # Save uploaded file
        file_path = session_dirs['base'] / f"input_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process image with debug if requested
        if debug_mode:
            model.set_debug_mode(True, session_dirs['base'])
            
        # Process image and get prediction
        stats = model.process_image(str(file_path))
        
        # Save translation to file
        translation_path = session_dirs['translation'] / "translation.txt"
        with open(translation_path, "w", encoding="utf-8") as f:
            f.write(stats["predicted_text"])
        
        # Add cleanup tasks based on parameters
        if debug_mode and cleanup_debug:
            background_tasks.add_task(
                lambda: shutil.rmtree(session_dirs['base']) if session_dirs['base'].exists() else None
            )
        
        if cleanup_translation:
            background_tasks.add_task(cleanup_translation_file, translation_path)
        
        # Return the translation file
        return FileResponse(
            path=str(translation_path),
            media_type='text/plain',
            filename='translation.txt'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "model_loaded": model.model is not None,
        "version": "2.0",
        "features": [
            "Multi-line detection",
            "Character segmentation",
            "Debug image generation",
            "Translation file generation",
            "Optional debug cleanup",
            "Optional translation cleanup"
        ],
        "debug_dir_size": sum(f.stat().st_size for f in DEBUG_DIR.rglob('*') if f.is_file()) / (1024 * 1024)  # Size in MB
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    cleanup_old_sessions(max_age_hours=1)  # More aggressive cleanup on shutdown