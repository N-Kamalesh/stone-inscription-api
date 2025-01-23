from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from pathlib import Path
import json
import cv2
from model import TamilInscriptionModel
from datetime import datetime
import base64

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

# Add new state management for continuous translations
CONTINUOUS_SESSIONS = {}  # Store continuous translation sessions

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

def cleanup_old_sessions(max_age_minutes: int = 15):
    """Clean up sessions older than specified minutes"""
    current_time = datetime.now()
    for session_dir in DEBUG_DIR.glob("*"):
        if session_dir.is_dir():
            dir_time = datetime.fromtimestamp(session_dir.stat().st_mtime)
            age_minutes = (current_time - dir_time).total_seconds() / 60
            if age_minutes > max_age_minutes:
                shutil.rmtree(session_dir)
                # Also cleanup corresponding translation directory
                translation_dir = TRANSLATION_DIR / session_dir.name
                if translation_dir.exists():
                    shutil.rmtree(translation_dir)
                # Remove from continuous sessions if exists
                session_id = session_dir.name
                if session_id in CONTINUOUS_SESSIONS:
                    del CONTINUOUS_SESSIONS[session_id]

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

@app.post("/start-continuous/")
async def start_continuous_session():
    """Start a new continuous translation session"""
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    CONTINUOUS_SESSIONS[session_id] = {
        "translations": [],  # List to store each image's translation
        "line_counts": []   # Store number of lines in each translation
    }
    return {"session_id": session_id}

@app.post("/continuous-translate/{session_id}")
async def continuous_translate(
    session_id: str,
    file: UploadFile = File(...),
    scale: float = Form(30.0),
    noise_divisor: float = Form(0.9),
    debug_mode: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """Process a new image in a continuous translation session"""
    try:
        if session_id not in CONTINUOUS_SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")

        # Create temporary directories for this image
        temp_session_id = f"temp_{uuid.uuid4().hex[:6]}"
        session_dirs = create_session_dirs(temp_session_id)

        # Save and process the image
        file_path = session_dirs['base'] / f"input_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if debug_mode:
            model.set_debug_mode(True, session_dirs['base'])

        # Process image
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError("Failed to read image file")

        model.scale_percent = scale
        model.noise_divisor = noise_divisor
        preprocessed = model.preprocess_image(image)
        preprocessed_b64 = image_to_base64(preprocessed)
        
        # Get translation
        characters = model.segment_image(preprocessed)
        predicted_text = model.predict_text(characters)
        
        # Split into lines
        lines = predicted_text.split('\n')
        
        # Store in session
        CONTINUOUS_SESSIONS[session_id]["translations"].append(lines)
        CONTINUOUS_SESSIONS[session_id]["line_counts"].append(len(lines))

        # Merge all translations line by line
        max_lines = max(CONTINUOUS_SESSIONS[session_id]["line_counts"])
        merged_lines = ["" for _ in range(max_lines)]
        
        for translation in CONTINUOUS_SESSIONS[session_id]["translations"]:
            for i, line in enumerate(translation):
                if merged_lines[i]:
                    merged_lines[i] += " " + line.strip()
                else:
                    merged_lines[i] = line.strip()

        merged_text = "\n".join(merged_lines)

        # Save merged translation
        translation_path = session_dirs['translation'] / "translation.txt"
        with open(translation_path, "w", encoding="utf-8") as f:
            f.write(merged_text)

        # Clean up temporary files
        background_tasks.add_task(cleanup_session, temp_session_id)

        return {
            "preprocessed_image": preprocessed_b64,
            "current_translation": lines,
            "merged_translation": merged_lines,
            "num_images": len(CONTINUOUS_SESSIONS[session_id]["translations"])
        }

    except Exception as e:
        if 'temp_session_id' in locals():
            cleanup_session(temp_session_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/complete-session/{session_id}")
async def complete_session(session_id: str):
    """Complete and clean up a continuous translation session"""
    try:
        if session_id not in CONTINUOUS_SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found")
            
        session_data = CONTINUOUS_SESSIONS.pop(session_id)
        max_lines = max(session_data["line_counts"])
        merged_lines = ["" for _ in range(max_lines)]
        
        for translation in session_data["translations"]:
            for i, line in enumerate(translation):
                if merged_lines[i]:
                    merged_lines[i] += " " + line.strip()
                else:
                    merged_lines[i] = line.strip()

        final_text = "\n".join(merged_lines)
        
        # Create a temporary file for download
        temp_file = TEMP_DIR / f"final_translation_{session_id}.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(final_text)
            
        return FileResponse(
            path=str(temp_file),
            media_type='text/plain',
            filename='final_translation.txt'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Schedule cleanup of old sessions
        background_tasks.add_task(cleanup_old_sessions, max_age_minutes=15)
        
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
        
        # Schedule cleanup for 15 minutes later instead of immediate cleanup
        if not debug_mode:
            background_tasks.add_task(cleanup_old_sessions, max_age_minutes=15)
            
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
        "version": "3.0",
    }

@app.on_event("startup")
async def startup_event():
    """Initialize model and clean up old sessions on startup"""
    try:
        if not MODEL_PATH.exists():
            model.train_and_save(BASE_DIR, str(MODEL_PATH))
        else:
            model.load_saved_model(str(MODEL_PATH))
        cleanup_old_sessions(max_age_minutes=15)  # Clean up sessions older than 15 minutes
    except Exception as e:
        print(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    cleanup_old_sessions(max_age_minutes=0)  # Clean up all sessions