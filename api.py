from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from pathlib import Path
import os
import cv2
from model import TamilInscriptionModel

app = FastAPI(title="Tamil Stone Inscription Reader API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PREPROCESSED_DIR = BASE_DIR / "preprocessed"
MODEL_PATH = BASE_DIR / "saved_model/tamil_inscription_model.joblib"
TRANSLATION_DIR = BASE_DIR / "translations"
SEGMENTED_DIR = BASE_DIR / "segmented"  # New directory for segmented images

for dir_path in [UPLOAD_DIR, PREPROCESSED_DIR, MODEL_PATH.parent, TRANSLATION_DIR, SEGMENTED_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Initialize model
model = TamilInscriptionModel()

@app.on_event("startup")
async def startup_event():
    """Load saved model on startup"""
    try:
        if not MODEL_PATH.exists():
            model.train_and_save(BASE_DIR, str(MODEL_PATH))
        else:
            model.load_saved_model(str(MODEL_PATH))
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict/")
async def predict_inscription(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks(),
    save_debug_images: bool = False  # Optional parameter to save intermediate results
):
    """
    Process an inscription image and return the predicted text.
    Optionally saves debug images showing segmentation results.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Generate unique identifier for this process
    process_id = uuid.uuid4().hex[:6]
    file_path = UPLOAD_DIR / f"{process_id}_{file.filename}"
    translation_path = TRANSLATION_DIR / f"translation_{process_id}.txt"
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read and preprocess image
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError("Invalid image file")
            
        preprocessed = model.preprocess_image(image)
        
        # Segment characters
        characters = model.segment_image(preprocessed)
        if not characters:
            raise ValueError("No characters detected in image")
        
        # Get predictions for each line
        text = model.predict_text(characters)
        
        # Save translation
        with open(translation_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Save debug images if requested
        debug_images = {}
        if save_debug_images:
            debug_dir = SEGMENTED_DIR / process_id
            debug_dir.mkdir(exist_ok=True)
            
            # Save preprocessed image
            cv2.imwrite(str(debug_dir / "preprocessed.png"), preprocessed)
            
            # Save segmented lines
            for i, line in enumerate(characters):
                line_dir = debug_dir / f"line_{i+1}"
                line_dir.mkdir(exist_ok=True)
                for j, char_img in enumerate(line):
                    cv2.imwrite(str(line_dir / f"char_{j+1}.png"), char_img)
            
            debug_images = {
                "preprocessed": str(debug_dir / "preprocessed.png"),
                "num_lines": len(characters),
                "chars_per_line": [len(line) for line in characters]
            }
        
        # Clean up files after response
        background_tasks.add_task(os.remove, str(file_path))
        background_tasks.add_task(os.remove, str(translation_path))
        if save_debug_images:
            background_tasks.add_task(shutil.rmtree, str(debug_dir))
        
        # Return response
        return FileResponse(
            path=str(translation_path),
            media_type='text/plain',
            filename='translation.txt',
            headers={
                "Content-Disposition": "attachment;filename=translation.txt",
                "Access-Control-Expose-Headers": "Content-Disposition",
                "X-Debug-Info": str(debug_images) if save_debug_images else ""
            }
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/analyze/")
async def analyze_inscription(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Analyze an inscription image and return statistics about detected lines and characters.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    process_id = uuid.uuid4().hex[:6]
    file_path = UPLOAD_DIR / f"{process_id}_{file.filename}"
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError("Invalid image file")
            
        preprocessed = model.preprocess_image(image)
        characters = model.segment_image(preprocessed)
        
        # Analyze results
        analysis = {
            "num_lines": len(characters),
            "chars_per_line": [len(line) for line in characters],
            "total_chars": sum(len(line) for line in characters),
            "average_chars_per_line": sum(len(line) for line in characters) / len(characters) if characters else 0
        }
        
        # Clean up
        background_tasks.add_task(os.remove, str(file_path))
        
        return JSONResponse(content=analysis)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
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
            "Analysis endpoints"
        ]
    }