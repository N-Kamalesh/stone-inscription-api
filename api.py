from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import shutil
import uuid
from pathlib import Path
import os
from model import TamilInscriptionModel
from fastapi.middleware.cors import CORSMiddleware

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

for dir_path in [UPLOAD_DIR, PREPROCESSED_DIR, MODEL_PATH.parent, TRANSLATION_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Initialize model
model = TamilInscriptionModel()

@app.on_event("startup")
async def startup_event():
    """Load saved model on startup"""
    try:
        # Check if saved model exists
        if not MODEL_PATH.exists():
            # First time setup - train and save model
            model.train_and_save(BASE_DIR, str(MODEL_PATH))
        else:
            # Load existing saved model
            model.load_saved_model(str(MODEL_PATH))
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict/")
async def predict_inscription(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    file_path = UPLOAD_DIR / file.filename
    translation_path = TRANSLATION_DIR / f"translation_{file.filename.split('.')[0]}_{uuid.uuid4().hex[:6]}.txt"
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image and get predictions
        result = model.process_image(str(file_path))
        
        # Save translation to file and ensure it exists
        model.save_translation(result["predicted_text"], translation_path)
        
        if not translation_path.exists():
            raise HTTPException(status_code=500, detail="Failed to create translation file")
        
        # Add translation file to be removed after response is sent
        background_tasks.add_task(os.remove, str(translation_path))
        background_tasks.add_task(os.remove, str(file_path))
        
        # Return the translation file in response
        return FileResponse(
            path=str(translation_path),
            media_type='text/plain',
            filename='translation.txt',
            headers={
                "Content-Disposition": "attachment;filename=translation.txt",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
           
        
@app.get("/")
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "model_loaded": model.model is not None
    }

