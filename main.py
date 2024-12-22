from model import TamilInscriptionModel
from pathlib import Path
import os

# Get the current directory
base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# Create required directories
(base_dir / 'uploads').mkdir(exist_ok=True)
(base_dir / 'preprocessed').mkdir(exist_ok=True)
(base_dir / 'saved_model').mkdir(exist_ok=True)
(base_dir / 'prediction_files' / 'train').mkdir(parents=True, exist_ok=True)

# Check if required files exist
model_path = base_dir / 'prediction_files' / 'model_tva_l_v1_2082024.h5'
csv_path = base_dir / 'prediction_files' / 'recognition-tva2082024.csv'

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found at {model_path}. "
        "Please place your trained model file in the prediction_files directory."
    )

if not csv_path.exists():
    raise FileNotFoundError(
        f"Character mapping file not found at {csv_path}. "
        "Please place your character mapping CSV file in the prediction_files directory."
    )

# Initialize and save model
model = TamilInscriptionModel()
save_path = base_dir / 'saved_model' / 'tamil_inscription_model.joblib'
model.train_and_save(base_dir, str(save_path))
print(f"Model successfully saved to {save_path}")