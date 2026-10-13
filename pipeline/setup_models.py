'''
This script sets up the folder structure an ddownload the necessary models for the pipeline.
'''
print("importing...")
import os
import sys
from doctr.models import ocr_predictor
from unstructured_inference.inference.layout import get_model
print("imports done.")
# --- Directory Setup ---
base_dir = os.path.dirname(os.path.abspath(__file__))
doctr_dir = os.path.join(base_dir, "models_backup", "doctr")
os.makedirs(doctr_dir, exist_ok=True)

# Ensure we are online for this script to download
if "HF_HUB_OFFLINE" in os.environ:
    del os.environ["HF_HUB_OFFLINE"] 

# --- 1. Unstructured Layout Model (YOLOX) ---
try:
    # This uses the Hugging Face cache system internally
    yolox_model = get_model("yolox") 
    print("--- SUCCESS: YOLOX model downloaded successfully to the default Hugging Face cache.")
except Exception as e:
    print(f"--- ERROR downloading YOLOX model: {e}")
    sys.exit(1)

# --- 2. Doctr OCR Models ---
os.environ["DOCTR_CACHE_DIR"] = doctr_dir # Set custom cache path
print(f"Downloading Doctr OCR models to custom location: {doctr_dir}...")
try:
    # This downloads the models (e.g., crnn, fast_base) into the custom cache
    predictor = ocr_predictor(pretrained=True)
    print(f"--- SUCCESS: Doctr models downloaded successfully to {doctr_dir}")
except Exception as e:
    print(f"--- ERROR downloading Doctr models: {e}")
    sys.exit(1)
