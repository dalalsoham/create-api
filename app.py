from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io

app = FastAPI()

# Path to Tesseract (if needed on Windows, uncomment and update)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------
# Preprocessing function
# -------------------------------
def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Sharpen + upscale image for better OCR results."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale (2x) for better recognition
    scale_percent = 200
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

    # Sharpen filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    return sharp


@app.post("/process_card/")
async def process_card(file: UploadFile = File(...)):
    try:
        # Read file
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Enhance for OCR
        processed_img = enhance_for_ocr(img)

        # Run OCR
        text = pytesseract.image_to_string(processed_img)

        return JSONResponse(content={"extracted_text": text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
