# import os
# import re
# import json
# import cv2
# import numpy as np
# from typing import List, Dict, Tuple, Optional

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from PIL import Image
# import pytesseract
# import shutil

# # ================= CONFIG =================
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # adjust if needed
# if os.path.exists(TESSERACT_PATH):
#     pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# TESSERACT_CONFIG = r'--oem 3 --psm 6 -l eng'

# TITLE_KEYWORDS = {"ceo","cto","cfo","coo","founder","director","manager","head","engineer","developer","executive","analyst","specialist","coordinator","supervisor","lead","officer","assistant","associate"}
# ORG_HINTS = ["pvt","ltd","llp","inc","corp","company","technologies","solutions","systems","enterprises","group","organization","consulting","services","software"]
# ADDRESS_HINTS = ["road","street","lane","avenue","sector","block","nagar","layout","city","state","pin","zip","india","park","society","complex","tower","building","floor"]

# # Common Indian names patterns
# INDIAN_NAME_PATTERNS = [
#     r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$',  # Three names
#     r'^[A-Z][a-z]+ [A-Z]\s+[A-Z][a-z]+$',      # Name with middle initial
#     r'^[A-Z][a-z]+ [A-Z][a-z]+$',              # Two names
# ]

# # ================= UTILS =================
# def resize_image_if_needed(bgr: np.ndarray, max_dimension=1500) -> np.ndarray:
#     """Resize image if too large to speed up processing"""
#     h, w = bgr.shape[:2]
#     if max(h, w) > max_dimension:
#         scale = max_dimension / max(h, w)
#         new_w = int(w * scale)
#         new_h = int(h * scale)
#         return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
#     return bgr

# def enhance_for_ocr(bgr: np.ndarray) -> np.ndarray:
#     """Simplified and faster image enhancement"""
#     # Resize if image is too large
#     bgr = resize_image_if_needed(bgr, max_dimension=1200)
    
#     # Simple denoising (faster than bilateral filter)
#     denoised = cv2.medianBlur(bgr, 3)
    
#     # Convert to grayscale and apply adaptive threshold
#     gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
#     # Simple contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(gray)
    
#     # Convert back to BGR for consistency
#     return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# def ocr_with_data(bgr: np.ndarray) -> Tuple[str, Dict[str, List]]:
#     """Faster OCR processing with simplified config"""
#     # Use simpler, faster tesseract config
#     fast_config = r'--oem 3 --psm 6'
    
#     try:
#         # Get text directly for speed
#         full_text = pytesseract.image_to_string(bgr, config=fast_config).strip()
        
#         # Get structured data only if needed
#         data = pytesseract.image_to_data(bgr, output_type=pytesseract.Output.DICT, config=fast_config)
        
#         # Simplified line processing
#         lines = {}
#         for i in range(len(data['text'])):
#             txt = data['text'][i].strip()
#             if not txt or len(txt) < 2:  # Skip very short text
#                 continue
            
#             key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
#             x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
#             if key not in lines:
#                 lines[key] = {'words': [], 'box': [x, y, x+w, y+h], 'height': h}
#             else:
#                 # Update bounding box
#                 lines[key]['box'][0] = min(lines[key]['box'][0], x)
#                 lines[key]['box'][1] = min(lines[key]['box'][1], y)
#                 lines[key]['box'][2] = max(lines[key]['box'][2], x+w)
#                 lines[key]['box'][3] = max(lines[key]['box'][3], y+h)
#                 lines[key]['height'] = max(lines[key]['height'], h)
            
#             lines[key]['words'].append(txt)
        
#         # Sort and process lines
#         line_texts, line_boxes = [], []
#         for _, v in sorted(lines.items(), key=lambda kv: (kv[1]['box'][1], kv[1]['box'][0])):
#             text = " ".join(v['words']).strip()
#             if text and len(text) > 2:
#                 line_texts.append(text)
#                 x1, y1, x2, y2 = v['box']
#                 line_boxes.append((x1, y1, x2, y2, float(v['height'])))
        
#         return full_text, {'lines': line_texts, 'boxes': line_boxes, 'raw': data}
        
#     except Exception as e:
#         # Fallback to simple text extraction
#         simple_text = pytesseract.image_to_string(bgr, config=r'--psm 6')
#         lines = [line.strip() for line in simple_text.split('\n') if line.strip()]
#         boxes = [(0, i*20, 100, (i+1)*20, 12) for i in range(len(lines))]
#         return simple_text, {'lines': lines, 'boxes': boxes, 'raw': {}}

# # ================= FIELD EXTRACTION =================
# # More flexible email regex to handle OCR errors
# EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+[@(][\w.-]+\.[A-Za-z]{2,}')
# URL_RE = re.compile(r'((?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?)')
# PHONE_RE = re.compile(r'(\+?\d[\d\s().-]{7,}\d)')

# def clean_email(email: str) -> str:
#     """Clean OCR errors in email addresses"""
#     # Replace common OCR errors
#     email = email.replace('(', '@').replace('[', '@').replace('{', '@')
#     # Remove extra characters
#     email = re.sub(r'[|\\/<>]', '', email)
#     return email.strip()

# def norm_phone(num: str) -> str:
#     n = re.sub(r'[^0-9+]', '', num)
#     if n.startswith('+'):
#         base = re.sub(r'[^0-9]', '', n[1:])
#         return '+' + base
#     return re.sub(r'[^0-9]', '', n)

# def is_likely_name(text: str) -> bool:
#     """Check if a line is likely to be a person's name"""
#     # Remove extra spaces and clean
#     text = re.sub(r'\s+', ' ', text.strip())
    
#     # Skip if contains obvious non-name patterns
#     if any(char in text for char in ['@', '+', '|', ':', 'www', '.com', '.org', '.in']):
#         return False
    
#     # Skip if contains digits (except Roman numerals)
#     if re.search(r'\d', text):
#         return False
    
#     # Check for name-like patterns
#     words = text.split()
#     if len(words) < 2 or len(words) > 4:
#         return False
    
#     # Check if each word starts with capital letter
#     if not all(word[0].isupper() for word in words if word):
#         return False
    
#     # Check against common Indian name patterns
#     for pattern in INDIAN_NAME_PATTERNS:
#         if re.match(pattern, text):
#             return True
    
#     # Additional check for reasonable name length
#     return 5 <= len(text) <= 50 and all(len(word) >= 2 for word in words)

# def is_company_name(text: str) -> bool:
#     """Check if a line is likely to be a company name"""
#     text_lower = text.lower()
    
#     # Check for organization hints
#     if any(hint in text_lower for hint in ORG_HINTS):
#         return True
    
#     # Check for common company patterns
#     company_patterns = ['ltd', 'inc', 'corp', 'pvt', 'llc', 'group', 'company', 'enterprises', 'solutions', 'technologies', 'systems', 'services']
#     if any(pattern in text_lower for pattern in company_patterns):
#         return True
    
#     # Check if it's a short business-like name
#     words = text.split()
#     if len(words) <= 3 and not any(char in text for char in ['@', '+', '|', ':']):
#         return True
    
#     return False

# def is_title(text: str) -> bool:
#     """Check if a line is likely to be a job title"""
#     text_lower = text.lower()
#     return any(keyword in text_lower for keyword in TITLE_KEYWORDS)

# def is_address_line(text: str) -> bool:
#     """Check if a line is likely to be part of an address"""
#     text_lower = text.lower()
    
#     # Check for address hints
#     if any(hint in text_lower for hint in ADDRESS_HINTS):
#         return True
    
#     # Check for pin code patterns
#     if re.search(r'\b\d{6}\b', text):
#         return True
    
#     # Check for address-like patterns (contains numbers and location words)
#     if re.search(r'\d+', text) and len(text.split()) >= 2:
#         return True
    
#     return False

# def extract_fields(lines: List[str], boxes: List[Tuple[int,int,int,int,float]]) -> Dict:
#     """Optimized field extraction with early returns"""
#     if not lines:
#         return {
#             "name": None, "company_name": None, "title": None,
#             "emails": None, "phones": None, "websites": None, "address": None
#         }
    
#     text = "\n".join(lines)
    
#     # Quick extractions first
#     email_matches = EMAIL_RE.findall(text)
#     emails = [clean_email(email) for email in email_matches if '@' in clean_email(email)]
#     emails = list(dict.fromkeys(emails)) if emails else None
    
#     urls = list(dict.fromkeys([u if u.startswith('http') else 'http://' + u for u in URL_RE.findall(text)]))
#     urls = urls if urls else None
    
#     phones = []
#     for p in PHONE_RE.findall(text):
#         n = norm_phone(p)
#         digits = re.sub(r'[^0-9]', '', n)
#         if 10 <= len(digits) <= 13 and n not in phones:
#             phones.append(n)
#     phones = phones if phones else None
    
#     # Simplified field detection - process only first 10 lines for speed
#     process_lines = lines[:min(10, len(lines))]
#     name, company, title, address = None, None, None, None
    
#     # Quick pattern matching
#     for i, line in enumerate(process_lines):
#         clean_line = re.sub(r'\s+', ' ', line.strip())
#         if not clean_line or len(clean_line) < 3:
#             continue
        
#         # Skip lines with contact info
#         if any(pattern in clean_line for pattern in ['@', '+91', 'www', '|']):
#             continue
        
#         # Name detection (first clean line that looks like a name)
#         if not name and is_likely_name(clean_line):
#             name = clean_line
#             continue
        
#         # Company detection
#         if not company and name != clean_line and is_company_name(clean_line):
#             company = clean_line
#             continue
        
#         # Title detection
#         if not title and is_title(clean_line):
#             title = clean_line
#             continue
    
#     # Address - simple approach: find lines with address indicators
#     addr_lines = []
#     for line in process_lines:
#         if (is_address_line(line) and 
#             line != name and line != company and line != title and
#             not any(pattern in line for pattern in ['@', '+91', 'www'])):
#             clean_addr = re.sub(r'[|\\<>]', '', line).strip()
#             if clean_addr and len(clean_addr) > 5:
#                 addr_lines.append(clean_addr)
    
#     address = ", ".join(addr_lines[:3]) if addr_lines else None  # Limit to 3 lines
    
#     return {
#         "name": name,
#         "company_name": company,
#         "title": title,
#         "emails": emails,
#         "phones": phones,
#         "websites": urls,
#         "address": address
#     }

# # ================= FASTAPI APP =================
# app = FastAPI()

# class CardData(BaseModel):
#     name: Optional[str]
#     company_name: Optional[str]
#     title: Optional[str]
#     emails: Optional[List[str]]
#     phones: Optional[List[str]]
#     websites: Optional[List[str]]
#     address: Optional[str]
#     faces: Optional[List[str]]
#     logo: Optional[str]
#     raw_text: str

# @app.post("/extract-card", response_model=CardData)
# async def extract_card(file: UploadFile = File(...)):
#     temp_path = None
#     try:
#         # Check file size (limit to 10MB)
#         if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
#             return JSONResponse(status_code=400, content={"error": "File too large. Max 10MB allowed."})
        
#         # Save uploaded file
#         temp_path = f"./temp_{file.filename}"
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # Load and validate image
#         bgr = cv2.imread(temp_path)
#         if bgr is None:
#             return JSONResponse(status_code=400, content={"error": "Invalid image file"})
        
#         # Quick processing
#         enhanced = enhance_for_ocr(bgr)
#         full_text, ocr_data = ocr_with_data(enhanced)
        
#         if not ocr_data['lines']:
#             return JSONResponse(status_code=400, content={"error": "No text detected in image"})
        
#         fields = extract_fields(ocr_data['lines'], ocr_data['boxes'])
        
#         result = fields.copy()
#         result["faces"] = []  # placeholder
#         result["logo"] = None  # placeholder
#         result["raw_text"] = full_text
        
#         return result
        
#     except pytesseract.TesseractNotFoundError:
#         return JSONResponse(status_code=500, content={"error": "Tesseract OCR not found. Please install Tesseract."})
#     except FileNotFoundError:
#         return JSONResponse(status_code=500, content={"error": "Tesseract executable not found at specified path."})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})
#     finally:
#         # Clean up temp file
#         if temp_path and os.path.exists(temp_path):
#             try:
#                 os.remove(temp_path)
#             except:
#                 pass  # Ignore cleanup errors



##########################################################################



# import os
# import re
# import json
# import cv2
# import numpy as np
# from typing import List, Dict, Tuple, Optional

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from PIL import Image
# import pytesseract
# import shutil

# # ================= CONFIG =================
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # adjust if needed
# if os.path.exists(TESSERACT_PATH):
#     pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# TESSERACT_CONFIG = r'--oem 3 --psm 6 -l eng'

# TITLE_KEYWORDS = {"ceo","cto","cfo","coo","founder","director","manager","head","engineer","developer","executive","analyst","specialist","coordinator","supervisor","lead","officer","assistant","associate"}
# ORG_HINTS = ["pvt","ltd","llp","inc","corp","company","technologies","solutions","systems","enterprises","group","organization","consulting","services","software"]
# ADDRESS_HINTS = ["road","street","lane","avenue","sector","block","nagar","layout","city","state","pin","zip","india","park","society","complex","tower","building","floor"]

# # Common Indian names patterns
# INDIAN_NAME_PATTERNS = [
#     r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$',  # Three names
#     r'^[A-Z][a-z]+ [A-Z]\s+[A-Z][a-z]+$',      # Name with middle initial
#     r'^[A-Z][a-z]+ [A-Z][a-z]+$',              # Two names
# ]

# # ================= UTILS =================
# def resize_image_if_needed(bgr: np.ndarray, max_dimension=1500) -> np.ndarray:
#     """Resize image if too large to speed up processing"""
#     h, w = bgr.shape[:2]
#     if max(h, w) > max_dimension:
#         scale = max_dimension / max(h, w)
#         new_w = int(w * scale)
#         new_h = int(h * scale)
#         return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
#     return bgr

# def enhance_for_ocr(bgr: np.ndarray) -> np.ndarray:
#     """Simplified and faster image enhancement"""
#     # Resize if image is too large
#     bgr = resize_image_if_needed(bgr, max_dimension=1200)
    
#     # Simple denoising (faster than bilateral filter)
#     denoised = cv2.medianBlur(bgr, 3)
    
#     # Convert to grayscale and apply adaptive threshold
#     gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
#     # Simple contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(gray)
    
#     # Convert back to BGR for consistency
#     return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# def ocr_with_data(bgr: np.ndarray) -> Tuple[str, Dict[str, List]]:
#     """Faster OCR processing with simplified config"""
#     # Use simpler, faster tesseract config
#     fast_config = r'--oem 3 --psm 6'
    
#     try:
#         # Get text directly for speed
#         full_text = pytesseract.image_to_string(bgr, config=fast_config).strip()
        
#         # Get structured data only if needed
#         data = pytesseract.image_to_data(bgr, output_type=pytesseract.Output.DICT, config=fast_config)
        
#         # Simplified line processing
#         lines = {}
#         for i in range(len(data['text'])):
#             txt = data['text'][i].strip()
#             if not txt or len(txt) < 2:  # Skip very short text
#                 continue
            
#             key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
#             x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
#             if key not in lines:
#                 lines[key] = {'words': [], 'box': [x, y, x+w, y+h], 'height': h}
#             else:
#                 # Update bounding box
#                 lines[key]['box'][0] = min(lines[key]['box'][0], x)
#                 lines[key]['box'][1] = min(lines[key]['box'][1], y)
#                 lines[key]['box'][2] = max(lines[key]['box'][2], x+w)
#                 lines[key]['box'][3] = max(lines[key]['box'][3], y+h)
#                 lines[key]['height'] = max(lines[key]['height'], h)
            
#             lines[key]['words'].append(txt)
        
#         # Sort and process lines
#         line_texts, line_boxes = [], []
#         for _, v in sorted(lines.items(), key=lambda kv: (kv[1]['box'][1], kv[1]['box'][0])):
#             text = " ".join(v['words']).strip()
#             if text and len(text) > 2:
#                 line_texts.append(text)
#                 x1, y1, x2, y2 = v['box']
#                 line_boxes.append((x1, y1, x2, y2, float(v['height'])))
        
#         return full_text, {'lines': line_texts, 'boxes': line_boxes, 'raw': data}
        
#     except Exception as e:
#         # Fallback to simple text extraction
#         simple_text = pytesseract.image_to_string(bgr, config=r'--psm 6')
#         lines = [line.strip() for line in simple_text.split('\n') if line.strip()]
#         boxes = [(0, i*20, 100, (i+1)*20, 12) for i in range(len(lines))]
#         return simple_text, {'lines': lines, 'boxes': boxes, 'raw': {}}

# # ================= FIELD EXTRACTION =================
# # More flexible email regex to handle OCR errors
# EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+[@(][\w.-]+\.[A-Za-z]{2,}')
# URL_RE = re.compile(r'((?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?)')
# PHONE_RE = re.compile(r'(\+?\d[\d\s().-]{7,}\d)')

# def clean_email(email: str) -> str:
#     """Clean OCR errors in email addresses"""
#     # Replace common OCR errors
#     email = email.replace('(', '@').replace('[', '@').replace('{', '@')
#     # Remove extra characters
#     email = re.sub(r'[|\\/<>]', '', email)
#     return email.strip()

# def norm_phone(num: str) -> str:
#     n = re.sub(r'[^0-9+]', '', num)
#     if n.startswith('+'):
#         base = re.sub(r'[^0-9]', '', n[1:])
#         return '+' + base
#     return re.sub(r'[^0-9]', '', n)

# def is_likely_name(text: str) -> bool:
#     """Check if a line is likely to be a person's name"""
#     # Remove extra spaces and clean
#     text = re.sub(r'\s+', ' ', text.strip())
    
#     # Skip if contains obvious non-name patterns
#     if any(char in text for char in ['@', '+', '|', ':', 'www', '.com', '.org', '.in']):
#         return False
    
#     # Skip if contains digits
#     if re.search(r'\d', text):
#         return False
    
#     # Skip if contains common title words
#     text_lower = text.lower()
#     if any(keyword in text_lower for keyword in TITLE_KEYWORDS):
#         return False
    
#     # Skip if contains organization indicators
#     if any(hint in text_lower for hint in ORG_HINTS):
#         return False
    
#     # Check for name-like patterns
#     words = text.split()
#     if len(words) < 2 or len(words) > 4:
#         return False
    
#     # Each word should be reasonable length and start with capital
#     for word in words:
#         if len(word) < 2 or not word[0].isupper():
#             return False
#         # Skip if word is all caps (likely company/title)
#         if word.isupper() and len(word) > 3:
#             return False
    
#     # Additional check for reasonable name length
#     return 5 <= len(text) <= 50

# def is_company_name(text: str) -> bool:
#     """Check if a line is likely to be a company name"""
#     text_lower = text.lower()
    
#     # Direct organization hints
#     if any(hint in text_lower for hint in ORG_HINTS):
#         return True
    
#     # Check for common business suffixes
#     business_suffixes = ['ltd', 'inc', 'corp', 'pvt', 'llc', 'llp']
#     if any(text_lower.endswith(suffix) for suffix in business_suffixes):
#         return True
    
#     # Check if it's likely a business name pattern
#     words = text.split()
    
#     # Single word companies or abbreviations
#     if len(words) == 1:
#         # All caps abbreviations (like "LCS")
#         if text.isupper() and 2 <= len(text) <= 6:
#             return True
#         # Business-like single words
#         if any(pattern in text_lower for pattern in ['tech', 'soft', 'system', 'solution', 'service', 'consult']):
#             return True
    
#     # Short company names with country/location (like "LCS, India")
#     if len(words) <= 4:
#         if any(word.lower() in ['india', 'usa', 'uk', 'singapore', 'dubai'] for word in words):
#             return True
#         # Contains punctuation like commas (business naming pattern)
#         if ',' in text:
#             return True
    
#     # Skip if it looks like a person's name
#     if is_likely_name(text):
#         return False
    
#     return False

# def is_title(text: str) -> bool:
#     """Check if a line is likely to be a job title"""
#     text_lower = text.lower()
#     return any(keyword in text_lower for keyword in TITLE_KEYWORDS)

# def is_address_line(text: str) -> bool:
#     """Check if a line is likely to be part of an address"""
#     text_lower = text.lower()
    
#     # Check for address hints
#     if any(hint in text_lower for hint in ADDRESS_HINTS):
#         return True
    
#     # Check for pin code patterns
#     if re.search(r'\b\d{6}\b', text):
#         return True
    
#     # Check for address-like patterns (contains numbers and location words)
#     if re.search(r'\d+', text) and len(text.split()) >= 2:
#         return True
    
#     return False

# def extract_fields(lines: List[str], boxes: List[Tuple[int,int,int,int,float]]) -> Dict:
#     """Improved field extraction with better name/company separation"""
#     if not lines:
#         return {
#             "name": None, "company_name": None, "title": None,
#             "emails": None, "phones": None, "websites": None, "address": None
#         }
    
#     text = "\n".join(lines)
    
#     # Quick extractions first
#     email_matches = EMAIL_RE.findall(text)
#     emails = [clean_email(email) for email in email_matches if '@' in clean_email(email)]
#     emails = list(dict.fromkeys(emails)) if emails else None
    
#     urls = list(dict.fromkeys([u if u.startswith('http') else 'http://' + u for u in URL_RE.findall(text)]))
#     urls = urls if urls else None
    
#     phones = []
#     for p in PHONE_RE.findall(text):
#         n = norm_phone(p)
#         digits = re.sub(r'[^0-9]', '', n)
#         if 10 <= len(digits) <= 13 and n not in phones:
#             phones.append(n)
#     phones = phones if phones else None
    
#     # Process lines for field extraction
#     process_lines = lines[:min(10, len(lines))]
#     name, company, title = None, None, None
    
#     # First pass: identify title and remove from consideration
#     for line in process_lines:
#         clean_line = re.sub(r'\s+', ' ', line.strip())
#         if not clean_line or len(clean_line) < 3:
#             continue
        
#         # Skip lines with contact info
#         if any(pattern in clean_line for pattern in ['@', '+91', 'www', '|']):
#             continue
            
#         if not title and is_title(clean_line):
#             title = clean_line
#             break
    
#     # Second pass: identify company first (often appears before name)
#     for line in process_lines:
#         clean_line = re.sub(r'\s+', ' ', line.strip())
#         if not clean_line or clean_line == title:
#             continue
        
#         # Skip lines with contact info
#         if any(pattern in clean_line for pattern in ['@', '+91', 'www', '|']):
#             continue
        
#         # Company detection - prioritize lines with clear business indicators
#         if not company and is_company_name(clean_line):
#             # Double-check it's not a person's name
#             if not is_likely_name(clean_line):
#                 company = clean_line
#                 break
    
#     # Third pass: identify name (avoid confusion with company)
#     for line in process_lines:
#         clean_line = re.sub(r'\s+', ' ', line.strip())
#         if not clean_line or clean_line == title or clean_line == company:
#             continue
        
#         # Skip lines with contact info
#         if any(pattern in clean_line for pattern in ['@', '+91', 'www', '|']):
#             continue
        
#         # Name detection
#         if not name and is_likely_name(clean_line):
#             # Make sure it's not already identified as company
#             if clean_line != company:
#                 name = clean_line
#                 break
    
#     # If we still don't have a company, look for any reasonable business name
#     if not company:
#         for line in process_lines:
#             clean_line = re.sub(r'\s+', ' ', line.strip())
#             if (clean_line and clean_line != name and clean_line != title and
#                 not any(pattern in clean_line for pattern in ['@', '+91', 'www', '|']) and
#                 len(clean_line.split()) <= 4):
#                 # Check if it contains business-like elements
#                 if (',' in clean_line or 
#                     any(word.isupper() for word in clean_line.split()) or
#                     any(word.lower() in ['india', 'tech', 'solutions', 'systems'] for word in clean_line.split())):
#                     company = clean_line
#                     break
    
#     # Address extraction
#     addr_lines = []
#     for line in process_lines:
#         if (is_address_line(line) and 
#             line != name and line != company and line != title and
#             not any(pattern in line for pattern in ['@', '+91', 'www'])):
#             clean_addr = re.sub(r'[|\\<>]', '', line).strip()
#             if clean_addr and len(clean_addr) > 5:
#                 addr_lines.append(clean_addr)
    
#     address = ", ".join(addr_lines[:3]) if addr_lines else None
    
#     return {
#         "name": name,
#         "company_name": company,
#         "title": title,
#         "emails": emails,
#         "phones": phones,
#         "websites": urls,
#         "address": address
#     }

# # ================= FASTAPI APP =================
# app = FastAPI()

# class CardData(BaseModel):
#     name: Optional[str]
#     company_name: Optional[str]
#     title: Optional[str]
#     emails: Optional[List[str]]
#     phones: Optional[List[str]]
#     websites: Optional[List[str]]
#     address: Optional[str]
#     faces: Optional[List[str]]
#     logo: Optional[str]
#     raw_text: str

# @app.post("/extract-card", response_model=CardData)
# async def extract_card(file: UploadFile = File(...)):
#     temp_path = None
#     try:
#         # Check file size (limit to 10MB)
#         if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
#             return JSONResponse(status_code=400, content={"error": "File too large. Max 10MB allowed."})
        
#         # Save uploaded file
#         temp_path = f"./temp_{file.filename}"
#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # Load and validate image
#         bgr = cv2.imread(temp_path)
#         if bgr is None:
#             return JSONResponse(status_code=400, content={"error": "Invalid image file"})
        
#         # Quick processing
#         enhanced = enhance_for_ocr(bgr)
#         full_text, ocr_data = ocr_with_data(enhanced)
        
#         if not ocr_data['lines']:
#             return JSONResponse(status_code=400, content={"error": "No text detected in image"})
        
#         fields = extract_fields(ocr_data['lines'], ocr_data['boxes'])
        
#         result = fields.copy()
#         result["faces"] = []  # placeholder
#         result["logo"] = None  # placeholder
#         result["raw_text"] = full_text
        
#         return result
        
#     except pytesseract.TesseractNotFoundError:
#         return JSONResponse(status_code=500, content={"error": "Tesseract OCR not found. Please install Tesseract."})
#     except FileNotFoundError:
#         return JSONResponse(status_code=500, content={"error": "Tesseract executable not found at specified path."})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})
#     finally:
#         # Clean up temp file
#         if temp_path and os.path.exists(temp_path):
#             try:
#                 os.remove(temp_path)
#             except:
#                 pass  # Ignore cleanup errors




#####################################################################
# app_easyocr.py
# import os
# import re
# import json
# import shutil
# import cv2
# import numpy as np
# from typing import List, Dict, Tuple, Optional

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# # EasyOCR for detection+recognition (CRAFT + CRNN style pipeline)
# import easyocr
# import phonenumbers  # pip install phonenumbers

# # ================= CONFIG =================
# # EasyOCR will auto-use GPU if available (torch). Set GPU=False to force CPU
# EASYOCR_GPU = False

# reader = easyocr.Reader(["en"], gpu=EASYOCR_GPU)  # load English; add other langs if needed

# # regex patterns
# EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
# URL_RE = re.compile(r'((?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?)')
# PHONE_CAND_RE = re.compile(r'(\+?\d[\d\s().-]{6,}\d)')  # catch candidates, we'll validate with phonenumbers


# # simple helpers
# def resize_image_if_needed(bgr: np.ndarray, max_dim=1600) -> np.ndarray:
#     h, w = bgr.shape[:2]
#     if max(h, w) > max_dim:
#         scale = max_dim / max(h, w)
#         return cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
#     return bgr

# def normalize_text(t: str) -> str:
#     t = t.strip()
#     # fix common OCR artifacts
#     t = t.replace('(', '').replace(')', '')
#     t = t.replace('|', ' ').replace('|||', ' ')
#     t = re.sub(r'\s+', ' ', t)
#     return t.strip()

# # parse easyocr output -> sorted lines
# def easyocr_to_lines(image: np.ndarray):
#     """
#     Runs EasyOCR and returns list of (text, bbox, conf)
#     bbox is [ (x1,y1), (x2,y2), (x3,y3), (x4,y4) ]
#     We'll convert to (xmin,ymin,xmax,ymax) and compute center-y for sorting.
#     """
#     results = reader.readtext(image, detail=1)  # detail=1 returns (bbox,text,conf)
#     lines = []
#     for bbox, text, conf in results:
#         txt = normalize_text(text)
#         if not txt:
#             continue
#         xs = [int(pt[0]) for pt in bbox]
#         ys = [int(pt[1]) for pt in bbox]
#         xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
#         lines.append({
#             "text": txt,
#             "bbox": (xmin, ymin, xmax, ymax, float(ymax-ymin)),
#             "conf": float(conf)
#         })
#     # sort by top (y) then left (x)
#     lines = sorted(lines, key=lambda r: (r["bbox"][1], r["bbox"][0]))
#     # merge nearby boxes on same line (optional)
#     merged = []
#     for r in lines:
#         if not merged:
#             merged.append(r)
#             continue
#         prev = merged[-1]
#         # heuristic: if vertical overlap and close horizontally -> same line
#         if abs(prev['bbox'][1] - r['bbox'][1]) < max(10, prev['bbox'][4]*0.5):
#             # merge text and bbox; confidence = avg weighted
#             merged_text = prev['text'] + " " + r['text']
#             xmin = min(prev['bbox'][0], r['bbox'][0])
#             ymin = min(prev['bbox'][1], r['bbox'][1])
#             xmax = max(prev['bbox'][2], r['bbox'][2])
#             ymax = max(prev['bbox'][3], r['bbox'][3])
#             new_conf = (prev['conf'] + r['conf'])/2
#             merged[-1] = {"text": normalize_text(merged_text), "bbox": (xmin,ymin,xmax,ymax, float(ymax-ymin)), "conf": new_conf}
#         else:
#             merged.append(r)
#     return merged

# # Field extraction + confidence aggregation
# def extract_fields_from_lines(lines: List[Dict]) -> Dict:
#     texts = [ln['text'] for ln in lines]
#     joined = "\n".join(texts)

#     # RAW quick extracts
#     emails = sorted(set(EMAIL_RE.findall(joined)))
#     # validate emails and clean obvious OCR mistakes
#     emails_clean = []
#     for e in emails:
#         ec = e.replace('(at)', '@').replace('[at]', '@').replace(' ', '')
#         ec = re.sub(r'[^A-Za-z0-9@._\-+]', '', ec)
#         if '@' in ec and '.' in ec.split('@')[-1]:
#             emails_clean.append(ec.lower())
#     emails_clean = emails_clean or None

#     urls = []
#     for m in URL_RE.findall(joined):
#         u = m if m.startswith('http') else 'http://' + m
#         urls.append(u)
#     urls = list(dict.fromkeys(urls)) or None

#     # phone candidates then validate with phonenumbers
#     phones = []
#     for m in PHONE_CAND_RE.findall(joined):
#         cand = re.sub(r'[\s()\-]', '', m)
#         try:
#             # try parse (assume India default if no + given)
#             if cand.startswith('+'):
#                 pn = phonenumbers.parse(cand, None)
#             else:
#                 pn = phonenumbers.parse(cand, "IN")
#             if phonenumbers.is_valid_number(pn):
#                 phones.append(phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164))
#         except Exception:
#             # try looser normalization
#             digits = re.sub(r'[^0-9+]', '', m)
#             if len(re.sub(r'[^0-9]', '', digits)) >= 10:
#                 phones.append(digits)
#     phones = list(dict.fromkeys(phones)) or None

#     # Heuristics for company, name, title, address using positions + confidence
#     # Sort lines by bbox center-y (already sorted) and build candidate lists with confidences
#     candidates = lines  # already sorted top->bottom

#     name, company, title, address = None, None, None
#     name_conf, company_conf, title_conf = 0.0, 0.0, 0.0

#     # 1) Look for title lines (contain title keywords)
#     title_keywords = set([
#         "ceo","cto","cfo","coo","founder","director","manager","head","engineer","developer","executive","analyst","specialist","coordinator","supervisor","lead","officer","assistant","associate"
#     ])
#     for ln in candidates:
#         low = ln['text'].lower()
#         if any(tk in low for tk in title_keywords):
#             title = ln['text']
#             title_conf = ln['conf']
#             break

#     # 2) Company detection: prefer top lines that contain org hints or are ALL CAPS or contain 'India' etc.
#     org_hints = ['pvt','ltd','llp','inc','corp','company','technologies','solutions','systems','enterprises','group','organization','consulting','services','software']
#     for ln in candidates[:6]:  # top few lines
#         low = ln['text'].lower()
#         is_org = any(h in low for h in org_hints) or (sum(1 for c in ln['text'] if c.isupper()) / max(1, len([c for c in ln['text'] if c.isalpha()])) > 0.6) or ('india' in low)
#         if is_org:
#             company = ln['text']
#             company_conf = ln['conf']
#             break

#     # 3) Name detection: look for lines with 2-3 capitalized words, not phone/email/url, and moderate-high conf
#     def likely_name_text(t: str) -> bool:
#         if '@' in t or 'www' in t or 'http' in t or any(c.isdigit() for c in t):
#             return False
#         parts = t.split()
#         if not (2 <= len(parts) <= 4):
#             return False
#         # each part should start with uppercase letter ideally
#         score = sum(1 for p in parts if p and p[0].isupper())
#         return (score / len(parts)) >= 0.6

#     for ln in candidates[:8]:
#         if company and ln['text'] == company:
#             continue
#         if likely_name_text(ln['text']):
#             name = ln['text']
#             name_conf = ln['conf']
#             break

#     # 4) Fallbacks: if name or company missing, use top lines heuristics
#     if not company and len(candidates) >= 1:
#         # top-most with length >2
#         for ln in candidates[:3]:
#             if len(ln['text']) > 2 and ln['text'] != name:
#                 company = company or ln['text']
#                 company_conf = ln['conf']
#                 break

#     if not name and len(candidates) >= 2:
#         for ln in candidates[:6]:
#             if ln['text'] != company and likely_name_text(ln['text']):
#                 name = ln['text']
#                 name_conf = ln['conf']
#                 break

#     # Address detection: lines containing numbers, pin codes or address hints, joined
#     addr_hints = ['road','street','lane','avenue','sector','block','nagar','layout','city','state','pin','pincode','park','society','complex','tower','building','floor']
#     addr_parts = []
#     for ln in candidates:
#         low = ln['text'].lower()
#         if any(h in low for h in addr_hints) or re.search(r'\b\d{5,6}\b', ln['text']):
#             # exclude if it's contact line
#             if not (EMAIL_RE.search(ln['text']) or PHONE_CAND_RE.search(ln['text']) or URL_RE.search(ln['text'])):
#                 addr_parts.append(ln['text'])
#     address = ", ".join(addr_parts) if addr_parts else None

#     # Field confidence aggregation (average conf of matching lines)
#     def field_confidence(field_text: Optional[str]) -> Optional[float]:
#         if not field_text:
#             return None
#         for ln in candidates:
#             if ln['text'] == field_text:
#                 return round(float(ln['conf']), 3)
#         return None

#     return {
#         "name": name,
#         "name_conf": field_confidence(name),
#         "company_name": company,
#         "company_conf": field_confidence(company),
#         "title": title,
#         "title_conf": field_confidence(title),
#         "emails": emails_clean,
#         "phones": phones,
#         "websites": urls,
#         "address": address,
#         "raw_lines": [ln['text'] for ln in candidates],
#         "raw_confidences": [ln['conf'] for ln in candidates]
#     }

# # ================= FASTAPI APP =================
# app = FastAPI(title="Business Card OCR (EasyOCR)")

# class CardData(BaseModel):
#     name: Optional[str]
#     company_name: Optional[str]
#     title: Optional[str]
#     emails: Optional[List[str]]
#     phones: Optional[List[str]]
#     websites: Optional[List[str]]
#     address: Optional[str]
#     # confidences are optional extra fields
#     name_conf: Optional[float] = None
#     company_conf: Optional[float] = None
#     title_conf: Optional[float] = None
#     raw_text: str

# @app.post("/extract-card", response_model=CardData)
# async def extract_card(file: UploadFile = File(...)):
#     temp_path = None
#     try:
#         # save uploaded
#         temp_path = f"./temp_{file.filename}"
#         with open(temp_path, "wb") as buf:
#             shutil.copyfileobj(file.file, buf)
#         # load image
#         img = cv2.imread(temp_path)
#         if img is None:
#             return JSONResponse(status_code=400, content={"error": "Invalid image file"})
#         img = resize_image_if_needed(img, max_dim=1600)
#         # run easyocr
#         lines = easyocr_to_lines(img)
#         if not lines:
#             return JSONResponse(status_code=400, content={"error": "No text detected"})
#         fields = extract_fields_from_lines(lines)
#         raw_text = "\n".join([ln['text'] for ln in lines])
#         response = {
#             "name": fields.get("name"),
#             "company_name": fields.get("company_name"),
#             "title": fields.get("title"),
#             "emails": fields.get("emails"),
#             "phones": fields.get("phones"),
#             "websites": fields.get("websites"),
#             "address": fields.get("address"),
#             "name_conf": fields.get("name_conf"),
#             "company_conf": fields.get("company_conf"),
#             "title_conf": fields.get("title_conf"),
#             "raw_text": raw_text
#         }
#         return response
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
#     finally:
#         if temp_path and os.path.exists(temp_path):
#             try:
#                 os.remove(temp_path)
#             except:
#                 pass





########################################################################################### 
# multiple file upload
import os
import re
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import pytesseract
import shutil

# ================= CONFIG =================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # adjust if needed
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

TESSERACT_CONFIG = r'--oem 3 --psm 6 -l eng'

TITLE_KEYWORDS = {"ceo","cto","cfo","coo","founder","director","manager","head","engineer","developer","executive","analyst","specialist","coordinator","supervisor","lead","officer","assistant","associate"}
ORG_HINTS = ["pvt","ltd","llp","inc","corp","company","technologies","solutions","systems","enterprises","group","organization","consulting","services","software"]
ADDRESS_HINTS = ["road","street","lane","avenue","sector","block","nagar","layout","city","state","pin","zip","india","park","society","complex","tower","building","floor"]

# Common Indian names patterns
INDIAN_NAME_PATTERNS = [
    r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$',  # Three names
    r'^[A-Z][a-z]+ [A-Z]\s+[A-Z][a-z]+$',      # Name with middle initial
    r'^[A-Z][a-z]+ [A-Z][a-z]+$',              # Two names
]

# ================= UTILS =================
def resize_image_if_needed(bgr: np.ndarray, max_dimension=1500) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return bgr

def enhance_for_ocr(bgr: np.ndarray) -> np.ndarray:
    bgr = resize_image_if_needed(bgr, max_dimension=1200)
    denoised = cv2.medianBlur(bgr, 3)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def ocr_with_data(bgr: np.ndarray) -> Tuple[str, Dict[str, List]]:
    fast_config = r'--oem 3 --psm 6'
    try:
        full_text = pytesseract.image_to_string(bgr, config=fast_config).strip()
        data = pytesseract.image_to_data(bgr, output_type=pytesseract.Output.DICT, config=fast_config)
        lines = {}
        for i in range(len(data['text'])):
            txt = data['text'][i].strip()
            if not txt or len(txt) < 2:
                continue
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            if key not in lines:
                lines[key] = {'words': [], 'box': [x, y, x+w, y+h], 'height': h}
            else:
                lines[key]['box'][0] = min(lines[key]['box'][0], x)
                lines[key]['box'][1] = min(lines[key]['box'][1], y)
                lines[key]['box'][2] = max(lines[key]['box'][2], x+w)
                lines[key]['box'][3] = max(lines[key]['box'][3], y+h)
                lines[key]['height'] = max(lines[key]['height'], h)
            lines[key]['words'].append(txt)
        line_texts, line_boxes = [], []
        for _, v in sorted(lines.items(), key=lambda kv: (kv[1]['box'][1], kv[1]['box'][0])):
            text = " ".join(v['words']).strip()
            if text and len(text) > 2:
                line_texts.append(text)
                x1, y1, x2, y2 = v['box']
                line_boxes.append((x1, y1, x2, y2, float(v['height'])))
        return full_text, {'lines': line_texts, 'boxes': line_boxes, 'raw': data}
    except Exception as e:
        simple_text = pytesseract.image_to_string(bgr, config=r'--psm 6')
        lines = [line.strip() for line in simple_text.split('\n') if line.strip()]
        boxes = [(0, i*20, 100, (i+1)*20, 12) for i in range(len(lines))]
        return simple_text, {'lines': lines, 'boxes': boxes, 'raw': {}}

# ================= FIELD EXTRACTION =================
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+[@(][\w.-]+\.[A-Za-z]{2,}')
URL_RE = re.compile(r'((?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?)')
PHONE_RE = re.compile(r'(\+?\d[\d\s().-]{7,}\d)')

def clean_email(email: str) -> str:
    email = email.replace('(', '@').replace('[', '@').replace('{', '@')
    email = re.sub(r'[|\\/<>]', '', email)
    return email.strip()

def norm_phone(num: str) -> str:
    n = re.sub(r'[^0-9+]', '', num)
    if n.startswith('+'):
        base = re.sub(r'[^0-9]', '', n[1:])
        return '+' + base
    return re.sub(r'[^0-9]', '', n)

def is_likely_name(text: str) -> bool:
    text = re.sub(r'\s+', ' ', text.strip())
    if any(char in text for char in ['@', '+', '|', ':', 'www', '.com', '.org', '.in']):
        return False
    if re.search(r'\d', text):
        return False
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in TITLE_KEYWORDS):
        return False
    if any(hint in text_lower for hint in ORG_HINTS):
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 4:
        return False
    for word in words:
        if len(word) < 2 or not word[0].isupper():
            return False
        if word.isupper() and len(word) > 3:
            return False
    return 5 <= len(text) <= 50

def is_company_name(text: str) -> bool:
    text_lower = text.lower()
    if any(hint in text_lower for hint in ORG_HINTS):
        return True
    business_suffixes = ['ltd', 'inc', 'corp', 'pvt', 'llc', 'llp']
    if any(text_lower.endswith(suffix) for suffix in business_suffixes):
        return True
    words = text.split()
    if len(words) == 1:
        if text.isupper() and 2 <= len(text) <= 6:
            return True
        if any(pattern in text_lower for pattern in ['tech', 'soft', 'system', 'solution', 'service', 'consult']):
            return True
    if len(words) <= 4:
        if any(word.lower() in ['india', 'usa', 'uk', 'singapore', 'dubai'] for word in words):
            return True
        if ',' in text:
            return True
    if is_likely_name(text):
        return False
    return False

def is_title(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in TITLE_KEYWORDS)

def is_address_line(text: str) -> bool:
    text_lower = text.lower()
    if any(hint in text_lower for hint in ADDRESS_HINTS):
        return True
    if re.search(r'\b\d{6}\b', text):
        return True
    if re.search(r'\d+', text) and len(text.split()) >= 2:
        return True
    return False

def extract_fields(lines: List[str], boxes: List[Tuple[int,int,int,int,float]]) -> Dict:
    if not lines:
        return {
            "name": None, "company_name": None, "title": None,
            "emails": None, "phones": None, "websites": None, "address": None
        }
    text = "\n".join(lines)
    email_matches = EMAIL_RE.findall(text)
    emails = [clean_email(email) for email in email_matches if '@' in clean_email(email)]
    emails = list(dict.fromkeys(emails)) if emails else None
    urls = list(dict.fromkeys([u if u.startswith('http') else 'http://' + u for u in URL_RE.findall(text)]))
    urls = urls if urls else None
    phones = []
    for p in PHONE_RE.findall(text):
        n = norm_phone(p)
        digits = re.sub(r'[^0-9]', '', n)
        if 10 <= len(digits) <= 13 and n not in phones:
            phones.append(n)
    phones = phones if phones else None
    process_lines = lines[:min(10, len(lines))]
    name, company, title = None, None, None
    for line in process_lines:
        clean_line = re.sub(r'\s+', ' ', line.strip())
        if not clean_line or len(clean_line) < 3:
            continue
        if any(pattern in clean_line for pattern in ['@', '+91', 'www', '|']):
            continue
        if not title and is_title(clean_line):
            title = clean_line
            break
    for line in process_lines:
        clean_line = re.sub(r'\s+', ' ', line.strip())
        if not clean_line or clean_line == title:
            continue
        if any(pattern in clean_line for pattern in ['@', '+91', 'www', '|']):
            continue
        if not company and is_company_name(clean_line):
            if not is_likely_name(clean_line):
                company = clean_line
                break
    for line in process_lines:
        clean_line = re.sub(r'\s+', ' ', line.strip())
        if not clean_line or clean_line == title or clean_line == company:
            continue
        if any(pattern in clean_line for pattern in ['@', '+91', 'www', '|']):
            continue
        if not name and is_likely_name(clean_line):
            if clean_line != company:
                name = clean_line
                break
    if not company:
        for line in process_lines:
            clean_line = re.sub(r'\s+', ' ', line.strip())
            if (clean_line and clean_line != name and clean_line != title and
                not any(pattern in clean_line for pattern in ['@', '+91', 'www', '|']) and
                len(clean_line.split()) <= 4):
                if (',' in clean_line or 
                    any(word.isupper() for word in clean_line.split()) or
                    any(word.lower() in ['india', 'tech', 'solutions', 'systems'] for word in clean_line.split())):
                    company = clean_line
                    break
    addr_lines = []
    for line in process_lines:
        if (is_address_line(line) and 
            line != name and line != company and line != title and
            not any(pattern in line for pattern in ['@', '+91', 'www'])):
            clean_addr = re.sub(r'[|\\<>]', '', line).strip()
            if clean_addr and len(clean_addr) > 5:
                addr_lines.append(clean_addr)
    address = ", ".join(addr_lines[:3]) if addr_lines else None
    return {
        "name": name,
        "company_name": company,
        "title": title,
        "emails": emails,
        "phones": phones,
        "websites": urls,
        "address": address
    }

# ================= FASTAPI APP =================
app = FastAPI()

class CardData(BaseModel):
    name: Optional[str]
    company_name: Optional[str]
    title: Optional[str]
    emails: Optional[List[str]]
    phones: Optional[List[str]]
    websites: Optional[List[str]]
    address: Optional[str]
    faces: Optional[List[str]]
    logo: Optional[str]
    raw_text: str
    file: Optional[str]

@app.post("/extract-card", response_model=CardData)
async def extract_card(file: UploadFile = File(...)):
    temp_path = None
    try:
        temp_path = f"./temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        bgr = cv2.imread(temp_path)
        if bgr is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})
        enhanced = enhance_for_ocr(bgr)
        full_text, ocr_data = ocr_with_data(enhanced)
        if not ocr_data['lines']:
            return JSONResponse(status_code=400, content={"error": "No text detected in image"})
        fields = extract_fields(ocr_data['lines'], ocr_data['boxes'])
        result = fields.copy()
        result["faces"] = []
        result["logo"] = None
        result["raw_text"] = full_text
        result["file"] = file.filename
        return result
    except pytesseract.TesseractNotFoundError:
        return JSONResponse(status_code=500, content={"error": "Tesseract OCR not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.post("/extract-cards")
async def extract_cards(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        temp_path = None
        try:
            temp_path = f"./temp_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            bgr = cv2.imread(temp_path)
            if bgr is None:
                results.append({"file": file.filename, "error": "Invalid image file"})
                continue
            enhanced = enhance_for_ocr(bgr)
            full_text, ocr_data = ocr_with_data(enhanced)
            if not ocr_data['lines']:
                results.append({"file": file.filename, "error": "No text detected"})
                continue
            fields = extract_fields(ocr_data['lines'], ocr_data['boxes'])
            result = fields.copy()
            result["faces"] = []
            result["logo"] = None
            result["raw_text"] = full_text
            result["file"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({"file": file.filename, "error": str(e)})
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    return {"results": results}
