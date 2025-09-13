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

# CONFIG
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

TESSERACT_CONFIG = r'--oem 3 --psm 6 -l eng'

TITLE_KEYWORDS = {
    # C-Suite / Founders
    "ceo","cto","cfo","coo","cio","cmo","cso","cpo","cbo","cro","cdo",
    "founder","co-founder","owner","partner","proprietor","chairman","chairperson",
    "president","vice president","vp","managing director","director","joint director",

    # Senior Roles
    "general manager","gm","associate director","assistant director","principal",
    "head","department head","hod","lead","leader","team lead","tech lead","architect",

    # Mid-level Roles
    "manager","assistant manager","associate manager","product manager",
    "project manager","program manager","operations manager","delivery manager",

    # Officers
    "officer","chief officer","senior officer","executive officer","admin officer",
    "compliance officer","account officer","finance officer","marketing officer",
    "sales officer","hr officer",

    # Specialists / Analysts
    "specialist","consultant","advisor","analyst","business analyst","data analyst",
    "financial analyst","investment analyst","market analyst","research analyst",
    "researcher","scientist","technologist","trainer","coach","mentor",

    # Engineers / Developers
    "engineer","developer","software engineer","senior engineer","systems engineer",
    "support engineer","field engineer","qa engineer","test engineer","devops engineer",
    "frontend developer","backend developer","fullstack developer","mobile developer",
    "data engineer","cloud engineer","ml engineer","ai engineer","research engineer",

    # Coordination Roles
    "coordinator","supervisor","administrator","controller","auditor","inspector",
    "secretary","registrar","facilitator","moderator",

    # Entry Level
    "assistant","associate","intern","trainee","apprentice","fellow","junior"
}


ORG_HINTS = [
    # Corporate Forms
    "pvt","ltd","llp","plc","inc","corp","company","co","enterprises","enterprise",
    "industries","industry","ventures","capital","partners","holdings","group",
    "associates","alliance","union","federation","trust","ngo","foundation",

    # Tech / Consulting
    "technologies","technology","solutions","systems","consulting","services",
    "software","hardware","infrastructure","platforms","innovation","labs","ai",
    "cloud","robotics","automation","analytics","digital","data","design","studio",

    # Education / Research
    "institute","academy","school","college","university","research","center",
    "council","organization","association","society","board","mission"
]


ADDRESS_HINTS = [
    # Common Address Words
    "road","street","st","lane","avenue","ave","sector","block","nagar","layout",
    "colony","phase","circle","cross","main","market","plaza","bazaar","marg","path",

    # Areas & Locations
    "city","town","village","district","state","mandal","taluk","region","zone",
    "ward","locality","society","complex","colony","residency","quarters","campus",

    # Buildings
    "park","garden","society","complex","tower","building","apartment","flat",
    "floor","wing","chamber","gate","arch","bridge","square","court","yard",

    # Other Identifiers
    "pin","zip","pincode","postal","india","highway","bypass","station","junction",
    "terminal","stand","hub","circle","chowk","bus stop","metro","railway"
]


# Common Indian names patterns
INDIAN_NAME_PATTERNS = [
    r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$',  # Three names
    r'^[A-Z][a-z]+ [A-Z]\s+[A-Z][a-z]+$',      # Name with middle initial
    r'^[A-Z][a-z]+ [A-Z][a-z]+$',              # Two names
]

# UTILS
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

# FIELD EXTRACTION
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

# FASTAPI APP 
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
