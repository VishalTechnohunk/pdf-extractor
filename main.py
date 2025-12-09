import io
import json
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai
import uuid
from pathlib import Path

SAVE_DIR = Path("invoices")
SAVE_DIR.mkdir(exist_ok=True)


# =======================
# Load environment
# =======================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

# =======================
# Pydantic Schemas
# =======================

class InvoiceItem(BaseModel):
    sn: Optional[int] = None
    qty: Optional[float] = None
    item_name: Optional[str] = None
    packing: Optional[str] = None
    hsn: Optional[str] = None
    discount: Optional[float] = None
    batch: Optional[str] = None
    mrp: Optional[float] = None
    rate: Optional[float] = None
    expiry: Optional[str] = None   # e.g. "11/28"
    sgst_percent: Optional[float] = None
    cgst_percent: Optional[float] = None
    amount: Optional[float] = None

class InvoiceTotals(BaseModel):
    subtotal: Optional[float] = None
    total_qty: Optional[float] = None
    total_sgst: Optional[float] = None
    total_cgst: Optional[float] = None
    total_gst: Optional[float] = None
    grand_total: Optional[float] = None
    amount_in_words: Optional[str] = None

class PartyInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[List[str]] = None
    gst: Optional[str] = None
    dl_no: Optional[List[str]] = None

class BankDetails(BaseModel):
    account_name: Optional[str] = None
    bank_name: Optional[str] = None
    account_no: Optional[str] = None
    ifsc_code: Optional[str] = None


class InvoiceData(BaseModel):
    invoice_no: Optional[str] = None
    invoice_date: Optional[str] = None         # "18-04-2025 13:22"
    seller: Optional[PartyInfo] = None
    buyer: Optional[PartyInfo] = None
    items: List[InvoiceItem] = []
    totals: Optional[InvoiceTotals] = None
    bank_details: Optional[BankDetails] = None
    
    # raw_text: Optional[str] = None             # optional for debugging

# =======================
# FastAPI app
# =======================

app = FastAPI(
    title="PDF Invoice → JSON API",
    description="Upload a pharmacy GST invoice PDF and get structured JSON using Gemini.",
    version="1.0.0",
)

# Allow frontend access if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# Helper: Extract text from PDF
# =======================


def save_invoice_json(data: dict) -> str:
    file_id = uuid.uuid4().hex[:8]  # short unique id
    filename = f"invoice_{file_id}.json"
    filepath = SAVE_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return str(filepath)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    text_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        text_chunks.append(page_text)
    return "\n".join(text_chunks)

# =======================
# Helper: Call Gemini with strict JSON prompt
# =======================
def call_gemini_for_invoice_json(pdf_text: str) -> dict:
    """
    Sends raw invoice text to Gemini and expects valid JSON
    matching the InvoiceData schema.
    """

    prompt = f"""
You are an expert at reading messy PDF invoice text and converting it into clean JSON.

Below is the raw text of a PHARMACY GST INVOICE. 
Extract all information and output ONLY a valid JSON object using this exact schema:

{{
  "invoice_no": string or null,
  "invoice_date": string or null,
  "seller": {{
    "name": string or null,
    "address": string or null,
    "phone": list of str or null,
    "gst": string or null,
    "dl_no": list of string or null
  }},
  "buyer": {{
    "name": string or null,
    "address": string or null,
    "phone": list of string or null,
    "gst": string or null,
    "dl_no": string or null
  }},
  "items": [
    {{
      "sn": integer or null,
      "qty": number or null,
      "item_name": string or null,
      "packing": string or null,
      "hsn": string or null,
      "discount": number or null,
      "batch": string or null,
      "mrp": number or null,
      "rate": number or null,
      "expiry": string or null,
      "sgst_percent": number or null,
      "cgst_percent": number or null,
      "amount": number or null
    }}
  ],
  "totals": {{
    "subtotal": number or null,
    "total_qty": number or null,
    "total_sgst": number or null,
    "total_cgst": number or null,
    "total_gst": number or null,
    "grand_total": number or null,
    "amount_in_words": string or null
  }},
    "bank_details": {{
    "account_name": string or null,
    "bank_name": string or null,
    "account_no": string or null,
    "ifsc_code": string or null
    }},
  "raw_text": string  // put the full invoice text here
}}

IMPORTANT RULES:
- Return ONLY JSON. No explanation, no markdown, no ```json``` fence.
- Convert numeric fields to numbers (no commas), e.g. 6,197.75 → 6197.75
- If something is missing, use null for that field.
- For the items list, include one object per line item (SN row) in the invoice.

INVOICE TEXT:
\"\"\" 
{pdf_text}
\"\"\"
"""

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()

    # Sometimes models wrap in ```json ... ```
    if raw.startswith("```"):
        # strip first and last fence
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Gemini did not return valid JSON. Raw output:\n{raw[:1000]}")
    return data


# =======================
# API Endpoint
# =======================

@app.post("/upload-pdf", response_model=InvoiceData)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF invoice. Returns structured JSON extracted using Gemini.
    """
    if file.content_type not in [
        "application/pdf",
        "application/octet-stream",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # 1. Extract text from PDF
    try:
        pdf_text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading PDF: {str(e)}"
        )

    if not pdf_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from PDF."
        )

    # 2. Call Gemini to convert text → JSON
    try:
        invoice_dict = call_gemini_for_invoice_json(pdf_text)
    except ValueError as ve:
        # Gemini returned invalid JSON
        raise HTTPException(
            status_code=500,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling LLM: {str(e)}"
        )

    # 3. Validate against Pydantic schema (InvoiceData)
    try:
        invoice = InvoiceData(**invoice_dict)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"JSON does not match InvoiceData schema: {str(e)}"
        )

  # Save extracted JSON into file
    file_path = save_invoice_json(invoice_dict)

    return invoice

@app.get("/")
async def root():
    return {"message": "PDF Invoice → JSON extractor is running."}
