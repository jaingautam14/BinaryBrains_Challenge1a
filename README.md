# PDF Processing Solution with OCR and Language Detection

A high-performance containerized solution for extracting structured data from PDF documents with advanced OCR capabilities and automatic language detection.

## 🚀 Key Features & What Makes It Different

### ✨ Unique Capabilities

* **Smart Title Extraction**: Advanced algorithm that reconstructs fragmented titles (handles cases like "RFP: R", "quest f" → "RFP: Request for Proposal...")
* **Hybrid Processing**: Intelligent fallback from text extraction to OCR for optimal performance
* **Multi-Language OCR**: Supports 12+ languages with Tesseract language packs
* **1-Based Page Numbering**: User-friendly page references starting from 1
* **Text Summarization**: Automatically shortens long headings for better readability
* **Total Page Count**: Includes document metadata in output
* **Performance Optimized**: Sub-10 second processing for 50-page PDFs

### 🌍 Language Detection & Support

**Supported Languages with Confidence Scoring:**

* **English** (en) 🇺🇸
* **Dutch** (hi) IN
* **French** (fr) 🇫🇷
* **Spanish** (es) 🇪🇸
* **German** (de) 🇩🇪
* **Italian** (it) 🇮🇹
* **Portuguese** (pt) 🇵🇹
* **Russian** (ru) 🇷🇺
* **Chinese Simplified** (zh-cn) 🇨🇳
* **Japanese** (ja) 🇯🇵
* **Korean** (ko) 🇰🇷
* **Arabic** (ar) 🇸🇦
* **Dutch** (nl) 🇳🇱
* **And 55+ more languages**

**Detection Features:**

* Primary language detection with confidence scores (0-1)
* Multi-language document support
* Additional language detection for mixed documents
* Fallback handling for insufficient text

## 🏗️ Project Structure

```
Challenge_1a/
├── process_pdfs.py              # Main processing engine
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── debug_titles.py             # Title extraction testing
├── test_language_detection.py  # Language detection testing
├── README.md                   # This documentation
└── sample_dataset/
    ├── pdfs/                   # Input PDF samples
    │   ├── file01.pdf
    │   ├── file02.pdf
    │   ├── file03.pdf
    │   ├── hindi_pdf.pdf
    │   └── japanese_sample.pdf
    ├── outputs/                # Generated JSON outputs
    │   ├── file01.json
    │   ├── sample_english.json
    │   └── sample_french.json
    └── schema/
        └── output_schema.json  # JSON validation schema
```

## 🛠️ Technical Approach

### Processing Pipeline

1. **PDF Analysis**: PyMuPDF for fast text extraction
2. **OCR Fallback**: Tesseract OCR with multi-language support for image-based PDFs
3. **Smart Title Extraction**: Pattern-based algorithm that handles fragmented titles
4. **Language Detection**: langdetect library with confidence scoring
5. **Heading Detection**: Hierarchical outline generation (H1-H6)
6. **Text Summarization**: Automatic shortening of long text passages
7. **JSON Output**: Schema-validated structured data export

### Models & Libraries Used

**PDF Processing:**

* `PyMuPDF (fitz)` - Fast PDF text extraction
* `pdfplumber` - Advanced table and structure handling
* `PyPDF2` - PDF metadata and structure analysis
* `pdf2image` - PDF to image conversion for OCR
* `pytesseract` - OCR engine integration

**Language Detection:**

* `langdetect` - Statistical language detection
* `polyglot` - Enhanced multi-language support

**Text Processing:**

* `spaCy` - NLP processing and text analysis
* `en_core_web_sm` - English language model

**Infrastructure:**

* `loguru` - Structured logging
* `jsonschema` - Output validation
* `numpy` - Numerical computations
* `Pillow (PIL)` - Image processing

## 🐳 How to Build and Run

1) First put all the pdfs in sample_dataset/pdfs/ 

### Build the Docker Image

```bash
docker build --platform linux/amd64 -t binary-brains .
```

### Run with Sample Data

#### Linux / macOS / WSL / Git Bash

```bash
docker run --rm \
  -v $(pwd)/sample_dataset/pdfs:/app/input:ro \
  -v $(pwd)/sample_dataset/outputs:/app/output \
  --network none \
  binary-brains
```

#### Windows PowerShell

```powershell
docker run --rm `
  -v "${PWD}/sample_dataset/pdfs:/app/input:ro" `
  -v "${PWD}/sample_dataset/outputs:/app/output" `
  --network none `
  binary-brains
```

# Get the desired output json files in sample_dataset/outputs

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python process_pdfs.py

# Test title extraction
python debug_titles.py

# Test language detection
python test_language_detection.py
```

## 📋 Output Format

The solution generates JSON files with enhanced structure:

```json
{
  "title": "RFP:Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library",
  "total_pages": 14,
  "language": {
    "primary": "en",
    "confidence": 0.999,
    "additional": [
      {
        "language": "fr",
        "confidence": 0.015
      }
    ]
  },
  "outline": [
    {
      "level": "H1",
      "text": "OVERVIEW OF ODL FUNDING MODEL",
      "page": 9
    },
    {
      "level": "H2",
      "text": "Executive Summary",
      "page": 2
    },
    {
      "level": "H3",
      "text": "Primary Goals",
      "page": 3
    }
  ]
}
```

### Field Descriptions

* **title**: Intelligently extracted document title
* **total pages**: Total number of pages in the PDF
* **language.primary**: ISO 639-1 language code (e.g., "en", "fr", "es")
* **language.confidence**: Detection confidence score (0-1)
* **language.additional**: Other detected languages with confidence scores
* **outline**: Hierarchical document structure with 1-based page numbers

## ⚡ Performance Specifications

### Constraints Met

* ✅ **Execution Time**: ≤ 10 seconds for 50-page PDFs
* ✅ **Model Size**: ≤ 200MB total footprint
* ✅ **Network**: Zero internet dependencies during runtime
* ✅ **Architecture**: AMD64 CPU compatible
* ✅ **Resources**: Optimized for 8 CPUs and 16GB RAM

### Performance Features

* **Parallel OCR Processing**: Multi-threaded image processing
* **Memory Optimization**: Efficient PDF handling for large documents
* **Smart Caching**: Reuses extracted text for multiple analysis steps
* **Resource Monitoring**: Built-in performance logging and metrics

## 🔧 Advanced Features

### Smart Title Extraction Algorithm

* **Fragment Recovery**: Reconstructs titles split across multiple lines
* **Pattern Recognition**: Identifies RFP, proposal, and document type patterns
* **Multi-Strategy Approach**: Falls back through multiple extraction methods
* **Content Cleaning**: Removes artifacts and formatting issues

### Intelligent Text Processing

* **Heading Detection**: Recognizes H1-H6 hierarchical structures
* **Text Summarization**: Shortens long passages while preserving meaning
* **Language-Aware Processing**: Adapts to different text structures
* **Schema Validation**: Ensures consistent output format

## 🚦 Testing & Validation

### Included Test Scripts

```bash
# Test title extraction accuracy
python debug_titles.py

# Test language detection on sample texts  
python test_language_detection.py
```

### Validation Checklist

* ✅ All PDFs in input directory processed
* ✅ JSON output files generated for each PDF
* ✅ Output format matches schema validation
* ✅ Processing completes within time constraints
* ✅ Language detection accuracy > 95% for clear text
* ✅ Title extraction handles fragmented cases

## 📝 License & Usage

This project uses only open-source libraries and tools, ensuring full compliance with open-source requirements. Suitable for commercial and academic use.
