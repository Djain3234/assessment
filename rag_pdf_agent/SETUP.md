"""
SETUP AND TESTING GUIDE
=======================

This document provides step-by-step instructions to set up and test the RAG PDF Agent.

PREREQUISITES
-------------
1. Python 3.8 or higher installed
2. Google Gemini API key (get one at: https://makersuite.google.com/app/apikey)
3. A PDF document to test with

INSTALLATION STEPS
------------------

Step 1: Navigate to project directory
```
cd rag_pdf_agent
```

Step 2: Create a virtual environment (recommended)
```
# Windows
python -m venv venv
.\\venv\\Scripts\\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

Step 3: Install dependencies
```
pip install -r requirements.txt
```

This will install:
- google-generativeai (Gemini API client)
- pypdf (PDF text extraction)
- sentence-transformers (embeddings)
- faiss-cpu (vector search)
- numpy (numerical operations)
- tqdm (progress bars)

Step 4: Set your Gemini API key

Windows PowerShell:
```
$env:GEMINI_API_KEY='your-api-key-here'
```

Windows CMD:
```
set GEMINI_API_KEY=your-api-key-here
```

Linux/Mac:
```
export GEMINI_API_KEY='your-api-key-here'
```

TESTING THE SYSTEM
------------------

Test 1: Basic Run
```
python main.py path/to/your/document.pdf
```

Expected output:
- PDF extraction progress bar
- Chunking progress bar
- Embedding generation progress
- FAISS index building
- Chat prompt

Test 2: With Options
```
python main.py document.pdf --model gemini-1.5-pro --top-k 3
```

Test 3: Using Cached Index (faster subsequent runs)
```
python main.py document.pdf --use-cache
```

EXAMPLE QUESTIONS TO ASK
------------------------

General questions:
- "What is this document about?"
- "Summarize the main points"
- "What are the key findings?"

Specific queries:
- "What was the revenue in Q4?"
- "Who are the key stakeholders mentioned?"
- "What are the recommendations?"

Follow-up questions (tests multi-turn capability):
- "What was the revenue?" → "How does it compare to last year?"
- "Who is the CEO?" → "What is their background?"

UNDERSTANDING THE OUTPUT
------------------------

1. Retrieval Debug Section:
```
[RETRIEVAL DEBUG]
Rank 1: Score: 0.8234 | Page: 13 | Chunk ID: 42
Text: "Q4 revenue reached $2.5 billion..."
```
- Shows top-k retrieved chunks
- Score: cosine similarity (0-1, higher is better)
- Page and Chunk ID for traceability

2. Grounded Answer:
```
Assistant: Q4 revenue reached $2.5 billion, representing a 15% 
year-over-year increase. [p13:c42]

Evidence:
"Q4 revenue reached $2.5 billion, up 15% year-over-year, 
driven by strong performance in our cloud services division."
```
- Short answer with citations
- Evidence section with quotes
- Citations in [p{page}:c{chunk}] format

3. Not Found Response:
```
Assistant: Not found in the document.
```
- Appears when answer is not in retrieved chunks
- Prevents hallucinations

TROUBLESHOOTING
---------------

Issue: "GEMINI_API_KEY environment variable not set"
Solution: Set the API key as shown in Step 4

Issue: "PDF file not found"
Solution: Use absolute path or correct relative path

Issue: "No text extracted from PDF"
Solution: PDF might be scanned/image-based. Use OCR preprocessing.

Issue: Import errors
Solution: Ensure all dependencies installed: pip install -r requirements.txt

Issue: FAISS errors on ARM Mac
Solution: Use faiss-cpu or build from source

ADVANCED USAGE
--------------

Adjust chunk size and overlap:
```
python main.py doc.pdf --chunk-size 1000 --chunk-overlap 300
```

Use different models:
```
python main.py doc.pdf --model gemini-1.5-pro
```

Disable debug output:
```
python main.py doc.pdf --no-debug
```

Programmatic usage:
See example_usage.py for Python API examples

PERFORMANCE NOTES
-----------------

First run:
- Extracts PDF text
- Generates embeddings
- Builds FAISS index
- Takes 30-60 seconds for typical documents

Subsequent runs with --use-cache:
- Loads pre-built index
- Starts in ~5 seconds

Memory usage:
- ~500MB for embedding model
- ~100-500MB for document index (varies by size)

LIMITATIONS
-----------

1. PDF extraction: Works best with text-based PDFs
2. Scanned PDFs: Requires OCR preprocessing
3. Images/tables: Text extraction only, no image understanding
4. Document size: Tested up to 500 pages
5. Language: Optimized for English (model supports others)

NEXT STEPS
----------

- Try with your own documents
- Adjust retrieval parameters (top-k, chunk size)
- Experiment with different Gemini models
- Integrate into your application using the Python API

For issues or questions, refer to README.md
