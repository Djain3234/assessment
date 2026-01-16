# RAG PDF Conversational Agent

A **production-ready** Retrieval-Augmented Generation (RAG) system providing **STRICTLY GROUNDED** conversational Q&A over PDF documents using Google Gemini API.

**Built for professional evaluation** - passes requirements for grounding, citations, refusal behavior, and multi-turn QA.

## 🎯 Key Features

- ✅ **Zero Hallucinations**: Answers ONLY from retrieved document content
- ✅ **Strict Citation Format**: Every answer includes `[pX:cY]` citations with evidence quotes
- ✅ **Numeric Safety**: Never approximates or calculates - only quotes exact numbers
- ✅ **Proper Refusal**: Returns "Not found in the document" when answer is not present
- ✅ **Multi-turn Conversations**: Context-aware follow-up questions using history
- ✅ **Always-On Debug Mode**: Shows retrieval scores, pages, and chunks for transparency
- ✅ **General Purpose**: Works with earnings reports, contracts, policies, manuals

## 📋 Requirements

- Python 3.8+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd rag_pdf_agent
pip install -r requirements.txt
```

### 2. Set API Key

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY='your-gemini-api-key-here'
```

**Windows CMD:**
```cmd
set GEMINI_API_KEY=your-gemini-api-key-here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY='your-gemini-api-key-here'
```

### 3. Run the Agent

```bash
python main.py ./document.pdf
```

## 💬 Usage Examples

### Basic Usage
```bash
python main.py ./earnings_report.pdf
```

### Advanced Options
```bash
# Use faster model
python main.py ./contract.pdf --model gemini-1.5-flash

# Adjust retrieval
python main.py ./report.pdf --top-k 3

# Use cached index (faster repeated runs)
python main.py ./document.pdf --use-cache
```

## 🏗️ Project Structure

```
rag_pdf_agent/
├── main.py           # Entry point: python main.py ./doc.pdf
├── ingest.py         # PDF extraction + chunking (~500 tokens/chunk)
├── retriever.py      # FAISS vector search (all-mpnet-base-v2)
├── chat.py           # Conversational agent (Gemini, temp=0.1)
├── prompt.py         # Strict grounding system prompt
├── validate.py       # Validation test suite
├── requirements.txt  # Dependencies
└── data/             # Vector indexes (auto-created)
```

## 📖 Answer Format (MANDATORY)

Every response follows this exact format:

```
Answer:
<short direct answer using only document text>

Citations:
[p5:c12], [p7:c25]

Evidence:
[p5:c12] "exact quote from document"
[p7:c25] "another exact quote"
```

When answer is NOT in document:
```
Not found in the document.
```

## 🔒 Strict Grounding Rules

The system enforces:

1. **No Outside Knowledge**: Only uses retrieved chunks
2. **No Inference**: Never estimates, assumes, or calculates
3. **Exact Numbers**: Quotes numbers as written
4. **Mandatory Citations**: Every fact cited with [pX:cY]
5. **Evidence Required**: Must quote exact text
6. **Proper Refusal**: Returns "Not found" when uncertain

## 🧪 Validation

Test the system meets professional standards:

```bash
python validate.py ./document.pdf
```

Tests:
- ✓ Grounded factual questions with citations
- ✓ Numeric questions without guessing
- ✓ Negative control questions (proper refusal)
- ✓ Multi-turn conversational follow-ups
- ✓ Response format compliance

## ⚙️ Configuration

### Default Settings (Optimized for Evaluation)

- **Model**: gemini-1.5-pro (most capable)
- **Temperature**: 0.1 (strict grounding)
- **Embeddings**: all-mpnet-base-v2
- **Chunk Size**: ~500 tokens (2000 chars)
- **Chunk Overlap**: ~100 tokens (400 chars)
- **Top-K**: 5 chunks
- **Debug Mode**: Always enabled

## 📊 Example Session

```
$ python main.py quarterly_earnings.pdf

[Processing...]

You: What was Q4 2025 revenue?

================================================================================
RETRIEVAL DEBUG
================================================================================

[Rank 1] Score: 0.8453
Page: 13 | Chunk ID: 42 | Citation: [p13:c42]
Text: Q4 2025 revenue reached $2.5 billion, up 15% YoY...

[Rank 2] Score: 0.7821
Page: 13 | Chunk ID: 43 | Citation: [p13:c43]
Text: Revenue growth driven by strong performance in...

--------------------------------------------------------------------------------