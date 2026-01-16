"""
Main entry point for RAG PDF Agent.
Usage: python main.py <path_to_pdf>
"""
import os
import sys
import argparse
from pathlib import Path
from ingest import PDFIngestor
from retriever import VectorRetriever
from chat import RAGChatAgent


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="RAG Conversational Agent for PDF Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ./document.pdf
  python main.py ./data/earnings_report.pdf --model gemini-1.5-pro
  python main.py ./contract.pdf --top-k 3 --no-debug

Environment Variables:
  GEMINI_API_KEY: Your Google Gemini API key (required)
        """
    )
    
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF document to analyze"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-pro",
        choices=["gemini-1.5-flash", "gemini-1.5-pro"],
        help="Gemini model to use (default: gemini-1.5-pro)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Size of text chunks in characters ~500 tokens (default: 2000)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=400,
        help="Overlap between chunks in characters ~100 tokens (default: 400)"
    )
    
    # Debug mode is ALWAYS enabled per requirements
    # parser.add_argument(
    #     "--no-debug",
    #     action="store_true",
    #     help="Disable retrieval debug output"
    # )
    
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached index if available (faster startup)"
    )
    
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if pdf_path.suffix.lower() != '.pdf':
        print(f"Error: File must be a PDF document: {pdf_path}")
        sys.exit(1)
    
    # Check for API key (now optional - will run in fallback mode if not available)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n" + "="*80)
        print("WARNING: GEMINI_API_KEY environment variable not set")
        print("="*80)
        print("\nRunning in FALLBACK MODE (retrieval-only, no LLM generation)")
        print("You will see retrieved document chunks without AI-generated answers.")
        print("\nTo enable full AI responses, set your Gemini API key:")
        print("  Windows (PowerShell): $env:GEMINI_API_KEY='your-key-here'")
        print("  Windows (CMD): set GEMINI_API_KEY=your-key-here")
        print("  Linux/Mac: export GEMINI_API_KEY='your-key-here'")
        print("\nGet your API key at: https://makersuite.google.com/app/apikey")
        print("="*80 + "\n")
    
    print("="*80)
    print("RAG PDF CONVERSATIONAL AGENT")
    print("="*80)
    print(f"Document: {pdf_path.name}")
    print(f"Model: {args.model}")
    print(f"Retrieval: Top-{args.top_k} chunks")
    print(f"Temperature: 0.1 (strict grounding)")
    print(f"Debug mode: ON (always enabled)")
    print("="*80)
    
    # Define index directory
    index_dir = Path("data") / "index" / pdf_path.stem
    
    # Check if we can use cached index
    use_existing_index = False
    if args.use_cache and index_dir.exists():
        faiss_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.pkl"
        if faiss_path.exists() and chunks_path.exists():
            use_existing_index = True
            print(f"\n✓ Using cached index from: {index_dir}")
    
    # Initialize retriever
    retriever = VectorRetriever()
    
    if use_existing_index:
        # Load existing index
        retriever.load_index(str(index_dir))
    else:
        # Step 1: Ingest PDF
        print("\n[STEP 1/3] Ingesting PDF...")
        ingestor = PDFIngestor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        chunks = ingestor.ingest_pdf(str(pdf_path))
        
        if not chunks:
            print("Error: No text extracted from PDF")
            sys.exit(1)
        
        # Step 2: Build vector index
        print("\n[STEP 2/3] Building vector index...")
        retriever.build_index(chunks)
        
        # Save index for future use
        retriever.save_index(str(index_dir))
    
    # Step 3: Start chat agent
    print("\n[STEP 3/3] Initializing chat agent...")
    agent = RAGChatAgent(
        retriever=retriever,
        gemini_api_key=api_key,  # Can be None for fallback mode
        model_name=args.model,
        top_k_retrieval=args.top_k,
        debug_mode=True  # Always enabled per requirements
    )
    
    # Start interactive chat loop
    agent.chat_loop()


if __name__ == "__main__":
    main()
