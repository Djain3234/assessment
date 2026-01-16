"""
Example: Programmatic usage of the RAG PDF Agent
This shows how to use the agent programmatically instead of CLI.
"""
import os
from ingest import PDFIngestor
from retriever import VectorRetriever
from chat import RAGChatAgent


def example_programmatic_usage():
    """Example of using the agent programmatically."""
    
    # Setup
    pdf_path = "./document.pdf"
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        return
    
    # Step 1: Ingest PDF
    print("Ingesting PDF...")
    ingestor = PDFIngestor(chunk_size=800, chunk_overlap=200)
    chunks = ingestor.ingest_pdf(pdf_path)
    
    # Step 2: Build index
    print("Building index...")
    retriever = VectorRetriever()
    retriever.build_index(chunks)
    
    # Step 3: Create agent
    agent = RAGChatAgent(
        retriever=retriever,
        gemini_api_key=api_key,
        model_name="gemini-1.5-flash",
        top_k_retrieval=5,
        debug_mode=True
    )
    
    # Step 4: Ask questions
    questions = [
        "What is the main topic of this document?",
        "Can you summarize the key findings?",
        "What are the recommendations?"
    ]
    
    for question in questions:
        print(f"\n{'='*80}")
        print(f"Q: {question}")
        print(f"{'='*80}")
        answer = agent.answer_query(question)
        print(f"A: {answer}\n")


def example_batch_questions():
    """Example of processing multiple documents."""
    
    # Assume index is already built
    retriever = VectorRetriever()
    retriever.load_index("./data/index/document")
    
    api_key = os.getenv("GEMINI_API_KEY")
    agent = RAGChatAgent(
        retriever=retriever,
        gemini_api_key=api_key,
        debug_mode=False  # Disable debug for batch processing
    )
    
    # Batch questions
    questions = [
        "What is the revenue?",
        "Who is the CEO?",
        "What are the main products?"
    ]
    
    results = []
    for q in questions:
        answer = agent.answer_query(q)
        results.append({"question": q, "answer": answer})
    
    # Process results
    for r in results:
        print(f"Q: {r['question']}")
        print(f"A: {r['answer']}\n")


if __name__ == "__main__":
    example_programmatic_usage()
