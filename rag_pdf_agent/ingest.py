"""
PDF ingestion module.
Extracts text from PDF, chunks it with overlap, and stores metadata.
"""
import os
from typing import List, Dict
from pypdf import PdfReader
from tqdm import tqdm


class PDFChunk:
    """Represents a text chunk with metadata."""
    def __init__(self, text: str, page_number: int, chunk_id: int):
        self.text = text
        self.page_number = page_number
        self.chunk_id = chunk_id
        
    def __repr__(self):
        return f"PDFChunk(page={self.page_number}, chunk_id={self.chunk_id}, text_len={len(self.text)})"
    
    def get_citation(self) -> str:
        """Returns citation format [p{page}:c{chunk_id}]"""
        return f"[p{self.page_number}:c{self.chunk_id}]"


class PDFIngestor:
    """Handles PDF text extraction and chunking."""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        """
        Initialize the PDF ingestor.
        
        Args:
            chunk_size: Target size of each chunk in characters (~500 tokens)
            chunk_overlap: Number of overlapping characters (~100 tokens)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF page by page.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries with page_number and text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        pages = []
        
        print(f"Extracting text from {len(reader.pages)} pages...")
        for page_num, page in enumerate(tqdm(reader.pages, desc="Reading PDF"), start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    'page_number': page_num,
                    'text': text
                })
        
        print(f"Successfully extracted {len(pages)} pages with text.")
        return pages
    
    def chunk_text(self, text: str, page_number: int, start_chunk_id: int) -> List[PDFChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            page_number: Page number for metadata
            start_chunk_id: Starting chunk ID
            
        Returns:
            List of PDFChunk objects
        """
        chunks = []
        chunk_id = start_chunk_id
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append(PDFChunk(
                    text=chunk_text.strip(),
                    page_number=page_number,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
            
            # Move forward by (chunk_size - overlap)
            start += (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def ingest_pdf(self, pdf_path: str) -> List[PDFChunk]:
        """
        Complete ingestion pipeline: extract pages and chunk text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of all PDFChunk objects
        """
        pages = self.extract_text_from_pdf(pdf_path)
        
        all_chunks = []
        chunk_id = 0
        
        print("Chunking text with overlap...")
        for page_data in tqdm(pages, desc="Chunking"):
            page_chunks = self.chunk_text(
                text=page_data['text'],
                page_number=page_data['page_number'],
                start_chunk_id=chunk_id
            )
            all_chunks.extend(page_chunks)
            chunk_id += len(page_chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(pages)} pages.")
        return all_chunks


if __name__ == "__main__":
    # Test the ingestor
    import sys
    if len(sys.argv) > 1:
        ingestor = PDFIngestor()
        chunks = ingestor.ingest_pdf(sys.argv[1])
        print(f"\nSample chunks:")
        for chunk in chunks[:3]:
            print(f"\n{chunk}")
            print(f"Text preview: {chunk.text[:100]}...")
