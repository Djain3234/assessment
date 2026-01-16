"""
Vector retrieval module using FAISS.
Builds index from chunks and performs similarity search.
"""
import os
import pickle
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ingest import PDFChunk


class VectorRetriever:
    """Handles vector embedding and similarity search."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the retriever with an embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.chunks = []
        
    def build_index(self, chunks: List[PDFChunk]) -> None:
        """
        Build FAISS index from PDF chunks.
        
        Args:
            chunks: List of PDFChunk objects to index
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunks list")
        
        self.chunks = chunks
        
        # Generate embeddings
        print(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        print(f"Index built successfully with {self.index.ntotal} vectors.")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[PDFChunk, float]]:
        """
        Retrieve top-k most similar chunks for a query.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of (PDFChunk, similarity_score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True
        )
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return chunks with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save_index(self, index_dir: str) -> None:
        """
        Save the FAISS index and chunks to disk.
        
        Args:
            index_dir: Directory to save index files
        """
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(index_dir, "faiss.index")
        faiss.write_index(self.index, index_path)
        
        # Save chunks
        chunks_path = os.path.join(index_dir, "chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Index saved to {index_dir}")
    
    def load_index(self, index_dir: str) -> None:
        """
        Load the FAISS index and chunks from disk.
        
        Args:
            index_dir: Directory containing index files
        """
        # Load FAISS index
        index_path = os.path.join(index_dir, "faiss.index")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        chunks_path = os.path.join(index_dir, "chunks.pkl")
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Index loaded from {index_dir} ({len(self.chunks)} chunks)")
    
    def print_retrieval_debug(self, query: str, results: List[Tuple[PDFChunk, float]]) -> None:
        """
        Print debug information about retrieved chunks.
        
        Args:
            query: The user query
            results: List of (PDFChunk, score) tuples
        """
        print("\n" + "="*80)
        print(f"RETRIEVAL DEBUG - Query: '{query}'")
        print("="*80)
        
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n[Rank {i}] Score: {score:.4f}")
            print(f"Page: {chunk.page_number} | Chunk ID: {chunk.chunk_id} | Citation: {chunk.get_citation()}")
            print(f"Text snippet: {chunk.text[:200]}...")
            print("-" * 80)


if __name__ == "__main__":
    # Test the retriever
    from ingest import PDFIngestor
    import sys
    
    if len(sys.argv) > 1:
        # Ingest PDF
        ingestor = PDFIngestor()
        chunks = ingestor.ingest_pdf(sys.argv[1])
        
        # Build index
        retriever = VectorRetriever()
        retriever.build_index(chunks)
        
        # Test retrieval
        test_query = "What is the main topic?"
        results = retriever.retrieve(test_query, top_k=3)
        retriever.print_retrieval_debug(test_query, results)
