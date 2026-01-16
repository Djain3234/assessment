"""
Conversational chat module with Gemini API.
Implements multi-turn Q&A with grounded responses.
"""
import os
from typing import List, Tuple, Optional
import google.generativeai as genai
from retriever import VectorRetriever
from prompt import (
    SYSTEM_INSTRUCTION,
    build_grounded_prompt,
    rewrite_query_with_history,
    parse_response_for_citations,
    is_not_found_response
)


class RAGChatAgent:
    """Conversational RAG agent using Gemini."""
    
    def __init__(
        self, 
        retriever: VectorRetriever,
        gemini_api_key: Optional[str],
        model_name: str = "gemini-1.5-pro",
        top_k_retrieval: int = 5,
        debug_mode: bool = True
    ):
        """
        Initialize the chat agent.
        
        Args:
            retriever: VectorRetriever instance with built index
            gemini_api_key: Google Gemini API key (optional - runs in fallback mode if None)
            model_name: Gemini model to use (gemini-1.5-pro or gemini-1.5-flash)
            top_k_retrieval: Number of chunks to retrieve
            debug_mode: Whether to print retrieval debug info
        """
        self.retriever = retriever
        self.top_k = top_k_retrieval
        self.debug_mode = debug_mode
        self.api_key = gemini_api_key
        self.fallback_mode = gemini_api_key is None
        
        # Configure Gemini only if API key is available
        if not self.fallback_mode:
            try:
                genai.configure(api_key=gemini_api_key)
                generation_config = genai.types.GenerationConfig(
                    temperature=0.1,  # Very low temperature for strict grounding
                )
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config
                )
                print(f"Chat agent initialized with model: {model_name}")
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini API: {e}")
                print("Falling back to retrieval-only mode...")
                self.fallback_mode = True
                self.model = None
        else:
            self.model = None
            print("Chat agent initialized in FALLBACK MODE (retrieval-only)")
        
        # Conversation history: list of (user_query, assistant_response) tuples
        self.conversation_history: List[Tuple[str, str]] = []
    
    def _rewrite_query_if_needed(self, query: str) -> str:
        """
        Rewrite follow-up questions to be standalone using conversation history.
        
        Args:
            query: User's current query
            
        Returns:
            Rewritten standalone query
        """
        # Skip rewrite in fallback mode or if no history
        if self.fallback_mode or not self.conversation_history:
            return query
        
        # Simple heuristic: if query is short or has pronouns, rewrite it
        pronouns = ["it", "they", "them", "this", "that", "these", "those", "he", "she"]
        needs_rewrite = (
            len(query.split()) < 5 or
            any(pronoun in query.lower().split() for pronoun in pronouns)
        )
        
        if not needs_rewrite:
            return query
        
        try:
            rewrite_prompt = rewrite_query_with_history(query, self.conversation_history)
            response = self.model.generate_content(rewrite_prompt)
            rewritten = response.text.strip()
            
            if self.debug_mode:
                print(f"\n[QUERY REWRITE]")
                print(f"Original: {query}")
                print(f"Rewritten: {rewritten}")
            
            return rewritten
        except Exception as e:
            print(f"Warning: Query rewrite failed: {e}")
            return query
    
    def answer_query(self, user_query: str) -> str:
        """
        Answer a user query with grounded response.
        
        Args:
            user_query: User's question
            
        Returns:
            Grounded response with citations
        """
        # Step 1: Rewrite query if it's a follow-up
        standalone_query = self._rewrite_query_if_needed(user_query)
        
        # Step 2: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(standalone_query, top_k=self.top_k)
        
        # Step 3: Print debug info
        if self.debug_mode:
            self.retriever.print_retrieval_debug(standalone_query, retrieved_chunks)
        
        # Step 4: Handle fallback mode (no LLM)
        if self.fallback_mode:
            answer = self._generate_fallback_response(user_query, retrieved_chunks)
        else:
            # Step 5: Build grounded prompt
            prompt = build_grounded_prompt(
                query=user_query,  # Use original query in prompt
                retrieved_chunks=retrieved_chunks,
                conversation_history=self.conversation_history
            )
            
            # Step 6: Generate response with Gemini
            try:
                full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{prompt}"
                response = self.model.generate_content(full_prompt)
                answer = response.text.strip()
            except Exception as e:
                print(f"\n⚠️  API Error: {e}")
                print("Falling back to retrieval-only mode for this query...\n")
                answer = self._generate_fallback_response(user_query, retrieved_chunks)
        
        # Step 7: Add to conversation history
        self.conversation_history.append((user_query, answer))
        
        return answer
    
    def _generate_fallback_response(self, query: str, chunks: List[Tuple]) -> str:
        """
        Generate a simple response using only retrieved chunks (no LLM).
        
        Args:
            query: User's question
            chunks: Retrieved document chunks as list of (PDFChunk, score) tuples
            
        Returns:
            Response with retrieved content
        """
        if not chunks:
            return "[FALLBACK MODE] No relevant content found in the document."
        
        response = "[FALLBACK MODE - Retrieval Only]\n\n"
        response += f"Found {len(chunks)} relevant passages:\n\n"
        
        for i, (chunk, score) in enumerate(chunks, 1):
            response += f"--- Passage {i} (Page {chunk.page_number}, Similarity: {score:.3f}) ---\n"
            response += chunk.text[:500]  # Show first 500 chars
            if len(chunk.text) > 500:
                response += "...\n"
            else:
                response += "\n"
            response += "\n"
        
        response += "\n💡 Note: Set GEMINI_API_KEY to get AI-generated answers instead of raw passages."
        return response
    
    def chat_loop(self) -> None:
        """
        Interactive chat loop.
        """
        print("\n" + "="*80)
        print("RAG CONVERSATIONAL AGENT")
        print("="*80)
        print("Ask questions about the document. Type 'quit' or 'exit' to end.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                # Get answer
                answer = self.answer_query(user_input)
                
                # Print answer
                print(f"\nAssistant: {answer}")
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def reset_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def get_history(self) -> List[Tuple[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()


if __name__ == "__main__":
    # Test mode - requires index to be built first
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chat.py <index_dir>")
        sys.exit(1)
    
    index_dir = sys.argv[1]
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Load retriever
    retriever = VectorRetriever()
    retriever.load_index(index_dir)
    
    # Start chat
    agent = RAGChatAgent(retriever, api_key)
    agent.chat_loop()
